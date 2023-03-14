from tabnanny import check
from ExampleAgent import ExampleAgent
from .a2c import A2C
from .player_agent import PlayerAgent
from .buffer import Buffer
import torch
from torch.nn import functional as F
import gym
from collections import deque
import numpy as np
import pathlib

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from torch.utils.tensorboard import SummaryWriter

class Agent(ExampleAgent):
    def __init__(self, FLAGS=None):
        super().__init__(FLAGS)

        self.a2c = A2C().to(device)
        
        self.optimizer = torch.optim.Adam(self.a2c.parameters())

        self.gamma = 0.999
        self.closs_coeff = 0.5
        self.eloss_coeff = 0.0001
        self.num_envs = 16
        self.path = pathlib.Path(__file__).parent / "policy.pt"
        #self.a2c.load_state_dict(torch.load(self.path))
        if self.flags.mode != 'train':
            self.num_envs = 1
            self.a2c.load_state_dict(torch.load(self.path))
            self.env = gym.make(
                FLAGS.env,
                savedir=FLAGS.savedir,
                max_episode_steps=FLAGS.max_steps,
                allow_all_yn_questions=True,
                allow_all_modes=True,
            )

            self.player_agent = PlayerAgent(FLAGS, self.a2c)

        else:
            self.num_envs = 32
            self.env = gym.vector.make(
                FLAGS.env,
                savedir=FLAGS.savedir,
                max_episode_steps=FLAGS.max_steps,
                allow_all_yn_questions=True,
                allow_all_modes=True,
                num_envs=self.num_envs,                
            )
        
            self.player_agent = [PlayerAgent(FLAGS, self.a2c) for _ in range(self.num_envs)]


    def get_action(self, env, obs):



        if self.flags.mode != 'train':
            return self.player_agent.get_action(env, obs)

        else:
            actions = []
            nn_usages = []
            for i in range(self.num_envs):
                action, usage = self.player_agent[i].get_action_train(env, obs, i)
                actions.append(action)
                nn_usages.append(usage)

            return torch.LongTensor(actions), nn_usages

    def get_actor_critic(self, env, obs):
        observed_glyphs = torch.from_numpy(obs['glyphs']).float().unsqueeze(0).to(device)
        observed_stats = torch.from_numpy(obs['blstats']).float().unsqueeze(0).to(device)

        actor, critic = self.a2c(observed_glyphs, observed_stats)
        return actor, critic

    def get_multiple_actor_critic(self, env, obss):
        obs_glyphs = [obs['glyphs'] for obs in obss]
        obs_stats = [obs['blstats'] for obs in obss]
        obs_glyphs = torch.FloatTensor(np.array(obs_glyphs)).to(device)
        obs_stats = torch.FloatTensor(np.array(obs_stats)).to(device)
        actor, critic = self.a2c(obs_glyphs, obs_stats)

        return actor, critic

    def optimize(self, actors, actions, rewards, obss, new_obss, dones):
        #make a return
        #actors = torch.cat(actors, dim=0).to(device)
        actions = torch.stack(actions, dim=0).to(device)
        actors, value = self.get_multiple_actor_critic(None, obss)
        _, new_value = self.get_multiple_actor_critic(None, new_obss)

        with torch.no_grad():
            returns = []
            r = new_value[-1]
            for t in reversed(range(len(rewards))):
                #reward = torch.from_numpy(rewards[t]).to(device)
                #done = torch.from_numpy(dones[t]).to(device)
                r = rewards[t] + self.gamma * (1.0 - dones[t]) * r
                returns.insert(0, r)
        returns = torch.stack(returns, dim=0)

        advantages = returns - value

        cross_entropy = F.nll_loss(
            F.log_softmax(actors, dim=-1),
            target=torch.flatten(actions),
            reduction='none',
        )
        cross_entropy = cross_entropy.view_as(advantages)
        actor_loss = torch.sum(cross_entropy * advantages.detach())

        critic_loss = 0.5 * torch.sum(advantages ** 2)

        policy = F.softmax(actors, dim=-1)
        log_policy = F.log_softmax(actors, dim=-1)
        entropy_loss = torch.sum(policy * log_policy)

        loss = actor_loss + self.closs_coeff * critic_loss + self.eloss_coeff * entropy_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.a2c.parameters(), 40.0)
        self.optimizer.step()

    def get_my_obs(self, obs, i):
        my_obs = {}
        for key in obs:
            my_obs[key] = obs[key][i]
        return my_obs

    def train(self):
        writer = SummaryWriter()
        
        env = self.env

        num_episodes = 0
        episode_scores = deque([], maxlen=100)
        episode_dungeonlv = deque([], maxlen=100)
        episode_explv = deque([], maxlen=100)
        episode_steps = deque([], maxlen=100)

        buffer = Buffer(self.num_envs)

        time_step = 0
        checker = 0
        obs = env.reset()

        while time_step < self.flags.max_steps:
            old_score, old_dlv, old_elv, old_steps = [], [], [], []
            for blstats in obs['blstats']:
                old_score.append(blstats[9])
                old_dlv.append(blstats[12])
                old_elv.append(blstats[18])
                old_steps.append(blstats[20])

            action, usages = self.get_action(env, obs)

            new_obs, reward, done, info = env.step(action)


            for i in range(self.num_envs):
                if usages[i]:
                    #save data
                    obs_i = self.get_my_obs(obs, i)
                    new_obs_i = self.get_my_obs(new_obs, i)
                    actor,_ = self.get_actor_critic(env, obs_i)
                    buffer.add(actor, action[i], np.tanh(reward[i]), obs_i, new_obs_i, np.array(done[i], dtype=int), i, checker)

                if done[i]:
                    num_episodes += 1
                    episode_scores.append(old_score[i])
                    episode_dungeonlv.append(old_dlv[i])
                    episode_explv.append(old_elv[i])
                    episode_steps.append(old_steps[i])

            obs = new_obs
            checker += 1
            print(buffer.record_length)
            if buffer.record_length >= 32 * 40:
                time_step += 32 * 40

                r_actor, r_action, r_reward, r_obs, r_new_obs, r_done = buffer.make_train_data(32 * 40)

                self.optimize(r_actor, r_action, r_reward, r_obs, r_new_obs, r_done)
                
                print("Elapsed Steps: {}%".format((time_step)/self.flags.max_steps*100))
                print("Episodes: {}".format(num_episodes))
                print("Last 100 Episode Mean Score: {}".format(sum(episode_scores)/len(episode_scores) if episode_scores else 0))
                print("Last 100 Episode Mean Dungeon Lv: {}".format(sum(episode_dungeonlv)/len(episode_dungeonlv) if episode_dungeonlv else 1))
                print("Last 100 Episode Mean Exp Lv: {}".format(sum(episode_explv)/len(episode_explv) if episode_explv else 1))
                print("Last 100 Episode Mean Step: {}".format(sum(episode_steps)/len(episode_steps) if episode_steps else 0))
                print()
                
                writer.add_scalar('Last 100 Episode Mean Score', sum(episode_scores)/len(episode_scores) if episode_scores else 0, time_step+1)
                writer.add_scalar('Last 100 Episode Mean Dungeon Lv', sum(episode_dungeonlv)/len(episode_dungeonlv) if episode_dungeonlv else 1, time_step+1)
                writer.add_scalar('Last 100 Episode Mean Exp Lv', sum(episode_explv)/len(episode_explv) if episode_explv else 1, time_step+1)
                writer.add_scalar('Last 100 Episode Mean Step', sum(episode_steps)/len(episode_steps) if episode_steps else 0, time_step+1)

                torch.save(self.a2c.state_dict(), self.path)
                # for i in range(self.num_envs):
                #     self.player_agent[i].combat_agent.a2c.load_state_dict(self.a2c.state_dict())