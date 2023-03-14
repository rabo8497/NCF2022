import random
import torch
from torch.nn import functional as F
from torch import nn
import numpy as np
import math
import pathlib

import gym

from ExampleAgent import ExampleAgent
from .a2c_lstm import A2C_LSTM
from collections import deque

from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent(ExampleAgent):
    def __init__(self, FLAGS=None):
        super().__init__(FLAGS)

        self.num_envs = 32
        self.max_steps_per_episode = 80

        self.gamma = 0.999
        self.closs_coef = 0.5
        self.eloss_coef = 0.0006

        self.a2c_lstm = A2C_LSTM().to(device)
        self.optimizer = torch.optim.Adam(self.a2c_lstm.parameters())
        
        self.path = pathlib.Path(__file__).parent / "policy.pt"
        if self.flags.mode != 'train':
            self.a2c_lstm.load_state_dict(torch.load(self.path))

            self.env = gym.make(
                FLAGS.env,
                savedir=FLAGS.savedir,
                max_episode_steps=FLAGS.max_steps,
                allow_all_yn_questions=True,
                allow_all_modes=True,
            )

            self.h_t = torch.zeros(1, 512).clone().to(device) #lstm cell의 dimension과 맞춰준다.
            self.c_t = torch.zeros(1, 512).clone().to(device) #lstm cell의 dimension과 맞춰준다.
        else:
            self.env = gym.vector.make(
                FLAGS.env,
                savedir=FLAGS.savedir,
                max_episode_steps=FLAGS.max_steps,
                allow_all_yn_questions=True,
                allow_all_modes=True,
                num_envs=self.num_envs,
            )

            self.h_t = torch.zeros(self.num_envs, 512).clone().to(device) #lstm cell의 dimension과 맞춰준다.
            self.c_t = torch.zeros(self.num_envs, 512).clone().to(device) #lstm cell의 dimension과 맞춰준다.


    def get_action(self, env, obs):
        actor, critic = self.get_actor_critic(env, obs)

        if self.flags.mode != 'train':
            action = torch.multinomial(F.softmax(actor, dim=1), num_samples=1)
            action = self.get_real_action(action, obs['tty_chars'], obs['chars'], obs['blstats'])       
            return action
        else:
            action = [torch.multinomial(F.softmax(actor_, dim=0), num_samples=1) for actor_ in actor]

            return torch.LongTensor(action)

    def get_real_action(self, action_, screen, original_map, blstats):
        action = action_
        if self.is_more(screen):
            action = 0
        elif self.is_yn(screen):
            action = 8
        elif self.is_locked(screen):
            action = 20
        elif self.asking_direction(screen):
            x, y = blstats[:2]

            direction = [(-1,0),(0,1),(1,0),(0,-1),(-1,1),(1,1),(1,-1),(-1,-1)]

            for i in range(8):
                ny = y + direction[i][0]
                nx = x + direction[i][1]
                if 0 <= ny < 21 and 0 <= nx < 79 and original_map[ny][nx] == ord('+'):
                    action = i + 1
                    break
            
        return action
    
    def get_actor_critic(self, env, obs):
        if self.flags.mode != 'train':
            observed_glyphs = torch.from_numpy(obs['glyphs']).float().unsqueeze(0).to(device)
            observed_stats = torch.from_numpy(obs['blstats']).float().unsqueeze(0).to(device)

            with torch.no_grad():
                actor, critic, self.h_t, self.c_t = self.a2c_lstm(observed_glyphs, observed_stats, self.h_t, self.c_t)
        else:
            observed_glyphs = torch.from_numpy(obs['glyphs']).float().to(device)
            observed_stats = torch.from_numpy(obs['blstats']).float().to(device)

            actor, critic, self.h_t, self.c_t = self.a2c_lstm(observed_glyphs, observed_stats, self.h_t, self.c_t)
        return actor, critic
    
    def optimize_td_loss(self, actors, actions, critics, returns):
        actors = torch.cat(actors).to(device)
        actions = torch.cat(actions).to(device)

        returns = torch.cat(returns)
        critics = torch.cat(critics)        
        advantages = returns - critics

        #compute actor loss
        cross_entropy = F.nll_loss(
            F.log_softmax(actors, dim=-1),
            target=torch.flatten(actions),
            reduction="none",
        )
        cross_entropy = cross_entropy.view_as(advantages)
        actor_loss = torch.sum(cross_entropy * advantages.detach())

        #compute critic loss
        critic_loss = 0.5 * torch.sum(advantages**2)

        #compute entropy loss
        policy = F.softmax(actors, dim=-1)
        log_policy = F.log_softmax(actors, dim=-1)
        entropy_loss = torch.sum(policy * log_policy)
        
        loss = actor_loss + self.closs_coef * critic_loss + self.eloss_coef * entropy_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.a2c_lstm.parameters(), 40.0)
        self.optimizer.step()

    def train(self):
        writer = SummaryWriter()
        
        env = self.env
        
        num_episodes = 0
        episode_scores = deque([], maxlen=100)
        episode_dungeonlv = deque([], maxlen=100)
        episode_explv = deque([], maxlen=100)
        episode_steps = deque([], maxlen=100)

        time_step = 0
        
        obs = env.reset()
        while time_step < self.flags.max_steps:
            actors, actions, critics, rewards, dones = [], [], [], [], []

            for mini_step in range(self.max_steps_per_episode):
                old_score, old_dlv, old_elv, old_steps = [], [], [], []
                for blstats in obs['blstats']:
                    old_score.append(blstats[9])
                    old_dlv.append(blstats[12])
                    old_elv.append(blstats[18])
                    old_steps.append(blstats[20])

                action = self.get_action(env, obs)
                actor, critic = self.get_actor_critic(env, obs)

                screen = obs['tty_chars']
                original_map = obs['chars']
                blstats = obs['blstats']
                real_actions = []
                real_action_reward = []
                for i in range(self.num_envs):
                    real_action = self.get_real_action(action[i], screen[i], original_map[i], blstats[i])
                    real_actions.append(real_action)
                    if real_action == action[i]:
                        real_action_reward.append(0)
                    else:
                        real_action_reward.append(0.02)
                real_action_reward = np.array(real_action_reward)

                real_actions = torch.LongTensor(real_actions)
                new_obs, reward, done, info = env.step(real_actions)
                
                done_ = torch.from_numpy(np.expand_dims(done, axis=1)).float().to(device)
                self.h_t, self.c_t = self.h_t*(1.0 - done_), self.c_t*(1.0 - done_)

                actors.append(actor)
                actions.append(action)
                critics.append(critic.squeeze())
                rewards.append(np.tanh((reward + real_action_reward)/100))
                dones.append(np.array(done, dtype=int))

                for i in range(self.num_envs):
                    if done[i]:
                        num_episodes += 1
                        episode_scores.append(old_score[i])
                        episode_dungeonlv.append(old_dlv[i])
                        episode_explv.append(old_elv[i])
                        episode_steps.append(old_steps[i])
                obs = new_obs

                if mini_step == self.max_steps_per_episode-1:
                    time_step += self.num_envs*self.max_steps_per_episode

                    with torch.no_grad():
                        _, new_critic = self.get_actor_critic(env, new_obs)

                        returns = []
                        r = new_critic.squeeze()
                        for t in reversed(range(len(rewards))):
                            reward = torch.from_numpy(rewards[t]).to(device)
                            done = torch.from_numpy(dones[t]).to(device)

                            r = reward + self.gamma * r * (1.0 - done)
                            returns.insert(0, r)

                    self.optimize_td_loss(actors, actions, critics, returns)

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
                    
                    torch.save(self.a2c_lstm.state_dict(), self.path)