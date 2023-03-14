import random
import torch
from torch.nn import functional as F
from torch import nn
import numpy as np
import math
import pathlib

import gym

from ExampleAgent import ExampleAgent
from .dqn import ReplayBuffer, DQN
from collections import deque

from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent(ExampleAgent):
    def __init__(self, FLAGS=None):
        super().__init__(FLAGS)

        self.env = gym.make(
            FLAGS.env,
            savedir=FLAGS.savedir,
            max_episode_steps=FLAGS.max_steps,
            allow_all_yn_questions=True,
            allow_all_modes=True,
        )

        self.buffer_size = 5000
        self.batch_size = 64
        self.update_freq = 10
        self.gamma = 0.999

        self.policy = DQN().to(device)
        self.target = DQN().to(device)
        self.buffer = ReplayBuffer(self.buffer_size)
        self.optimizer = torch.optim.Adam(self.policy.parameters())
        
        self.path = pathlib.Path(__file__).parent / "policy.pt"
        if self.flags.mode != 'train':
            self.policy.load_state_dict(torch.load(self.path))

    def get_action(self, env, obs):
        observed_glyphs = torch.from_numpy(obs['glyphs']).float().unsqueeze(0).to(device)
        observed_stats = torch.from_numpy(obs['blstats']).float().unsqueeze(0).to(device)

        with torch.no_grad():
            q = self.policy(observed_glyphs, observed_stats)
        
        _, action = q.max(1)
        return action.item()
    
    def optimize_td_loss(self):
        glyphs, stats, next_glyphs, next_stats, actions, rewards, dones = self.buffer.sample(self.batch_size)
        glyphs = torch.from_numpy(glyphs).float().to(device)
        stats = torch.from_numpy(stats).float().to(device)
        next_glyphs = torch.from_numpy(next_glyphs).float().to(device)
        next_stats = torch.from_numpy(next_stats).float().to(device)
        actions = torch.from_numpy(actions).long().to(device)
        rewards = torch.from_numpy(rewards).float().to(device)
        dones = torch.from_numpy(dones).float().to(device)

        with torch.no_grad():
            q_next = self.policy(next_glyphs, next_stats)
            _, action_next = q_next.max(1)
            q_next_max = self.target(next_glyphs, next_stats)
            q_next_max = q_next_max.gather(1, action_next.unsqueeze(1)).squeeze()
        
        q_target = rewards + (1 - dones) * self.gamma * q_next_max
        q_curr = self.policy(glyphs, stats)
        q_curr = q_curr.gather(1, actions.unsqueeze(1)).squeeze()

        loss = F.smooth_l1_loss(q_curr, q_target)
        self.optimizer.zero_grad()
        loss.backward()
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
        max_steps_per_episode = 32*80

        obs = env.reset()
        while time_step < self.flags.max_steps:
            log_probs, critics, rewards, dones, entropies = [], [], [], [], []

            for mini_step in range(max_steps_per_episode):
                old_score = obs['blstats'][9]
                old_dlv = obs['blstats'][12]
                old_elv = obs['blstats'][18]
                old_steps = obs['blstats'][20]

                eps_threshold = math.exp(-len(episode_scores)/50)
                if random.random() <= eps_threshold:
                    action = env.action_space.sample()
                else:
                    action = self.get_action(env, obs)

                new_obs, reward, done, info = env.step(action)

                if done:
                    num_episodes += 1
                    episode_scores.append(old_score)
                    episode_dungeonlv.append(old_dlv)
                    episode_explv.append(old_elv)
                    episode_steps.append(old_steps)

                    obs = env.reset()
                else:
                    self.buffer.push(obs['glyphs'],
                                    obs['blstats'],
                                    new_obs['glyphs'],
                                    new_obs['blstats'],
                                    action,
                                    np.tanh(reward/100),
                                    float(done))
                    obs = new_obs
                
                if time_step > self.batch_size:
                    self.optimize_td_loss()

                if time_step % self.update_freq == 0:
                    self.target.load_state_dict(self.policy.state_dict())
                    torch.save(self.policy.state_dict(), self.path)
                
                if mini_step == max_steps_per_episode-1:
                    time_step += max_steps_per_episode

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