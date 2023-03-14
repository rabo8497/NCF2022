import argparse
import ast
import contextlib
import os
import random
import termios
import time
import timeit
import tty
import importlib
import statistics

import gym

import nle  # noqa: F401
from nle import nethack

class ExampleAgent():
    def __init__(self, FLAGS=None):
        self.flags = FLAGS

    def go_back(self, num_lines):
        print("\033[%dA" % num_lines)

    def get_action(self):
        raise NotImplementedError('Should implement get_action function')

    def run_episodes(self):
        env = self.env
        
        obs = env.reset()

        steps = 0
        episodes = 0
        reward = 0.0
        action = None

        mean_sps = 0
        mean_reward = 0.0

        total_start_time = timeit.default_timer()
        start_time = total_start_time

        scores = []

        while True:
            # NetHack interface render
            if not self.flags.no_render:
                print("-" * 8 + " " * 71)
                print(f"Previous reward: {str(reward):64s}")
                act_str = repr(env._actions[action]) if action is not None else ""
                print(f"Previous action: {str(act_str):64s}")
                print("-" * 8)
                env.render(self.flags.render_mode)
                print("-" * 8)
                print(obs["blstats"])
                self.go_back(num_lines=33)
            
            # NLE는 done=True일 때 obs가 초기화되기 때문에, step 이전에 score를 저장
            score = obs['blstats'][9]

            action = self.get_action(env, obs)

            if action is None:
                break

            obs, reward, done, info = env.step(action)

            # lstm 모델의 경우 h_t, c_t를 매 episode마다 초기화
            if self.flags.use_lstm:
                self.h_t, self.c_t = self.h_t*(1.0 - done), self.c_t*(1.0 - done)
            steps += 1

            if not done and steps < self.flags.max_steps:
                continue

            time_delta = timeit.default_timer() - start_time

            print("End status:", info["end_status"].name)
            print("Final Score:", score)
            scores.append(score)

            sps = steps / time_delta
            print("Episode: %i. Steps: %i. SPS: %f" % (episodes, steps, sps))

            episodes += 1
            mean_sps += (sps - mean_sps) / episodes

            start_time = timeit.default_timer()

            steps = 0

            if episodes == self.flags.ngames:
                break
            
            obs = env.reset()

        env.close()
        ret = "Finished after %i episodes and %f seconds, Mean sps: %f, Avg score: %f, Median score: %f" % (episodes, timeit.default_timer() - total_start_time, mean_sps, sum(scores)/episodes, statistics.median(scores))
        print(ret)
    
    def is_more(self, screen):
        for line in screen:
            interp = ''
            for letter in line:
                interp += chr(letter)

                if 'More' in interp:
                    return True
        return False

    def is_yn(self, screen):
        for line in screen:
            interp = ''
            for letter in line:
                interp += chr(letter)
            
            if '[yn]' in interp:
                return True
        return False

    def is_locked(self, screen):
        for line in screen:
            interp = ''
            for letter in line:
                interp += chr(letter)
            
            if 'locked' in interp:
                return True
        return False
    
    def asking_direction(self, screen):
        for line in screen:
            interp = ''
            for letter in line:
                interp += chr(letter)
            
            if 'direction?' in interp:
                return True
        return False

    def preprocess_map(self, obs):
        raise NotImplementedError('Should implement preprocess_map if you need.')
    
    def evaluate(self, seed):
        env = self.env
        
        env.seed(seed, seed)
        obs = env.reset()

        steps = 0
        episodes = 0
        reward = 0.0
        action = None

        mean_sps = 0
        mean_reward = 0.0

        total_start_time = timeit.default_timer()
        start_time = total_start_time

        while True:
            score = obs['blstats'][9]

            action = self.get_action(env, obs)

            if action is None:
                break

            obs, reward, done, info = env.step(action)
            steps += 1

            if not done and steps < self.flags.max_steps:
                continue

            time_delta = timeit.default_timer() - start_time
            sps = steps / time_delta
            
            break

        env.close()

        ret = "Finished after %f seconds, sps: %f, score: %f" % (timeit.default_timer() - total_start_time, sps, score)
        print(ret)

        return ret