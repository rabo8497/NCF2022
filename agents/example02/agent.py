import random

import gym
from ExampleAgent import ExampleAgent

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

    def get_action(self, env, obs):
        x, y = obs['blstats'][:2]
        screen = obs['tty_chars']
        if self.is_more(screen):
            action = 0
        elif self.is_yn(screen):
            action = 8
        elif self.is_locked(screen):
            action = 20
        elif self.asking_direction(screen):
            original_map = obs['chars'] 
            
            north = original_map[y-1][x]
            east = original_map[y][x+1]
            south = original_map[y+1][x]
            west = original_map[y][x-1]

            if north == ord('+'):
                action = 1
            elif east == ord('+'):
                action = 2
            elif south == ord('+'):
                action = 3
            elif west == ord('+'):
                action = 4
        else:
            action = random.randint(1, 16)
        
        return action