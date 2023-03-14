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
        action = random.randint(1, 16)
        return action