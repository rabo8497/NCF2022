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
            pre_map = self.preprocess_map(obs)

            north = pre_map[y-1][x]
            east = pre_map[y][x+1]
            south = pre_map[y+1][x]
            west = pre_map[y][x-1]

            if north:
                action = 1
            elif east:
                action = 2
            elif south:
                action = 3
            elif west:
                action = 4
        
        return action

    def preprocess_map(self, obs):
        pre_map = []

        unavailable = [ord(' '), ord('`')]
        door_or_wall = [ord('|'), ord('-')]

        chars = obs['chars']
        colors = obs['colors']
        for y in range(21):
            pre_line = []
            for x in range(79):
                char = chars[y][x]
                color = colors[y][x]
                
                pre_char = True #pre_char이 True면 해당 좌표에 이동할 수 있다.
                if char in unavailable:
                    pre_char = False #공백과 바위가 있는 곳으로는 이동할 수 없다.
                elif char in door_or_wall and color == 7:
                    pre_char = False #해당 좌표의 문자가 회색(color == 7) | 혹은 -라면 벽이므로 이동할 수 없다.
                elif char == ord('#') and color == 6:
                    pre_char = False # bar의 경우 이동할 수 없다.
                pre_line.append(pre_char)
            pre_map.append(pre_line)
        return pre_map