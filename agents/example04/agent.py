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

        self.dxy = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self.parent = [[None for _ in range(79)] for _ in range(21)]
        self.children = [[None for _ in range(79)] for _ in range(21)]
        self.visited = [[False for _ in range(79)] for _ in range(21)]
        self.goal = (None, None)
        self.dungeon_lv = 1

        self.last_pos = None
        self.frozen_cnt = 0
    
    def new_lv(self):
        self.parent = [[None for _ in range(79)] for _ in range(21)]
        self.children = [[None for _ in range(79)] for _ in range(21)]
        self.visited = [[False for _ in range(79)] for _ in range(21)]
        self.goal = (None, None)

        self.last_pos = None
        self.frozen_cnt = 0

    def get_action(self, env, obs):
        cur_lv = obs['blstats'][12]
        time = obs['blstats'][20]
        if self.dungeon_lv != cur_lv or time == 1:
            self.dungeon_lv = cur_lv
            self.new_lv()

        x, y = obs['blstats'][:2]
        screen = obs['tty_chars']

        action = None
            
        if self.last_pos == (x, y):
            self.frozen_cnt += 1
            if self.frozen_cnt > 10:
                self.frozen_cnt = 0
                action = env.action_space.sample()
                
                return action
        else:
            self.frozen_cnt = 0
        self.last_pos = (x, y)

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
        elif (y, x) == self.goal:
            action = 18
        else:
            pre_map = self.preprocess_map(obs)
            self.visited[y][x] = True

            if self.children[y][x] == None:
                py, px = None, None
                if self.parent[y][x] != None:
                    dxy_i = self.parent[y][x] - 1 #부모 노드가 있다면 부모 노드의 위치 기억
                    py = y + self.dxy[dxy_i][0]
                    px = x + self.dxy[dxy_i][1]
                
                children_list = []
                for i in range(4):
                    ny = y + self.dxy[i][0]
                    nx = x + self.dxy[i][1]

                    #범위 안이면서, 이동 가능하면서, 부모 노드가 아니고, 방문하지 않았다면 자식 노드
                    is_child = (0 <= ny < 21 and 0 <= nx < 79) and pre_map[ny][nx] and (ny, nx) != (py, px) and not self.visited[ny][nx]
                    children_list.append(is_child)

                    #자식 노드에 부모 노드에 대한 정보 추가
                    if is_child:
                        self.parent[ny][nx] = (i+2)%4 + 1
                
                #해당 좌표에 자식 리스트 추가
                self.children[y][x] = children_list
            
            for i in range(4):
                ny = y + self.dxy[i][0]
                nx = x + self.dxy[i][1]
                if (0 <= ny < 21 and 0 <= nx < 79) and self.children[y][x][i] and not self.visited[ny][nx]:
                    action = i+1
                    return action
            action = self.parent[y][x]

        #더 이상 탐색할 곳이 없다면 계단 올라가기
        if action == None:
            action = 17

        return action
        
    def preprocess_map(self, obs):
        pre = []

        available = [ord('.'), ord('#')]
        unavailable = [ord(' '), ord('`')]
        door_or_wall = [ord('|'), ord('-')]

        chars = obs['chars']
        colors = obs['colors']
        for y in range(21):
            pre_line = []
            for x in range(79):
                char = chars[y][x]
                color = colors[y][x]
                
                pre_char = True
                if char in unavailable:
                    pre_char = False
                elif char in door_or_wall and color == 7:
                    pre_char = False
                elif char == ord('#') and color == 6:
                    pre_char = False
                elif char == ord('>'):
                    self.goal = (y, x)
                pre_line.append(pre_char)
            pre.append(pre_line)
        return pre