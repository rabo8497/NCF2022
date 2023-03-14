import sys
import gym
import copy
from re import T
from time import sleep
from queue import PriorityQueue
from collections import deque
import nle.nethack as nh
import numpy as np

import random
import torch
from ExampleAgent import ExampleAgent
from .combat_agent import CombatAgent

INF = 987654321
DEBUG = False

wall_set = set(['|','-'])
walkable_objects = set(['.','#','<','>','+','@','$',')','[','%','?','/','=','!','(','"','*',])
#walkable_objects = set(['.','#','<','>','+','$','^',')','[','%','?','=','!','"','*',])

log = open("log.txt", "w")
    
def is_monster(letter):
    if letter >= 65 and letter <= 90:
        return True
    elif letter >= 97 and letter <= 122:
        return True
    elif letter == ord(":"):
        return True
    elif letter == ord("Z") :
        return True
    else:
        return False

def is_enemy(original_map, color_map, pos):
    letter = original_map[pos[1]][pos[0]]
    if color_map[pos[1]][pos[0]] == 15:
        return False
    else:
        return is_monster(letter)

def find_closest_enemy(obs, pos):
    original_map = obs['chars']
    color_map = obs['colors']
    x, y = obs['blstats'][:2]
    enemy_pos_list = []
    for j in range(21):
        for i in range(max(0, x - 10), min(79, x + 11)):
            if manhattan_distance(pos,(i,j)) >= 10:
                continue
            if is_enemy(original_map, color_map, (i, j)):
                enemy_pos_list.append((i, j))
    if not enemy_pos_list:
        return None, None
    enemy_dist = [abs(i - pos[0]) + abs(j - pos[1]) for (i, j) in enemy_pos_list] ## L1 dist
    min_dist = min(enemy_dist)
    return enemy_pos_list[enemy_dist.index(min_dist)], min_dist

def is_walkable(original_map, pos):
    letter = original_map[pos[1]][pos[0]]
    if chr(letter) in walkable_objects:
        return True
    elif is_monster(letter):
        return True
    else:
        return False

def is_corridor(original_map, pos):
    letter = original_map[pos[1]][pos[0]]
    if chr(letter) == '.':
        return False
    else:
        return is_walkable(original_map, pos)

def find_room_size(original_map, pos):
    x, y = pos
    min_x = x
    min_x_found = False
    max_x = x
    max_x_found = False
    min_y = y
    min_y_found = False
    max_y = y
    max_y_found = False
    while(True):
        if not min_x_found:
            min_x -= 1
            count_list = []
            for j in range(y - 1, y + 2):
                count_list.append(is_walkable(original_map, (min_x, j)))
            count = count_list.count(False)
            if (count >= 2):
                min_x += 1
                min_x_found = True
        if not max_x_found:
            max_x += 1
            count_list = []
            for j in range(y - 1, y + 2):
                count_list.append(is_walkable(original_map, (max_x, j)))
            count = count_list.count(False)
            if (count >= 2):
                max_x -= 1
                max_x_found = True
        if not min_y_found:
            min_y -= 1
            count_list = []
            for i in range(x - 1, x + 2):
                count_list.append(is_walkable(original_map, (i, min_y)))
            count = count_list.count(False)
            if (count >= 2):
                min_y += 1
                min_y_found = True
        if not max_y_found:
            max_y += 1
            count_list = []
            for i in range(x - 1, x + 2):
                count_list.append(is_walkable(original_map, (i, max_y)))
            count = count_list.count(False)
            if (count >= 2):
                max_y -= 1
                max_y_found = True
        if min_x_found is True and max_x_found is True and min_y_found is True and max_y_found is True:
            break
    return (min_x, min_y), (max_x, max_y)

def move_dir_to_action(dir):
    if dir[0] == 1:
        if dir[1] == -1:
            return 5
        elif dir[1] == 0:
            return 2
        elif dir[1] == 1:
            return 6
    elif dir[0] == 0:
        if dir[1] == -1:
            return 1
        elif dir[1] == 0:
            return 19
        elif dir[1] == 1:
            return 3
    elif dir[0] == -1:
        if dir[1] == -1:
            return 8
        elif dir[1] == 0:
            return 4
        elif dir[1] == 1:
            return 7
    return 19
            
def is_diagonal_error(screen):
    for line in screen:
        interp = ''
        for letter in line:
            interp += chr(letter)

            if 'diagonally' in interp:
                return True
    return False

def find_in_message(screen, str):
    for i in range(22):
        line = screen[i]
        interp = ''
        for letter in line:
            interp += chr(letter)

            if str in interp:
                return True
    return False

def find_eat(obs, not_eat=[], ran=10) :
    player_pos = (obs['blstats'][0], obs['blstats'][1])
    where = np.where(obs['chars']==37)
    if len(where[0]) == 0 :
        return (False, (None, None))
    distans = []
    for i in range(len(where[0])) :
        eat_pos = (where[1][i], where[0][i])
        if eat_pos in not_eat :
            distans.append(999)
            continue
        distans.append(manhattan_distance(player_pos, eat_pos))
    if len(distans) == 0:
        return (False, (None, None))
    i = distans.index(min(distans))
    if min(distans) > ran :
        return (False, (None, None))
    return (True, (where[1][i], where[0][i]))

def find_zap(obs) :
    player_pos = (obs['blstats'][0], obs['blstats'][1])
    where = np.where(obs['chars']==ord('@'))
    if len(where[0]) <= 1 :
        return False
    elif len(where[0]) >= 2 :
        return True
    else :
        return False

def is_enter_dungeon(obs,pos,visited):
    original_map = obs['chars']
    color_map = obs['colors']
    enter = is_entered_room(obs,pos,visited,1)
    (min_x, min_y), (max_x, max_y) = find_room_size(obs['chars'], pos)

    if(enter==False):
        return False

    mon_cnt = 0
    for i in range(min_x, max_x):
        for j in range(min_y,max_y):
            if is_enemy(original_map, color_map, (i,j)):
                mon_cnt += 1
    if(mon_cnt>=3):
        return True
    else:
        return False

def is_enter_shop(obs,pos,visited):
    original_map = obs['chars']
    enter = is_entered_room(obs,pos,visited,0)
    (min_x, min_y), (max_x, max_y) = find_room_size(obs['chars'], pos)
    keeper_cnt = 0

    if(enter==False):
        return False

    for i in range(min_x, max_x):
        for j in range(min_y,max_y):
            if (original_map[j][i]==ord('@')):
                keeper_cnt += 1
    if(keeper_cnt>=2):
        return True
    else:
        return False

def is_entered_room(obs,pos,visited,shop_or_dungeon):   #shop = 0, dungeon = 1
    original_map = obs['chars']
    y = pos[1]
    x = pos[0]
    if(shop_or_dungeon==0):
        if(visited[y-1][x]==False and original_map[y+1][x]==ord('#') and visited[y+1][x]==True):
            if(original_map[y][x-1]==ord('-') and original_map[y][x+1]==ord('-')):
                return True, (0,-1)
            else:
                return False
        elif(visited[y+1][x]==False and original_map[y-1][x]==ord('#') and visited[y-1][x]==True):
            if(original_map[y][x-1]==ord('-') and original_map[y][x+1]==ord('-')):
                return True, (0,1)
            else:
                return False
        elif(visited[y][x-1]==False and original_map[y][x+1]==ord('#') and visited[y][x+1]==True):
            if(original_map[y-1][x]==ord('|') and original_map[y+1][x]==ord('|')):
                return True, (-1,0)
            elif(original_map[y-1][x]==ord('-') and original_map[y+1][x]==ord('|')):
                return True, (-1,0)
            elif(original_map[y-1][x]==ord('|') and original_map[y+1][x]==ord('-')):
                return True, (-1,0)
            else:
                return False
        elif(visited[y][x+1]==False and original_map[y][x-1]==ord('#') and visited[y][x-1]==True):
            if(original_map[y-1][x]==ord('|') and original_map[y+1][x]==ord('|')):
                return True, (1,0)
            elif(original_map[y-1][x]==ord('-') and original_map[y+1][x]==ord('|')):
                return True, (1,0)
            elif(original_map[y-1][x]==ord('|') and original_map[y+1][x]==ord('-')):
                return True, (1,0)
            else:
                return False
        else:
            return False
    elif(shop_or_dungeon == 1):
        if(visited[y-1][x-1]==False and visited[y-1][x]==False and visited[y-1][x+1]==False and visited[y][x-1]==False and visited[y][x+1]==False and original_map[y+2][x]==ord('#') and visited[y+2][x]==True):
            if(original_map[y+1][x-1]==ord('-') and original_map[y+1][x+1]==ord('-')):
                return True
            else:
                return False
        elif(visited[y+1][x-1]==False and visited[y+1][x]==False and visited[y+1][x+1]==False and visited[y][x-1]==False and visited[y][x+1]==False and original_map[y-2][x]==ord('#') and visited[y-2][x]==True):
            if(original_map[y-1][x-1]==ord('-') and original_map[y-1][x+1]==ord('-')):
                return True
            else:
                return False
        elif(visited[y-1][x]==False and visited[y-1][x-1]==False and visited[y][x-1]==False and visited[y+1][x-1]==False and visited[y+1][x]==False and original_map[y][x+2]==ord('#') and visited[y][x+2]==True):
            if(original_map[y-1][x+1]==ord('|') and original_map[y+1][x+1]==ord('|')):
                return True
            elif(original_map[y-1][x+1]==ord('-') and original_map[y+1][x+1]==ord('|')):
                return True
            elif(original_map[y-1][x+1]==ord('|') and original_map[y+1][x+1]==ord('-')):
                return True
            else:
                return False
        elif(visited[y-1][x]==False and visited[y-1][x+1]==False and visited[y][x+1]==False and visited[y+1][x+1]==False and visited[y+1][x]==False and original_map[y][x-2]==ord('#') and visited[y][x-2]==True):
            if(original_map[y-1][x-1]==ord('|') and original_map[y+1][x-1]==ord('|')):
                return True
            elif(original_map[y-1][x-1]==ord('-') and original_map[y+1][x-1]==ord('|')):
                return True
            elif(original_map[y-1][x-1]==ord('|') and original_map[y+1][x-1]==ord('-')):
                return True
            else:
                return False
        else:
            return False

def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def last_act_lst_oppo(last) :
    #print(last)
    result = []
    for t in range(len(last)) :
        if last[t] == 1 :
            result.append(3)
        elif last[t] == 2 :
            result.append(4)
        elif last[t] == 3 :
            result.append(1)
        elif last[t] == 4 :
            result.append(2)
        elif last[t] == 5 :
            result.append(7)
        elif last[t] == 6 :
            result.append(8)
        elif last[t] == 7 :
            result.append(5)
        elif last[t] == 8 :
            result.append(6)
        else :
            continue
    result.reverse()
    return result


class Floor():
    def __init__(self, flags):
        self.flags = flags
        self.room = []
        self.corridor = []
        self.dungeon = deque() #한 방에 몬스터의 수가 3 이상일때 던전으로 판단
        self.current_place = None
        self.parent = [[None for _ in range(79)] for _ in range(21)]
        self.children = [[None for _ in range(79)] for _ in range(21)]
        self.visited = [[False for _ in range(79)] for _ in range(21)]
        self.occ_map = [[1/(21*79) for _ in range(79)] for _ in range(21)]
        self.search_number = [[0 for _ in range(79)] for _ in range(21)]
        self.search_completed = False
        self.goal = None
        self.search_pos = None
        self.upst = []
        self.memory_upst = [[] for o in range(10)]

        self.last_pos = None
        self.frozen_cnt = 0
        #self.dxy = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self.dxy = [(-1, 0), (0, 1), (1, 0), (0, -1), (-1, 1), (1, 1), (1, -1), (-1, -1)]
        self.opposite = [3, 4, 1, 2, 7, 8, 5, 6]
        #self.opposite = [3, 4, 1, 2, 5, 8, 7, 6]
        #4 3 2 1 7 6 5 8 
    def block(self,obs,pos,shop_or_dungeon):
        original_map = obs['chars']
        y = pos[1]
        x = pos[0]
        if(shop_or_dungeon==0):
            if(original_map[y-1][x]==ord('|') and original_map[y+1][x]==ord('|') and self.visited[y][x-1]):
                return (1,0),(1,1),(1,-1)
            elif(original_map[y-1][x]==ord('-') and original_map[y+1][x]==ord('|') and self.visited[y][x-1]):
                return (1,0),(1,1),(1,-1)
            elif(original_map[y-1][x]==ord('|') and original_map[y+1][x]==ord('-') and self.visited[y][x-1]):
                return (1,0),(1,1),(1,-1)
            elif(original_map[y-1][x]==ord('|') and original_map[y+1][x]==ord('|') and self.visited[y][x+1]):
                return (-1,0),(-1,1),(-1,-1)
            elif(original_map[y-1][x]==ord('-') and original_map[y+1][x]==ord('|') and self.visited[y][x+1]):
                return (-1,0),(-1,1),(-1,-1)
            elif(original_map[y-1][x]==ord('|') and original_map[y+1][x]==ord('-') and self.visited[y][x+1]):
                return (-1,0),(-1,1),(-1,-1)
            elif(original_map[y][x-1]==ord('-') and original_map[y][x+1]==ord('-') and self.visited[y-1][x]):
                return (0,1),(1,1),(-1,1)
            elif(original_map[y][x-1]==ord('-') and original_map[y][x+1]==ord('-') and self.visited[y+1][x]):
                return (0,-1),(1,-1),(-1,-1)
        elif(shop_or_dungeon==1):
            if(original_map[y-1][x-1]==ord('|') and original_map[y+1][x-1]==ord('|') and self.visited[y][x-1]):
                return (1,0),(1,1),(1,-1),(0,-1),(0,1)
            elif(original_map[y-1][x-1]==ord('-') and original_map[y+1][x-1]==ord('|') and self.visited[y][x-1]):
                return (1,0),(1,1),(1,-1),(0,-1),(0,1)
            elif(original_map[y-1][x-1]==ord('|') and original_map[y+1][x-1]==ord('-') and self.visited[y][x-1]):
                return (1,0),(1,1),(1,-1),(0,-1),(0,1)
            elif(original_map[y-1][x+1]==ord('|') and original_map[y+1][x+1]==ord('|') and self.visited[y][x+1]):
                return (-1,0),(-1,1),(-1,-1),(0,-1),(0,1)
            elif(original_map[y-1][x+1]==ord('-') and original_map[y+1][x+1]==ord('|') and self.visited[y][x+1]):
                return (-1,0),(-1,1),(-1,-1),(0,-1),(0,1)
            elif(original_map[y-1][x+1]==ord('|') and original_map[y+1][x+1]==ord('-') and self.visited[y][x+1]):
                return (-1,0),(-1,1),(-1,-1),(0,-1),(0,1)
            elif(original_map[y-1][x-1]==ord('-') and original_map[y-1][x+1]==ord('-') and self.visited[y-1][x]):
                return (0,1),(1,1),(-1,1),(-1,0),(1,0)
            elif(original_map[y+1][x-1]==ord('-') and original_map[y+1][x+1]==ord('-') and self.visited[y+1][x]):
                return (0,-1),(1,-1),(-1,-1),(-1,0),(1,0)

    def search(self, env, obs,d):
        x, y = obs['blstats'][:2]
        screen = obs['tty_chars']

        action = None
        
        #if is_enter_dungeon(obs, (x,y),self.visited):
            #sleep(5)
        #    pos = (x,y)
        #    if pos not in self.dungeon:
        #        self.dungeon.append(pos)
        #    for (i,j) in self.block(obs,pos,1):
        #        obs['chars'][j+y][i+x] = ' '
        #        self.occ_map[j+y][i+x] = 0
        #        self.visited[j+y][i+x] = True

        if is_enter_shop(obs, (x,y),self.visited):
            #sleep(5)
            pos = (x,y)
            for (i,j) in self.block(obs,pos,0):
                obs['chars'][j+y][i+x] = ' '
                self.occ_map[j+y][i+x] = 0
                self.visited[j+y][i+x] = True
        
        if self.last_pos == (x, y):
            self.frozen_cnt += 1
            if self.frozen_cnt > 10:
                self.frozen_cnt = 0
                action = env.action_space.sample()
                
                if self.flags.mode == 'train':
                    for ii, aa in enumerate(action) :
                        if aa == 21 :
                            action[ii] = 19
                else :
                    if action == 21 :
                        action = 19
                if self.flags.mode != 'train':
                    return [action]
                else:
                    return [action[0]]
        else:
            self.frozen_cnt = 0
        self.last_pos = (x, y)
        
        pre_map = self.preprocess_map(obs)
        self.calculate_occupancy_map(obs, pre_map, (x,y))
        self.visited[y][x] = True

        if self.children[y][x] == None:
            py, px = None, None
            if self.parent[y][x] != None:
                dxy_i = self.parent[y][x] - 1 #부모 노드가 있다면 부모 노드의 위치 기억
                py = y + self.dxy[dxy_i][0]
                px = x + self.dxy[dxy_i][1]
            
            children_list = []
            for i in range(8):
                ny = y + self.dxy[i][0]
                nx = x + self.dxy[i][1]

                #범위 안이면서, 이동 가능하면서, 부모 노드가 아니고, 방문하지 않았다면 자식 노드
                is_child = (0 <= ny < 21 and 0 <= nx < 79) and pre_map[ny][nx] and (ny, nx) != (py, px) and not self.visited[ny][nx]

                #재탐색할때 지웠던거 다시 채우기
                is_previous_child = ((0 <= ny < 21 and 0 <= nx < 79) and 
                                    pre_map[ny][nx] and    
                                    (ny, nx) != (py, px) and 
                                    self.parent[ny][nx] != None and 
                                    (y == ny + self.dxy[self.parent[ny][nx] - 1][0] and x == nx + self.dxy[self.parent[ny][nx] - 1][1]))

                is_child = is_child or is_previous_child

                children_list.append(is_child)

                #자식 노드에 부모 노드에 대한 정보 추가
                if is_child:
                    self.parent[ny][nx] = self.opposite[i]
            
            #해당 좌표에 자식 리스트 추가
            self.children[y][x] = children_list
        
        for i in range(8):
            ny = y + self.dxy[i][0]
            nx = x + self.dxy[i][1]
            if (0 <= ny < 21 and 0 <= nx < 79) and self.children[y][x][i] and not self.visited[ny][nx]:
                action = i+1
                return [action]
            
        # 막다른 길에서는 search로 길 찾기 트라이
        wall_check = 4
        for i in range(4):
            ny = y + self.dxy[i][0]
            nx = x + self.dxy[i][1]
            if (0 <= ny < 21 and 0 <= nx < 79) and pre_map[ny][nx] is True:
                wall_check -= 1
        if wall_check >= 3:
            #대각선에 문이 있으면 찾을 때까지 찾는다
            door_or_wall = [ord('|'), ord('-'), ord('+')]
            for i in range(4, 8):   
                ny = y + self.dxy[i][0]
                nx = x + self.dxy[i][1]
                if (0 <= ny < 21 and 0 <= nx < 79) and obs['chars'][ny][nx] in door_or_wall and obs['colors'][ny][nx] != 7:
                    action = 22
            #아니면 5번만 (확률 53%)

            if self.search_number[y][x] < 7:
                action = 22
                self.search_number[y][x] += 1
            else:
                action = self.parent[y][x]
        else:
            action = self.parent[y][x]

        #더 이상 탐색할 곳이 없다면 계단 올라가기
        if action == None and d == 0:
            self.search_completed = True
            action = 19
            #action = 17
        elif action == None and d == 1:
            pass
        return [action]
    
    def preprocess_map(self, obs):
        pre = []

        available = [ord('.'), ord('#')]
        unavailable = [ord(' '), ord('`'), ord('^'), ord('*'), ord('(')]
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
                    self.goal = (x, y)
                elif char == ord('<') :
                    self.upst.append((x,y))
                pre_line.append(pre_char)
            pre.append(pre_line)
        return pre
    
    def navigate(self, obs, now_pos, target_pos):
        pre_map = self.preprocess_map(obs)
        # use A*
        # use 0.5 * L1 length for heuristic
        # return a list of actions to go from now_pos to target_pos
        pq = PriorityQueue()
        visited = [[False for _ in range(79)] for _ in range(21)]
        parent = [[None for _ in range(79)] for _ in range(21)]
        distance = [[INF for _ in range(79)] for _ in range(21)]
        pq.put((0.5 * manhattan_distance(now_pos, target_pos), now_pos))
        distance[now_pos[1]][now_pos[0]] = 0
        while not pq.empty():
            _, (x, y) = pq.get()
            if visited[y][x]:
                continue
            visited[y][x] = True
            if (x, y) == target_pos:
                rx = x
                ry = y
                action_list = []
                while not (rx == now_pos[0] and ry == now_pos[1]):
                    action_list.append(parent[ry][rx])
                    return_dir = self.opposite[parent[ry][rx] - 1]
                    rx = rx + self.dxy[return_dir - 1][1]
                    ry = ry + self.dxy[return_dir - 1][0]
                if not action_list:
                    action_list = [19]
                return action_list
            for i in range(8):
                ny = y + self.dxy[i][0]
                nx = x + self.dxy[i][1]
                if (0 <= ny < 21 and 0 <= nx < 79) and pre_map[ny][nx] and not visited[ny][nx]:
                    parent[ny][nx] = i + 1
                    distance[ny][nx] = distance[y][x] + 1
                    pq.put((0.5 * manhattan_distance((nx, ny), target_pos) + distance[ny][nx], (nx, ny)))
        #not reachable
        return [19]
    
    def calculate_occupancy_map(self, obs, pre_map, pos):
        diff_factor = 0.5
        prob_culled = 0
        if self.visited[pos[1]][pos[0]] is False:
            prob_culled += self.occ_map[pos[1]][pos[0]]
            self.occ_map[pos[1]][pos[0]] = 0
            self.visited[pos[1]][pos[0]] = True
        if (prob_culled > 0):
            for i in range(21):
                for j in range(79):
                    if self.visited[i][j] is False:
                        self.occ_map[i][j] /= 1 - prob_culled
            cur_occ_map = copy.deepcopy(self.occ_map)
            for i in range(21):
                for j in range(79):
                    dirs = [(-1,0),(0,1),(1,0),(0,-1)]
                    neighbour_probs = 0
                    for dir in dirs:
                        try:
                            neighbour_probs += cur_occ_map[i + dir[0]][j + dir[1]]
                        except IndexError:
                            continue
                    self.occ_map[i][j] = (1- diff_factor) * cur_occ_map[i][j] + diff_factor * neighbour_probs / 4
                    if self.visited[i][j]:
                        self.occ_map[i][j] = 0
    
    def calculate_search_position(self):
        pq = PriorityQueue()
        for i in range(21):
            for j in range(79):
                if self.visited[i][j]:
                    dirs = [(-1,0),(0,1),(1,0),(0,-1)]
                    neighbour_probs = 0
                    for dir in dirs:
                        try:
                            neighbour_probs += self.occ_map[i + dir[0]][j + dir[1]]
                        except IndexError:
                            continue
                    if neighbour_probs > 0:
                        pq.put((-neighbour_probs, (j,i)))
        return pq     
       
class PlayerAgent(ExampleAgent):
    def __init__(self, FLAGS, net):
        super().__init__(FLAGS)
        self.net = net
        self.floor = {}
        self.combat_agent = CombatAgent(self.net)
        self.action_list = []
        self.last_action = None
        self.orange_count = 5
        self.last_time = 0
        self.is_nn_used = False
        self.last_pos = None
        
        self.eat_pos = None
        self.key = 0
        self.not_eat = []
        self.pre_eat_pos = None
        self.find_eat_count = 0
        self.last_action_list = deque([], maxlen=20)
        self.zap_key = True

    def reset(self):
        self.floor = {}
        self.combat_agent = CombatAgent(self.net)
        self.action_list = []
        self.last_action = None
        self.orange_count = 5
        self.last_pos = None

        self.eat_pos = None
        self.key = 0
        self.not_eat = []
        self.pre_eat_pos = None
        self.find_eat_count = 0
        self.last_action_list = deque([], maxlen=20)
        self.zap_key = True

    def get_action_train(self, env, obs, i):
        try:
            my_obs = {}
            for key in obs:
                my_obs[key] = obs[key][i]
            action = self.action_select(env, my_obs)
            return (action, self.is_nn_used)
        except Exception as e:
            return (19, False)

    def get_action(self, env, obs):
        try:
            action = self.action_select(env, obs)
            return action
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            #print("exception: ", e)
            #print("line ", exc_tb.tb_lineno)
            return 19

    def action_select_debug(self, env, obs):
        return self.action_select(env, obs)

    def action_select(self, env, obs):
        try:
            self.is_nn_used = False
            time = obs['blstats'][20]
            if time == 1 and self.last_time != 1:
                self.reset()
            self.last_time = time
            #sleep(0.025)
            x, y = obs['blstats'][:2]


            screen = obs['tty_chars']
            original_map = obs['chars']
            if obs['blstats'][12] in self.floor:
                current_floor = self.floor[obs['blstats'][12]]
            else:
                current_floor = Floor(self.flags)
                self.floor[obs['blstats'][12]] = current_floor

            if self.last_pos == (x, y):
                self.frozen_cnt += 1
                if self.frozen_cnt > 10:
                    self.frozen_cnt = 0
                    action = env.action_space.sample()
                    if self.flags.mode == 'train':
                        for ii, aa in enumerate(action) :
                            if aa == 21 :
                                action[ii] = 19
                    else :
                        if action == 21 :
                            action = 19
                    if self.flags.mode != 'train':
                        return action
                    else:
                        return action[0]
            else:
                self.frozen_cnt = 0
            self.last_pos = (x, y)
                    
            if find_in_message(screen, "hidden"):
                current_floor.children[y][x] = None
                current_floor.search_completed = False
                current_floor.search_pos = None
            closest_enemy_pos, min_dist = find_closest_enemy(obs, (x, y))
            if self.combat_agent.battle and closest_enemy_pos is None:
                self.combat_agent.end_battle(obs, (x, y))
        

            #actions
            if find_in_message(screen, 'More'):
                self.action_list.append(0)
            elif find_in_message(screen, 'attack'):
                #a = input()
                self.action_list.append(8)
            elif find_in_message(screen, 'Still climb'):
                self.action_list.append(6)
            elif find_in_message(screen, '[yn]'):
                self.action_list.append(8)
            elif find_in_message(screen, 'locked'):
                self.action_list.append(20)
            elif find_in_message(screen, 'direction?'):
                original_map = obs['chars'] 
                
                north = original_map[y-1][x]
                east = original_map[y][x+1]
                south = original_map[y+1][x]
                west = original_map[y][x-1]
                northeast = original_map[y-1][x+1]
                southeast = original_map[y+1][x+1]
                southwest = original_map[y+1][x-1]
                northwest = original_map[y-1][x-1]

                if north == ord('+'):
                    self.action_list.append(1)
                    self.action_list.append(1)
                elif east == ord('+'):
                    self.action_list.append(2)
                    self.action_list.append(2)
                elif south == ord('+'):
                    self.action_list.append(3)
                    self.action_list.append(3)
                elif west == ord('+'):
                    self.action_list.append(4)
                    self.action_list.append(4)
                if northeast == ord('+'):
                    self.action_list.append(5)
                    self.action_list.append(5)
                elif southeast == ord('+'):
                    self.action_list.append(6)
                    self.action_list.append(6)
                elif southwest == ord('+'):
                    self.action_list.append(7)
                    self.action_list.append(7)
                elif northwest == ord('+'):
                    self.action_list.append(8)
                    self.action_list.append(8)
            elif find_in_message(screen, 'diagonally'):
                action_list = []
                if self.last_action == 5:
                    action_list = [1,2]
                elif self.last_action == 6:
                    action_list = [3,2]
                elif self.last_action == 7:
                    action_list = [3,4]
                elif self.last_action == 8:
                    action_list = [1,4]
                if original_map[y][x-1] == ord('-') or original_map[y][x+1] == ord('-'):
                    action_list.reverse()
                ny = y + current_floor.dxy[self.last_action - 1][0]
                nx = x + current_floor.dxy[self.last_action - 1][1]
                if original_map[ny + 1][nx] == ord('|') or original_map[ny - 1][nx] == ord('|'):
                    action_list.reverse()
                self.action_list += action_list

            elif find_zap(obs) and self.zap_key:
                self.zap_key = False
                #a = input()
                self.action_list = last_act_lst_oppo(self.last_action_list)


            elif not self.combat_agent.battle and min_dist is not None and min_dist < 4 :#and obs['blstats'][14] > obs['blstats'][14]/3:# get into battle
                combat_action, self.is_nn_used = self.combat_agent.start_battle(env, obs, current_floor, closest_enemy_pos)
                self.action_list.append(combat_action)
                #log.write("f")
            
            elif self.combat_agent.battle:                                              # keep battle
                
                original_map = obs['chars']
                color_map = obs['colors']
                
                combat_action, self.is_nn_used = self.combat_agent.get_action(env, obs, current_floor, closest_enemy_pos)
                #sleep(0.5)
                self.action_list.append(combat_action)
                #log.write("f")

            elif self.action_list:                                                      # solve continuations
                self.last_action = self.action_list.pop()
                self.last_action_list.append(self.last_action)
                return self.last_action
            elif obs['blstats'][21] <= 2 and obs['blstats'][10] < obs['blstats'][11] // 3 :
                return 19
            # 바꾼 부분#############
            elif (x, y) == self.eat_pos :
                #a = input()
                self.find_eat_count = 0
                if obs['blstats'][21] > 0 and find_in_message(screen, "see") and find_in_message(screen, "here"):
                    eat_list = ["gnoom", "fox", "rat", "orange", "lizard", "lichen", "egg", "tripe", "slime mold", "food ration", "apple", "cheese", "cake", "cookie", "andwich", "pear", "pie", "ripe rations", "tin", "banana"]
                    if obs['blstats'][21] > 1 :
                        eat_list.append('corpse')
                    not_eat_count = 0
                    if obs['blstats'][21] > 2 :
                        self.not_eat = []
                        self.action_list = [8, 21]
                        self.eat_pos = None
                    for c in eat_list :
                        if find_in_message(screen, c) :
                            self.action_list = [8, 21]
                            self.eat_pos = None
                        else :
                            not_eat_count += 1
                    if not_eat_count == len(eat_list) :
                        self.not_eat.append((x, y))
                        self.action_list.append(random.randrange(1,9))
                    self.eat_pos = None
                else :
                    if obs['blstats'][21] > 2 :
                        self.not_eat = []
                        self.action_list += [8, 21]
                        self.eat_pos = None
                    self.eat_pos = None
                    self.not_eat.append((x, y))
                    self.action_list.append(random.randrange(1,9))

            #elif obs['blstats'][21] > 2 and find_eat(obs, self.not_eat, 10)[0] :
            #    if obs['blstats'][21] > 2 :
            #        self.not_eat = []
            #    if self.eat_pos == None :
            #        self.eat_pos = find_eat(obs, self.not_eat, 10)[1]
            #    else :
            #        if self.find_eat_count > 3 :
            #            self.find_eat_count = 0
            #            self.not_eat.append(self.eat_pos)
            #            self.eat_pos = None
            #        self.find_eat_count += 1
            #    if (x, y) != self.eat_pos and self.eat_pos is not None:
            #        if self.pre_eat_pos == None :
            #            self.pre_eat_pos = (x, y)
                    #print(self.eat_pos)
            #        navigate_ans = current_floor.navigate(obs, (x, y), self.eat_pos)
            #        if navigate_ans is None:
            #            print("navigagte_ans ERROR!")
            #        self.action_list += navigate_ans

            elif obs['blstats'][21] > 0 and find_eat(obs, self.not_eat, 4)[0] :
                if obs['blstats'][21] > 2 :
                    self.not_eat = []
                if self.eat_pos == None :
                    self.eat_pos = find_eat(obs, self.not_eat, 4)[1]
                else :
                    if self.find_eat_count > 3 :
                        self.find_eat_count = 0
                        self.not_eat.append(self.eat_pos)
                        self.eat_pos = None
                    self.find_eat_count += 1
                if (x, y) != self.eat_pos and self.eat_pos is not None:
                    if self.pre_eat_pos == None :
                        self.pre_eat_pos = (x, y)
                    #print(self.eat_pos)
                    navigate_ans = current_floor.navigate(obs, (x, y), self.eat_pos)
                    if navigate_ans is None:
                        print("navigagte_ans ERROR!")
                    self.action_list += navigate_ans    
            
            elif obs['blstats'][21] > 1 and self.orange_count > 0:                     # eat oragne
                #a = input()
                self.action_list += [4, 21]
                self.orange_count -= 1

            elif (current_floor.search_completed == True and len(current_floor.dungeon)>0):
                target = current_floor.dungeon.pop()
                for (i,j) in self.block(obs,(x,y)):
                    obs['chars'][j+y][i+x] = '.'
                navigate_ans = current_floor.navigate(obs, (x, y), target)
                if navigate_ans is None:
                    print("navigagte_ans ERROR!")
                self.action_list += navigate_ans
                while(current_floor.search_completed == False):
                    search_ans = current_floor.search(env, obs,1)
                    if search_ans is None:
                        print("search_ans ERROR!")
                    self.action_list += search_ans
                i = obs['blstats'][:2][0]
                j = obs['blstats'][:2][1]
                navigate_ans = current_floor.navigate(obs, (i,j), (x,y))
                self.action_list += navigate_ans


            elif (current_floor.search_completed == False):
                search_ans = current_floor.search(env, obs,0)
                if search_ans is None:
                    print("search_ans ERROR!")
                self.action_list += search_ans
            else:
                if current_floor.goal is not None:
                    if (x, y) != current_floor.goal:
                        navigate_ans = current_floor.navigate(obs, (x, y), current_floor.goal)
                        if navigate_ans is None:
                            print("navigagte_ans ERROR!")
                        self.action_list += navigate_ans
                    elif (x, y) == current_floor.goal:
                        self.action_list.append(18) # go to next floor
                        current_floor.upst = []
                        current_floor.memory_upst[obs['blstats'][12]+1].append(current_floor.goal)
                elif DEBUG:
                    while(True):
                        action = input("actions? ")
                        if action == "connections":
                            log.write(current_floor.room[0].connection_pos)
                        elif action == "navigate" or action == 'n':
                            self.action_list += current_floor.navigate(obs, (x, y), current_floor.goal)
                            break
                        #elif action == "search" or action == 's':
                        #    return current_floor.corridor[0].search(original_map, (x,y))
                        elif action == "preprocessmap":
                            preprocess_map = current_floor.preprocess_map(obs)
                            for j in range(21):
                                for i in range(79):
                                    if preprocess_map[j][i]:
                                        log.write(".")
                                    else:
                                        log.write(" ")
                                log.write("\n")
                        else:
                            self.action_list.append(int(action))
                            break
                else: #current_floor.goal is None
                    if current_floor.search_pos is None:
                        # for j in range(21):
                        #     for i in range(79):
                        #         log.write("%.3f" % current_floor.occ_map[j][i])
                        #     log.write("\n")
                        current_floor.search_pos = current_floor.calculate_search_position()
                    if (x, y) != current_floor.search_pos.queue[0][1]:
                        self.action_list += current_floor.navigate(obs, (x,y), current_floor.search_pos.queue[0][1])
                    else:
                        self.action_list += [22] * 20
                        current_floor.search_pos.get()
                    self.last_action = self.action_list.pop()
                    self.last_action_list.append(self.last_action)
                    return self.last_action
            self.last_action = self.action_list.pop()
            self.last_action_list.append(self.last_action)
            return self.last_action
            
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            #print("exception: ", e)
            #print("line ", exc_tb.tb_lineno)
            #print('a', self.eat_pos)
            #a = input()
            aaa = env.action_space.sample()
            if aaa == 21 :
                aaa = 19
            return aaa
