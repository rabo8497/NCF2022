from .a2c import A2C
import torch
from torch.nn import functional as F
import time
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CombatAgent():
    def __init__(self, net) -> None:
        self.battle = False
        self.move_memory = []
        self.dxy = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self.battle_start_pos = None

        self.a2c = net
        
    def start_battle(self, env, obs, floor, enemy_pos):
        x, y = obs['blstats'][:2]
        self.battle_start_pos = (x, y)
        self.battle = True
        return self.get_action(env, obs, floor, enemy_pos)

    def is_monster(self, letter):
        if letter >= 65 and letter <= 90:
            return True
        elif letter >= 97 and letter <= 122:
            return True
        elif letter == ord(":"):
            return True
        #elif letter == ord("@") :
        #    time.sleep(3)
        #    return True
        else:
            return False

    def is_enemy(self, original_map, color_map, pos):
        letter = original_map[pos[1]][pos[0]]
        if color_map[pos[1]][pos[0]] == 15:
            return False
        else:
            return self.is_monster(letter)

    def end_battle(self, obs, pos):
        if self.battle_start_pos is not None and self.battle_start_pos == pos:
            self.battle_start_pos = None
            self.battle = False 
    def is_walkable(self, original_map, pos):
        walkable_objects = set(['.','#','<','>','+','@','$',')','[','%','?','/','=','!','(','"','*',])
        letter = original_map[pos[1]][pos[0]]
        if chr(letter) in walkable_objects:
            return True
        elif self.is_monster(letter):
            return True
        else:
            return False
    def get_action(self, env, obs, floor, enemy_pos):
        x, y = obs['blstats'][:2]
        pos = (x,y)
        original_map = obs['chars']
        color_map = obs['colors']
        if self.is_enemy(original_map, color_map, (x+1, y)) :
            return torch.tensor([[2]]), False
        elif self.is_enemy(original_map, color_map, (x-1, y)) :
            return torch.tensor([[4]]), False
        elif self.is_enemy(original_map, color_map, (x, y+1)) :
            return torch.tensor([[3]]), False
        elif self.is_enemy(original_map, color_map, (x, y-1)) :
            return torch.tensor([[1]]), False


        elif self.is_enemy(original_map, color_map, (x+1, y+1)) :
            return torch.tensor([[6]]), False
        elif self.is_enemy(original_map, color_map, (x-1, y+1)) :
            return torch.tensor([[7]]), False
        elif self.is_enemy(original_map, color_map, (x+1, y-1)) :
            return torch.tensor([[5]]), False
        elif self.is_enemy(original_map, color_map, (x-1, y-1)) :
            return torch.tensor([[8]]), False




        #time.sleep(1)
        if enemy_pos is not None:
            ##battle!
            actor, _ = self.get_actor_critic(env, obs)
            acac = F.softmax(actor, dim=1)
            action = torch.multinomial(acac, num_samples=1)
            while True :
                if action.item() == 1 and not self.is_walkable(original_map, (x, y-1)):
                    action = torch.multinomial(acac, num_samples=1)
                elif action.item() == 2 and not self.is_walkable(original_map, (x+1, y)) :
                    action = torch.multinomial(acac, num_samples=1)
                elif action.item() == 3 and not self.is_walkable(original_map, (x, y+1)):
                    action = torch.multinomial(acac, num_samples=1)
                elif action.item() == 4 and not self.is_walkable(original_map, (x-1, y)) :
                    action = torch.multinomial(acac, num_samples=1)
                else : break
            return action, True
        else:  ##get back to original position
            if not self.move_memory:
                self.move_memory += floor.navigate(obs, (x,y), self.battle_start_pos)
            return self.move_memory.pop(), False

    def get_actor_critic(self, env, obs):
        observed_glyphs = torch.from_numpy(obs['glyphs']).float().unsqueeze(0).to(device)
        observed_stats = torch.from_numpy(obs['blstats']).float().unsqueeze(0).to(device)

        actor, critic = self.a2c(observed_glyphs, observed_stats)
        return actor, critic

    def optimize(self, actors, actions, critics, returns):
        actors = torch.cat(actors).to(device)
        pass