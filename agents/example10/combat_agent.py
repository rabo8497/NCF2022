from .a2c import A2C
import torch
from torch.nn import functional as F

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
        
    def end_battle(self, obs, pos):
        if self.battle_start_pos is not None and self.battle_start_pos == pos:
            self.battle_start_pos = None
            self.battle = False 
        
    def get_action(self, env, obs, floor, enemy_pos):
        x, y = obs['blstats'][:2]
        pos = (x,y)
        if enemy_pos is not None:
            ##battle!
            actor, _ = self.get_actor_critic(env, obs)
            action = torch.multinomial(F.softmax(actor, dim=1), num_samples=1)
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