import numpy as np
from collections import deque

class Trajectory():
    __slots__ = [
        "actors",
        "actions",
        "rewards",
        "obss",
        "new_obss",
        "dones",
        "time_step",
    ]

    def __init__(self):
        self.actors  = []
        self.actions = []
        self.rewards = []
        self.obss = []
        self.new_obss = []
        self.dones = []
        self.time_step = None

    def add(self, actor, action, reward, obs, new_obs, done, time_step):
        self.actors.append(actor)
        self.actions.append(action)
        self.rewards.append(reward)
        self.obss.append(obs)
        self.new_obss.append(new_obs)
        self.dones.append(done)
        self.time_step = time_step

    def get(self):
        self.dones[-1] = 1
        return self.actors, self.actions, self.rewards, self.obss, self.new_obss, self.dones

    def length(self):
        return len(self.actors)



class Buffer():
    def __init__(self, num_actors) -> None:
        self.record_length = 0
        self.finished_episodes = deque([])
        self.ongoing_episodes = [None for _ in range(num_actors)]

    def add(self, actor, action, reward, obs, new_obs, done, actor_num, time_step):
        if self.ongoing_episodes[actor_num] is None:
            self.ongoing_episodes[actor_num] = Trajectory()
        # 전에 쓰던거랑 다른게 들어왔으면 전에 쓰던걸 옮겨놓고 새로쓰기
        elif self.ongoing_episodes[actor_num].time_step != time_step - 1:
           self.finished_episodes.append(self.ongoing_episodes[actor_num])
           self.record_length += self.ongoing_episodes[actor_num].length()
           self.ongoing_episodes[actor_num] = Trajectory() 
        self.ongoing_episodes[actor_num].add(actor, action, reward, obs, new_obs, done,time_step)

    def make_train_data(self, size):
        assert size <= self.record_length
        actor_return, action_return, reward_return, obs_return, new_obs_return, done_return = [], [], [], [], [], []

        while self.finished_episodes:
            finished_episode = self.finished_episodes.popleft()
            actors, actions, rewards, obss, new_obss, dones = finished_episode.get()
            actor_return += actors
            action_return += actions
            reward_return += rewards
            obs_return += obss
            new_obs_return += new_obss
            done_return += dones

        actor_return = actor_return[:size]
        action_return = action_return[:size]
        reward_return = reward_return[:size]
        obs_return = obs_return[:size]
        new_obs_return = new_obs_return[:size]
        done_return = done_return[:size]

        self.record_length = 0

        return actor_return, action_return, reward_return, obs_return, new_obs_return, done_return
