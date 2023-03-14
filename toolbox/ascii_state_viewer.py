import logging
import os
import platform

import gym
import numpy as np
import zmq

from IPython import embed

logger = logging.getLogger(__name__)


class AsciiStatePub(gym.core.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        context = zmq.Context()
        self.socket = context.socket(zmq.PUB)
        cfg = config.select("comm")
        self.topic = cfg.ascii_state_topic
        port = cfg.ascii_state_pub_port
        self.socket.bind(f"tcp://*:{port}")

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.socket.send(f"{self.topic} {self.model}".encode("utf-8"))
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.socket.send(f"{self.topic} {self.model}".encode("utf-8"))
        return obs, reward, done, info


if __name__ == "__main__":

    cfg = config.select("comm")
    port = cfg.ascii_state_pub_port
    topic = cfg.ascii_state_topic

    # Socket to talk to server
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(f"tcp://localhost:{port}")
    socket.setsockopt(zmq.SUBSCRIBE, topic.encode("utf-8"))
    os.system("clear" if platform.system() == "Linux" else "cls")

    while True:
        string = socket.recv()
        topic, data = string.split(b" ", 1)
        data = data.decode("utf-8")
        print(data)
        # https://tldp.org/HOWTO/Bash-Prompt-HOWTO/x361.html
        print("\033[0;0H", end="")
