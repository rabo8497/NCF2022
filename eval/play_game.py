import argparse
import datetime
import importlib
import logging
import os
import shlex
import subprocess
from pathlib import Path
import re

from tqdm import tqdm

logger = logging.getLogger(__name__)


def run_play_game(agent, env, seed, timeout, verbose):

    # agent example -> agents.my_agent
    cmd = f"python -m eval.play_game --agent={agent} --env={env} --seed={seed}"

    result = [0.0, 0.0]
    remain = 3  # 시간 초과발생할 경우 최대 3번까지 재시도
    log_buff = []
    while remain > 0:
        remain -= 1

        try:
            pout = subprocess.run(
                shlex.split(cmd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )

            stdout_line = pout.stdout.split(b"\n")
            stdout_line = [line.rstrip().decode("utf-8") for line in stdout_line]
            stderr_line = pout.stderr.split(b"\n")
            stderr_line = [line.rstrip().decode("utf-8") for line in stderr_line]
            lines = (
                [f"{cmd}\n\n"]
                + ["## STDOUT ##\n"]
                + stdout_line
                + ["\n## STDERR ##\n"]
                + stderr_line
            )
            log_buff += lines

            result[0] = float(re.split('[, ]', stdout_line[0])[-1])
            result[1] = float(re.split('[, ]', stdout_line[0])[2])

            break

        except subprocess.TimeoutExpired:
            # 비정상적으로 게임이 종료된 경우
            # 시간초과
            result = [0.0, 0.0]

    return result, log_buff


def play_game(args):
    # agent 초기화
    try:
        module = args.agent
        name = "Agent"
        agent = getattr(importlib.import_module(module), name)(args)

        result = agent.evaluate(args.seed)
    
    # except (AttributeError, ImportError):
    except Exception as e:
        import traceback

        logger.error(f"Agent 클래스를 임포트 할 수 없음: {agent_path}, {e}")
        traceback.print_exc()
        result = [0.0, 0.0]
    return result


if __name__ == "__main__":

    parser = argparse.ArgumentParser("NC Fellowship 2022-play_game")
    parser.add_argument(
        "--agent",
        type=str,
    )
    parser.add_argument(
        "--env",
        type=str,
        default="NetHackChallenge-v0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1_000_000,
    )
    parser.add_argument(
        "--savedir",
        default="nle_data/play_data",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="test",
        choices=["test", "train"],
        help="Test or train. Defaults to 'test'",
    )
    
    args = parser.parse_args()

    result = play_game(args)