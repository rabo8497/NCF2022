import multiprocessing as mp
import time
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory
from types import SimpleNamespace
from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm
import zmq
import zmq.devices
from IPython import embed

# data_size = 10_000_000  # 100_000_000
data_size = 10_000_000  # 100_000_000
n_iters = 256  # int(1e2)
n_workers = 10
task_load = 0.001


@dataclass
class ElapsedTime:
    __slots__ = ("start", "end", "spent")
    start: float
    end: float
    spent: float


@contextmanager
def elapsed_time() -> ElapsedTime:
    start = time.monotonic()
    rst = ElapsedTime(start=start, end=None, spent=None)
    yield rst

    rst.end = time.monotonic()
    rst.spent = rst.end - start


@contextmanager
def wait(target: float):
    start = time.monotonic()
    yield

    end = time.monotonic()
    elapsed_time = end - start
    remain = max(0, target - elapsed_time)
    time.sleep(remain)


#
# zmq queue
#


def run_proxy(f_port, b_port):
    proxy = zmq.devices.Proxy(zmq.ROUTER, zmq.DEALER)
    proxy.bind_in(f"tcp://*:{f_port}")
    proxy.bind_out(f"tcp://*:{b_port}")
    proxy.start()


def zmq_queue_func(port):
    context = zmq.Context()
    sock = context.socket(zmq.REQ)
    sock.connect(f"tcp://127.0.0.1:{port}")

    msg = np.random.random(data_size // 2).tobytes()

    while True:
        sock.send(msg, copy=False)
        recv = sock.recv(copy=False).bytes

        # do somthing
        with wait(task_load):
            assert np.allclose(np.frombuffer(recv), np.frombuffer(msg) + 1)
            msg = np.random.random(data_size // 2).tobytes()


def zmq_queue_main(conditions: List[Tuple[str, int]]):
    #
    # zmq queue
    #
    logs = []

    f_port, b_port = 22345, 22543
    proxy_proc = mp.Process(target=run_proxy, args=(f_port, b_port), daemon=True)
    proxy_proc.start()

    for name, n_workers in conditions:

        context = zmq.Context()
        sock = context.socket(zmq.REP)

        sock.connect(f"tcp://localhost:{b_port}")

        proc = [
            mp.Process(target=zmq_queue_func, args=(f_port,))
            for rank in range(n_workers)
        ]
        [p.start() for p in proc]
        time.sleep(10)
        for i in tqdm.tqdm(np.arange(n_iters), desc=name):
            with elapsed_time() as etime1:
                msg = sock.recv(copy=False).bytes

            recv = (np.frombuffer(msg) + 1).tobytes()

            with elapsed_time() as etime2:
                sock.send(recv, copy=False)

            logs.append(dict(kind=name, etime=etime1.spent + etime1.spent))

        [p.terminate() for p in proc]
        [p.join() for p in proc]
        sock.close()
        context.term()

    proxy_proc.terminate()
    proxy_proc.join()
    return logs


#
# pipe
#


def pipe_func(rank: int, conn: Connection):

    msg = np.random.random(data_size // 2).tobytes()

    while True:
        # send
        conn.send_bytes(msg)
        # recv
        recv = conn.recv_bytes()
        # do somthing
        with wait(task_load):
            assert np.allclose(np.frombuffer(recv), np.frombuffer(msg) + 1)
            msg = np.random.random(data_size // 2).tobytes()


def pipe_main(conditions: List[Tuple[str, int]]):
    #
    # Pipe
    #
    logs = []

    for name, n_workers in conditions:

        conns: Connection = []
        procs: mp.Process = []
        for rank in range(n_workers):
            parent, child = mp.Pipe()
            conns.append(parent)
            procs.append(mp.Process(target=pipe_func, args=(rank, child)))

        [p.start() for p in procs]

        time.sleep(10)

        tbar = tqdm.tqdm(total=n_iters, desc=name)
        i = 0

        while True:
            # recv
            with elapsed_time() as etime1:
                msgs = []
                avail_conns = mp.connection.wait(conns, 0)
                for conn in avail_conns:
                    msg = conn.recv_bytes()
                    msgs.append(msg)

            tbar.write(f"{list(range(len(msgs)))}")
            msgs = [(np.frombuffer(m) + 1).tobytes() for m in msgs]

            # send
            with elapsed_time() as etime2:
                for conn, msg in zip(avail_conns, msgs):
                    conn.send_bytes(msg)

            dones = len(msgs)
            if dones > 0:
                logs.append(
                    dict(
                        kind=name,
                        etime=(etime1.spent + etime2.spent) / dones,
                    )
                )
                tbar.update(dones)
                if tbar.n >= tbar.total:
                    break
        tbar.close()

        [p.terminate() for p in procs]
        [p.join() for p in procs]

    return logs


#
# zmq
#


def zmq_func(rank: int):

    context = zmq.Context()
    sock = context.socket(zmq.PAIR)
    sock.connect(f"ipc:///tmp/r{rank}")

    msg = np.random.random(data_size // 2).tobytes()

    while True:
        # send
        sock.send(msg)
        # recv
        recv = sock.recv()
        # do somthing
        with wait(task_load):
            assert np.allclose(np.frombuffer(recv), np.frombuffer(msg) + 1)
            msg = np.random.random(data_size // 2).tobytes()


def zmq_main(conditions: List[Tuple[str, int]]):
    logs = []

    for name, n_workers in conditions:

        context = zmq.Context()
        poller = zmq.Poller()

        conns = []
        procs = []
        for rank in range(n_workers):
            sock = context.socket(zmq.PAIR)
            sock.bind(f"ipc:///tmp/r{rank}")
            poller.register(sock, zmq.POLLIN)
            conns.append(sock)
            procs.append(mp.Process(target=zmq_func, args=(rank,)))

        [p.start() for p in procs]

        time.sleep(10)

        tbar = tqdm.tqdm(total=n_iters, desc=name)
        i = 0

        while True:
            # recv
            with elapsed_time() as etime1:
                rank_msgs = []
                socks = dict(poller.poll())
                for rank, conn in enumerate(conns):
                    if socks.get(conn) == zmq.POLLIN:
                        msg = conn.recv()
                        rank_msgs.append((rank, msg))

            tbar.write(",".join([f"{r}" for r, _ in rank_msgs]))
            rank_msgs = [(r, (np.frombuffer(m) + 1).tobytes()) for r, m in rank_msgs]

            # send
            with elapsed_time() as etime2:
                for rank, msg in rank_msgs:
                    conn = conns[rank]
                    conn.send(msg)

            dones = len(rank_msgs)
            if dones > 0:
                logs.append(
                    dict(
                        kind=name,
                        etime=(etime1.spent + etime2.spent) / dones,
                    )
                )
                tbar.update(dones)
                if tbar.n >= tbar.total:
                    break
        tbar.close()

        [p.terminate() for p in procs]
        [p.join() for p in procs]

    return logs


#
# shared memory
#


class Connector:
    def ready(self) -> bool:
        raise NotImplementedError()

    def send(self, data: Any):
        raise NotImplementedError()

    def recv(self) -> Any:
        raise NotImplementedError()


class PipeConnector(Connector):
    def __init__(self, conn) -> None:
        self.conn = conn

    def ready(self):
        return self.conn.poll()

    def send(self, data):
        self.conn.send(data)

    def recv(self):
        return self.conn.recv()


class ZmqConnector(Connector):
    def __init__(self, sock, poller) -> None:
        self.sock = sock
        self.poller = poller

    def ready(self, socks):
        return socks.get(self.sock) == zmq.POLLIN

    def send(self, data):
        self.sock.send_pyobj(data)

    def recv(self):
        return self.sock.recv_pyobj()


class SharedMemoryController:
    def __init__(
        self,
        rank: int,
        mem: SharedMemory,
        mem_size: int,
        conn: Connector,
    ) -> None:
        self.mem = mem
        self.max_size = mem_size
        self.rank = rank
        self.start_pos = rank * self.max_size
        self.conn = conn

    def ready(self, *args) -> bool:
        return self.conn.ready(*args)

    def write(self, values: bytes):
        length = len(values)
        assert length < self.max_size
        end_pos = self.start_pos + length
        self.mem.buf[self.start_pos : end_pos] = values
        self.conn.send((self.start_pos, end_pos))

    def read(self) -> bytes:
        start_pos, end_pos = self.conn.recv()
        assert (end_pos - start_pos) < self.max_size
        return self.mem.buf[start_pos:end_pos]


def pipe_shm_func(rank, pipe, shm: SharedMemory):

    smc = SharedMemoryController(rank, shm, data_size * 8, pipe)

    msg = np.random.random(data_size // 2).tobytes()

    while True:
        # send
        smc.write(msg)
        # recv
        recv = smc.read()
        # do somthing
        with wait(task_load):
            assert np.allclose(np.frombuffer(recv), np.frombuffer(msg) + 1)
            msg = np.random.random(data_size // 2).tobytes()


def pipe_shm_main(conditions: List[Tuple[str, int]]):
    logs = []

    for name, n_workers in conditions:
        smcs: List[SharedMemoryController] = []
        procs = []
        with SharedMemoryManager() as smm:
            shm = smm.SharedMemory(size=n_workers * data_size * 8)
            for rank in range(n_workers):
                parent, child = mp.Pipe()
                smcs.append(
                    SharedMemoryController(
                        rank, shm, data_size * 8, PipeConnector(parent)
                    )
                )
                procs.append(mp.Process(target=pipe_shm_func, args=(rank, child, shm)))
            [p.start() for p in procs]
            time.sleep(10)
            tbar = tqdm.tqdm(total=n_iters, desc=name)
            i = 0
            while True:
                # recv
                with elapsed_time() as etime1:
                    rank_msgs = []
                    for rank, smc in enumerate(smcs):
                        if smc.ready():
                            msg = smc.read()
                            rank_msgs.append((rank, msg))

                tbar.write(",".join([f"{r}" for r, _ in rank_msgs]))
                rank_msgs = [
                    (r, (np.frombuffer(m) + 1).tobytes()) for r, m in rank_msgs
                ]

                # send
                with elapsed_time() as etime2:
                    for rank, msg in rank_msgs:
                        smc = smcs[rank]
                        smc.write(msg)

                dones = len(rank_msgs)
                if dones > 0:
                    logs.append(
                        dict(
                            kind=name,
                            etime=(etime1.spent + etime2.spent) / dones,
                        )
                    )
                    tbar.update(dones)
                    if tbar.n >= tbar.total:
                        break
            tbar.close()

        [p.terminate() for p in procs]
        [p.join() for p in procs]

    return logs


#
# ZMQ Shared Memory
#


def zmq_shm_func(rank: int, shm: SharedMemory):

    context = zmq.Context()
    sock = context.socket(zmq.PAIR)
    sock.connect(f"ipc:///tmp/r{rank}")
    smc = SharedMemoryController(rank, shm, data_size * 8, ZmqConnector(sock, None))

    msg = np.random.random(data_size // 2).tobytes()

    while True:
        # send
        smc.write(msg)
        # recv
        recv = smc.read()
        # do somthing
        with wait(task_load):
            assert np.allclose(np.frombuffer(recv), np.frombuffer(msg) + 1)
            msg = np.random.random(data_size // 2).tobytes()


def zmq_shm_main(conditions: List[Tuple[str, int]]):

    logs = []

    for name, n_workers in conditions:

        context = zmq.Context()
        poller = zmq.Poller()

        smcs: List[SharedMemoryController] = []
        procs: List[mp.Process] = []

        with SharedMemoryManager() as smm:
            shm = smm.SharedMemory(size=n_workers * data_size * 8)
            for rank in range(n_workers):
                sock = context.socket(zmq.PAIR)
                sock.bind(f"ipc:///tmp/r{rank}")
                poller.register(sock, zmq.POLLIN)
                smcs.append(
                    SharedMemoryController(
                        rank, shm, data_size * 8, ZmqConnector(sock, poller)
                    )
                )
                procs.append(mp.Process(target=zmq_shm_func, args=(rank, shm)))

            [p.start() for p in procs]

            time.sleep(10)

            tbar = tqdm.tqdm(total=n_iters, desc=name)
            i = 0

            while True:
                # recv
                with elapsed_time() as etime1:
                    rank_msgs = []
                    socks = dict(poller.poll())
                    for rank, smc in enumerate(smcs):
                        if smc.ready(socks):
                            msg = smc.read()
                            rank_msgs.append((rank, msg))

                tbar.write(",".join([f"{r}" for r, _ in rank_msgs]))
                rank_msgs = [
                    (r, (np.frombuffer(m) + 1).tobytes()) for r, m in rank_msgs
                ]

                # send
                with elapsed_time() as etime2:
                    for rank, msg in rank_msgs:
                        smc = smcs[rank]
                        smc.write(msg)

                dones = len(rank_msgs)
                if dones > 0:
                    logs.append(
                        dict(
                            kind=name,
                            etime=(etime1.spent + etime2.spent) / dones,
                        )
                    )
                    tbar.update(dones)
                    if tbar.n >= tbar.total:
                        break
            tbar.close()

            [p.terminate() for p in procs]
            [p.join() for p in procs]

    return logs


if __name__ == "__main__":

    logs = []

    def conditions(n, ws):
        return [(f"{n}{w}", w) for w in ws]

    ws = [10, 20, 30, 40, 80, 160]
    # logs += zmq_queue_main()
    logs += pipe_main(conditions("p", ws))
    logs += zmq_main(conditions("z", ws))
    logs += pipe_shm_main(conditions("s", ws))
    logs += zmq_shm_main(conditions("x", ws))

    logs = pd.DataFrame(logs)
    etimes = logs.groupby("kind").mean()["etime"]
    # order = etimes.index[etimes.argsort()]
    # sns.barplot(data=logs, x="kind", y="etime", order=order)
    # plt.savefig("shard.png")

    sns.barplot(data=logs, x="kind", y="etime")
    # plt.savefig("shard.png")
    plt.show()

    embed()
    exit()
