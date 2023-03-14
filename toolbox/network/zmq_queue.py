import argparse
import enum
import multiprocessing as mp
import pickle
import platform
import threading
import time

import numpy as np
import tqdm
import zmq
from IPython import embed


def queue_proxy(frontend_port, backend_port):
    # https://soooprmx.com/zmq-프록시-사용하기
    proxy = zmq.device.Proxy(zmq.ROUTER, zmq.DEALER)
    proxy.bind_in(f"tcp://*:{frontend_port}")
    proxy.bind_out(f"tcp://*:{backend_port}")
    proxy.start()


def device(context, frontend_port, protocol, backend_port):
    try:
        if context is None:
            context = zmq.Context(1)
        # Socket facing clients
        frontend = context.socket(zmq.XREP)
        frontend.bind(f"tcp://*:{frontend_port}")
        # Socket facing services
        backend = context.socket(zmq.XREQ)
        if protocol == "tcp":
            backend.bind(f"tcp://*:{backend_port}")
        elif protocol == "ipc":
            backend.bind("ipc://backend.ipc")
        elif protocol == "inproc":
            backend.bind(f"inproc://backend")
        else:
            raise NotImplementedError()
        zmq.device(zmq.QUEUE, frontend, backend)
    except Exception as exc:
        print(exc)
        print("bringing down zmq device")
    finally:
        pass
        frontend.close()
        backend.close()
        context.term()


class Server:
    """
    [summary]
    Learner
    """

    def __init__(self, frontend_port, protocol, backend_port):
        self.context = zmq.Context()
        self.sock = self.context.socket(zmq.REP)

        if platform.system() == "Windows":
            protocol = "tcp"

        if protocol == "tcp":
            from zmq.devices import ProcessDevice

            pd = ProcessDevice(zmq.QUEUE, zmq.ROUTER, zmq.DEALER)
            pd.bind_in(f"tcp://*:{frontend_port}")
            pd.bind_out(f"tcp://*:{backend_port}")
            pd.setsockopt_in(zmq.IDENTITY, "ROUTER".encode())
            pd.setsockopt_out(zmq.IDENTITY, "DEALER".encode())
            pd.start()
            self.sock.connect(f"tcp://localhost:{backend_port}")

        elif protocol == "ipc":
            mp.Process(
                target=device,
                args=(None, frontend_port, protocol, backend_port),
                daemon=True,
            ).start()
            self.sock.connect(f"ipc://backend.ipc")

        elif protocol == "inproc":
            threading.Thread(
                target=device,
                args=(self.context, frontend_port, protocol, backend_port),
            ).start()
            self.sock.connect("inproc://backend")

        elif protocol == "proxy":
            mp.Process(
                target=queue_proxy, args=(frontend_port, backend_port), daemon=True,
            ).start()

        else:
            raise NotImplementedError()

    def stop(self):
        pass

    def run(self):
        while True:
            start_time = time.monotonic()
            n_pings = 0
            for _ in tqdm.trange(100_000):
                xs = self.sock.recv_multipart(copy=False)
                n_pings += 1
                ys = xs[:]
                self.sock.send_multipart(ys, copy=False)
                xs = [pickle.loads(x) for x in xs]
                tqdm.tqdm.write(f"{xs}")
            tqdm.tqdm.write(f"{n_pings / (time.monotonic() - start_time)}")


class Client:
    """
    [summary]
    Actor
    """

    def __init__(self, server_ip, frontend_port, timeout) -> None:
        self.server_ip = server_ip
        self.frontend_port = frontend_port
        self.timeout = timeout * 1000
        self._init_socket()

    def _init_socket(self):
        self.context = zmq.Context()
        self.sock = self.context.socket(zmq.REQ)
        self.sock.connect(f"tcp://{self.server_ip}:{self.frontend_port}")
        self.poller = zmq.Poller()
        self.poller.register(self.sock, zmq.POLLIN)

    def step(self, keys):
        self.sock.send_multipart(keys, copy=False)
        socks = dict(self.poller.poll(self.timeout))
        if socks and socks.get(self.sock) == zmq.POLLIN:
            succ = True
            values = self.sock.recv_multipart(zmq.NOBLOCK, copy=False)
            values = [v.bytes for v in values]
        else:
            self._init_socket()
            succ, values = False, [b""]
        return succ, values

    def step_pyobjs(self, objs):
        keys = [pickle.dumps(obj, pickle.HIGHEST_PROTOCOL) for obj in objs]
        succ, values = self.step(keys)
        objs = [pickle.loads(value) for value in values]
        return succ, objs


def test_actor(server_ip, frontend_port, timeout):
    client = Client(server_ip, frontend_port, timeout)
    while True:
        key = np.random.random(10).tolist()
        succ, value = client.step_pyobjs(key)
        assert np.allclose(key, value)
        time.sleep(0.01)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--server_ip", type=str)
    parser.add_argument("--frontend_port", type=int, default=60000)
    parser.add_argument("--backend_port", type=int, default=60001)
    parser.add_argument("--n_workers", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument(
        "--protocol", choices=["tcp", "ipc", "inproc", "proxy"], default="tcp"
    )
    args = parser.parse_args()

    if args.server_ip is None:
        #
        # Server
        #
        server = Server(args.frontend_port, args.protocol, args.backend_port)
        server.run()

    else:
        #
        # Client
        #
        client = Client(args.server_ip, args.frontend_port, args.timeout)
        for key in range(25):
            keys = [key, key + 1]
            succ, value = client.step_pyobjs(keys)
            print(key, value)
            time.sleep(0.1)

        workers = list()
        for rank in range(args.n_workers):
            worker = mp.Process(
                target=test_actor,
                args=(args.server_ip, args.frontend_port, args.timeout),
                daemon=True,
            )
            workers.append(worker)
        for w in workers:
            w.start()

        while True:
            pass
