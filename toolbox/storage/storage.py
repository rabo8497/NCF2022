import argparse
import errno
import multiprocessing as mp
import pickle
import stat
import time
from fnmatch import fnmatch

import pickleshare
import zmq
from IPython import embed
from zmq.devices.basedevice import ProcessDevice

# from ..network import zmq_queue
from ..process import kill_all_processes


def queue_proxy(frontend_port, backend_port):
    # https://soooprmx.com/zmq-프록시-사용하기
    proxy = zmq.device.Proxy(zmq.ROUTER, zmq.DEALER)
    proxy.bind_in(f"tcp://*:{frontend_port}")
    proxy.bind_out(f"tcp://*:{backend_port}")
    proxy.start()


def queue_device(frontend_port, backend_port):
    device = ProcessDevice(zmq.QUEUE, zmq.ROUTER, zmq.DEALER)
    device.bind_in("tcp://127.0.0.1:%d" % frontend_port)
    device.bind_out("tcp://127.0.0.1:%d" % backend_port)
    device.start()


class Code:
    FAIL = b"00"
    SUCC = b"01"
    CLR = b"02"
    GET = b"03"
    SET = b"04"
    HAS = b"05"
    KEYS = b"06"
    SIZE = b"07"
    LEN = b"08"


class BytesDB(pickleshare.PickleShareDB):
    def __getitem__(self, key):
        """ db['key'] reading """
        fil = self.root / key
        try:
            mtime = fil.stat()[stat.ST_MTIME]
        except OSError:
            raise KeyError(key)

        if fil in self.cache and mtime == self.cache[fil][1]:
            return self.cache[fil][0]
        try:
            # The cached item has expired, need to read
            with fil.open("rb") as f:
                obj = f.read()
        except:
            raise KeyError(key)

        self.cache[fil] = (obj, mtime)
        return obj

    def __setitem__(self, key, value):
        """ db['key'] = 5 """
        fil = self.root / key
        parent = fil.parent
        if parent and not parent.is_dir():
            parent.mkdir(parents=True)
        with fil.open("wb") as f:
            f.write(value)
        try:
            self.cache[fil] = (value, fil.stat().st_mtime)
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise


def run_server(root, backend_port):
    context = zmq.Context()
    sock = context.socket(zmq.REP)
    sock.connect(f"tcp://localhost:{backend_port}")

    db = BytesDB(root)

    while True:
        received = sock.recv_multipart(copy=False)
        received = [r.bytes for r in received]
        op, data = received[0], received[1:]

        if op == Code.CLR:
            db.clear()
            rst = [Code.SUCC]

        elif op == Code.SET:
            key, value = data[0], data[1]
            db[key.decode()] = value
            rst = [Code.SUCC]

        elif op == Code.HAS:
            key = data[0].decode("utf-8")
            rst = [Code.SUCC if key in db else Code.FAIL]

        elif op == Code.GET:
            key = data[0].decode()
            rst = [
                Code.SUCC if key in db else Code.FAIL,
                db.get(key, b""),
            ]

        elif op == Code.KEYS:
            pattern = data[0].decode()
            rst = [key.encode() for key in db.keys() if fnmatch(key, pattern)]

        elif op == Code.SIZE:
            size = sum([len(v) for v in db.values()])
            rst = [pickle.dumps(size, protocol=pickle.HIGHEST_PROTOCOL)]

        elif op == Code.LEN:
            rst = [pickle.dumps(len(db), protocol=pickle.HIGHEST_PROTOCOL)]

        else:
            raise NotImplementedError()

        sock.send_multipart(rst, copy=False)


class Server:
    def __init__(self, root, port) -> None:
        self.root = root
        self.frontend_port = port
        self.backend_port = port + 1

    def start(self):
        queue_device(self.frontend_port, self.backend_port)
        self.server = mp.Process(
            target=run_server, args=(self.root, self.backend_port), daemon=False,
        )
        self.server.start()
        return self

    def stop(self):
        kill_all_processes(self.server.pid)
        return self

    def __del__(self):
        self.stop()


class Client:
    def __init__(self, server_ip, port, timeout) -> None:
        self.server_ip = server_ip
        self.port = port
        self.timeout = timeout
        self._init_socket()

    def _init_socket(self):
        self.context = zmq.Context()
        self.sock = self.context.socket(zmq.REQ)
        self.sock.connect(f"tcp://{self.server_ip}:{self.port}")

    def step(self, code, key=b"", value=b""):
        # send
        data = [code]
        if key:
            data.append(key.encode())
            if value:
                data.append(value)
        self.sock.send_multipart(data, copy=False)
        values = self.sock.recv_multipart(copy=False)
        values = [v.bytes for v in values]
        return values

    def clear(self):
        return Code.SUCC == self.step(Code.CLR)[0]

    def set(self, key, value):
        assert isinstance(key, str)
        value = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        return Code.SUCC == self.step(Code.SET, key, value)[0]

    def has(self, key):
        assert isinstance(key, str)
        return Code.SUCC == self.step(Code.HAS, key)[0]

    def get(self, key, default=None):
        assert isinstance(key, str)
        succ, value = self.step(Code.GET, key)
        return pickle.loads(value) if succ == Code.SUCC else default

    def keys(self, pattern="*"):
        assert isinstance(pattern, str)
        resp = self.step(Code.KEYS, pattern)
        return [v.decode() for v in resp]

    def size(self):
        resp = self.step(Code.SIZE)
        return pickle.loads(resp[0])

    def len(self):
        resp = self.step(Code.LEN)
        return pickle.loads(resp[0])

    def set_model(key, model):
        pass

    def get_model(key, model):
        pass


if __name__ == "__main__":

    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str)
    parser.add_argument("--server_ip", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=60002)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    try:
        server = Server(args.root, args.port).start()

        db = Client(args.server_ip, args.port, args.timeout)
        db.clear()
        db.set("hello", 15)
        value = db.get("hello")
        assert 15 == value

        for data in range(25):
            key, value = f"{data}", data
            db.set(key, value)
            value2 = db.get(key)
            print(value, value2)
            # time.sleep(0.1)

        data = np.random.random(1000000)
        db.set("data", data.tobytes())
        ret = db.get("data")
        assert data.tobytes() == ret

        embed()
        exit()

        # db.clear()
        # print("Should be empty:", db.items())
        # db["hello"] = 15
        # db["aku ankka"] = [1, 2, 313]
        # db["paths/are/ok/key"] = [1, (5, 46)]
        # print(db.keys())

        # server = start_redis(port, password)
        # c = redis.Redis(port=port, password=password)
        # c.set(1, 1)
        # print(c.get(1))
        # c.save()  # disk에 저장(동기)
        # # c.bgsave()  # disk에 저장(비동기)
        server.stop()

    except Exception as exc:
        import traceback

        traceback.print_exc()

    finally:
        kill_all_processes()

