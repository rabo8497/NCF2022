import mmap
import multiprocessing as mp
import pathlib

# from toolbox_rust.memory import Pipe as rPipe
# from toolbox_rust.memory import vec_to_bytes
from multiprocessing.managers import SharedMemoryManager

# import blosc2
import numpy as np
import zmq
from IPython import embed
from numpy.random import shuffle

from .shared_memory import Pipe, SharedMemory


def pipe_func(pipe):
    while True:
        data = pipe.recv()
        if len(data) == 0:
            break
        data = np.frombuffer(data) + 1
        pipe.send(data.tobytes())


def pipe_bytes_func(pipe):
    while True:
        data = pipe.recv_bytes()
        if len(data) == 0:
            break
        data = np.frombuffer(data) + 1
        pipe.send_bytes(data.tobytes())


def decode_uint64(value):
    return (value).to_bytes(8, byteorder="little", signed=False)


def encode_uint64(bytes_array):
    return int.from_bytes(bytes(bytes_array), byteorder="little", signed=False)


def pipe_buff_func(pipe_size, pipe_data):

    buff_data = bytearray(128)
    while True:
        length = pipe_size.recv()
        if len(buff_data) < length:
            buff_data = bytearray(int(length * 1.1))
        pipe_data.recv_bytes_into(buff_data)
        if length == 0:
            break
        rst = np.frombuffer(buff_data[:length]) + 1
        rst = rst.tobytes()
        pipe_size.send(len(rst))
        pipe_data.send_bytes(rst)


def pipe_smm_func(pipe, shm):
    try:
        while True:
            length = pipe.recv()
            recv = shm.buf[:length]

            if length == 0:
                break
            else:
                rst = np.frombuffer(recv) + 1

            shm.buf[:length] = rst.tobytes()
            pipe.send(length)
    except:
        import traceback

        traceback.print_exc()


def pipe_mmap_func(pipe):
    try:
        f = open("temp/test.mem", "r+b")
        buff = mmap.mmap(f.fileno(), 128)
        while True:
            length = pipe.recv()
            if len(buff) < length:
                buff = mmap.mmap(f.fileno(), length)
            recv = buff[:]

            if length == 0:
                break
            else:
                rst = np.frombuffer(recv) + 1

            buff[:] = rst.tobytes()
            pipe.send(length)
    except:
        import traceback

        traceback.print_exc()
    finally:
        f.close()


def pipe_mmap2_func(item_size):
    try:
        parent = Pipe("temp/parent.mem", 1, item_size, init=False)
        child = Pipe("temp/child.mem", 1, item_size, init=False)
        while True:
            recv = parent.recv(0)

            if recv == b"exit":
                break
            rst = np.frombuffer(recv) + 1

            child.send(0, rst.tobytes())
    except:
        import traceback

        traceback.print_exc()
    finally:
        parent.close()
        child.close()


def pipe_mmap3_func(item_size):
    parent = rPipe("temp/parent.mem", 1, item_size, init=False)
    child = rPipe("temp/child.mem", 1, item_size, init=False)
    while True:
        recv = vec_to_bytes(parent.recv(0))

        if recv == b"exit":
            break
        rst = np.frombuffer(recv) + 1

        child.send(0, rst.tobytes())


def pipe_bytes_comp_func(pipe):
    while True:
        data = pipe.recv_bytes()
        data = blosc2.decompress(data)  # , as_bytearray=True)
        if len(data) == 0:
            break
        data = (np.frombuffer(data) + 1).tobytes()
        data = blosc2.compress(data)  # , shuffle=blosc2.SHUFFLE, cname="lz4")
        pipe.send_bytes(data)


def zmq_func(address):
    try:
        context = zmq.Context()
        sock = context.socket(zmq.REP)
        sock.bind(address)

        while True:
            recv = sock.recv(copy=False)
            if len(recv) == 0:
                break
            rst = np.frombuffer(recv) + 1
            sock.send(rst.tobytes(), copy=False)
    except:
        import traceback

        traceback.print_exc()


if __name__ == "__main__":

    import time

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    import tqdm

    data_size = 10_000_000  # 100_000_000
    n_iters = 10  # int(1e2)
    logs = list()

    #
    # send_bytes + compression
    #

    # parent, child = mp.Pipe()
    # proc = mp.Process(target=pipe_bytes_comp_func, args=(child,))
    # proc.start()
    # for i in tqdm.tqdm(np.arange(n_iters)):
    #     start = time.monotonic()
    #     data = np.random.random(data_size // 2).tobytes()
    #     compressed = blosc2.compress(data)  # , shuffle=blosc2.SHUFFLE, cname="lz4")
    #     parent.send_bytes(compressed)
    #     recved = parent.recv_bytes()
    #     recved = blosc2.decompress(recved)  # , as_bytearray=True)
    #     etime = time.monotonic() - start
    #     logs.append(dict(kind="send_bytes + lz4", etime=etime))
    #     assert np.allclose(np.frombuffer(data) + 1, np.frombuffer(recved))
    # parent.send_bytes(blosc2.compress(b"", shuffle=blosc2.SHUFFLE, cname="lz4"))
    # proc.join()

    #
    # zmq
    #

    address = "tcp://127.0.0.1:6666"

    context = zmq.Context()
    sock = context.socket(zmq.REQ)
    sock.connect(address)

    proc = mp.Process(target=zmq_func, args=(address,))
    proc.start()
    time.sleep(1)
    for i in tqdm.tqdm(np.arange(n_iters)):
        start = time.monotonic()
        data = np.random.random(data_size // 2).tobytes()
        sock.send(data, copy=False)

        recv = sock.recv(copy=False)
        etime = time.monotonic() - start
        logs.append(dict(kind="zmq-tcp", etime=etime))
        assert np.allclose(np.frombuffer(data) + 1, np.frombuffer(recv))
    sock.send(b"")
    proc.join()
    sock.close()
    context.term()

    # ipc
    address = "ipc:///tmp/zmqtest"

    context = zmq.Context()
    sock = context.socket(zmq.REQ)
    sock.connect(address)

    proc = mp.Process(target=zmq_func, args=(address,))
    proc.start()
    time.sleep(1)
    for i in tqdm.tqdm(np.arange(n_iters)):
        start = time.monotonic()
        data = np.random.random(data_size // 2).tobytes()
        sock.send(data, copy=False)

        recv = sock.recv(copy=False)
        etime = time.monotonic() - start
        logs.append(dict(kind="zmq-ipc", etime=etime))
        assert np.allclose(np.frombuffer(data) + 1, np.frombuffer(recv))
    sock.send(b"")
    proc.join()
    sock.close()
    context.term()

    #
    # shared memory
    #

    parent, child = mp.Pipe()
    with SharedMemoryManager() as smm:
        shm = smm.SharedMemory(size=data_size * 8)
        proc = mp.Process(target=pipe_smm_func, args=(child, shm))
        proc.start()
        for i in tqdm.tqdm(np.arange(n_iters)):
            start = time.monotonic()
            data = np.random.random(data_size // 2).tobytes()
            shm.buf[: len(data)] = data
            parent.send(len(data))

            length = parent.recv()
            recv = shm.buf[:length]
            etime = time.monotonic() - start
            logs.append(dict(kind="shm", etime=etime))
            assert np.allclose(np.frombuffer(data) + 1, np.frombuffer(recv))
        parent.send(0)
        proc.join()

    #
    # mmap wrapper 테스트
    #

    # parent = rPipe("temp/parent2.mem", 1, data_size * 8, init=True)
    # child = rPipe("temp/child2.mem", 1, data_size * 8, init=True)
    # # proc = mp.Process(target=pipe_mmap2_func, args=(data_size * 8 // 2,))
    # # proc.start()
    # for i in tqdm.tqdm(np.arange(n_iters)):
    #     start = time.monotonic()
    #     data = np.random.random(data_size // 2).tobytes()

    #     parent.send(0, data)
    #     # recv = vec_to_bytes(parent.recv(0))
    #     recv = parent.recv(0)

    #     etime = time.monotonic() - start
    #     logs.append(dict(kind="rust", etime=etime))

    #     # assert np.allclose(np.frombuffer(data) + 1, np.frombuffer(recv) + 1)
    # parent.send(0, b"exit")
    # # proc.join()
    # parent.close()
    # child.close()

    #
    # mmap wrapper 테스트
    #

    parent = Pipe("temp/parent.mem", 1, data_size * 8, init=True)
    child = Pipe("temp/child.mem", 1, data_size * 8, init=True)
    proc = mp.Process(target=pipe_mmap2_func, args=(data_size * 8 // 2,))
    proc.start()
    for i in tqdm.tqdm(np.arange(n_iters)):
        start = time.monotonic()
        data = np.random.random(data_size // 2).tobytes()

        parent.send(0, data)
        recv = child.recv(0)

        etime = time.monotonic() - start
        logs.append(dict(kind="mmap2", etime=etime))
        assert np.allclose(np.frombuffer(data) + 1, np.frombuffer(recv))
    parent.send(0, b"exit")
    proc.join()
    parent.close()
    child.close()

    #
    # mmap 테스트
    #

    parent, child = mp.Pipe()
    with open("temp/test.mem", "wb") as f:
        f.write(b"0" * data_size * 8)
    f = open("temp/test.mem", "r+b")
    buff = mmap.mmap(f.fileno(), data_size * 8)
    proc = mp.Process(target=pipe_mmap_func, args=(child,))
    proc.start()
    for i in tqdm.tqdm(np.arange(n_iters)):
        start = time.monotonic()
        data = np.random.random(data_size // 2).tobytes()
        if len(data) > len(buff):
            buff = mmap.mmap(f.fileno(), len(data))
        buff[: len(data)] = data
        parent.send(len(data))

        length = parent.recv()
        recv = buff[:length]
        etime = time.monotonic() - start
        logs.append(dict(kind="mmap", etime=etime))
        assert np.allclose(np.frombuffer(data) + 1, np.frombuffer(recv))
    parent.send(0)
    proc.join()
    f.close()

    #
    # send 테스트
    #

    parent, child = mp.Pipe()
    proc = mp.Process(target=pipe_func, args=(child,))
    proc.start()
    for i in tqdm.tqdm(np.arange(n_iters)):
        start = time.monotonic()
        data = np.random.random(data_size // 2).tobytes()
        parent.send(data)
        recved = parent.recv()
        etime = time.monotonic() - start
        logs.append(dict(kind="send", etime=etime))
        assert np.allclose(np.frombuffer(data) + 1, np.frombuffer(recved))
    parent.send(b"")
    proc.join()

    #
    # send_bytes 테스트
    #

    parent, child = mp.Pipe()
    proc = mp.Process(target=pipe_bytes_func, args=(child,))
    proc.start()
    for i in tqdm.tqdm(np.arange(n_iters)):
        start = time.monotonic()
        data = np.random.random(data_size // 2).tobytes()
        parent.send_bytes(data)
        recved = parent.recv_bytes()
        etime = time.monotonic() - start
        logs.append(dict(kind="send_bytes", etime=etime))
        assert np.allclose(np.frombuffer(data) + 1, np.frombuffer(recved))
    parent.send_bytes(b"")
    proc.join()

    logs = pd.DataFrame(logs)
    etimes = logs.groupby("kind").mean()["etime"]
    order = etimes.index[etimes.argsort()]
    sns.barplot(data=logs, x="kind", y="etime", order=order)
    # plt.show()
    plt.savefig("shard.png")

    # embed()
    # exit()

