import errno
import mmap
import multiprocessing as mp
import pathlib
import pickle
import sys
import threading
import timeit
from multiprocessing import shared_memory
from multiprocessing.managers import SharedMemoryManager

import blosc2
import numpy as np
from IPython import embed
from numpy.random import shuffle

from .shared_memory import Pipe, SharedMemory


class SimpleBytesQueue:
    def __init__(self):
        self.queue = mp.SimpleQueue()

    def get_bytes(self):
        with self.queue._rlock:
            res = self.queue._reader.recv_bytes()
        return res

    def put_bytes(self, obj):
        if self.queue._wlock is None:
            # writes to a message oriented win32 pipe are atomic
            self.queue._writer.send_bytes(obj)
        else:
            with self.queue._wlock:
                self.queue._writer.send_bytes(obj)


_sentinel = object()


class BytesQueue:
    def __init__(self, max_size=100):
        self.queue = mp.Queue(max_size)

    def put_bytes(self, obj):
        if not self.queue._sem.acquire(True, None):
            raise mp.Queue.Full

        with self.queue._notempty:
            if self.queue._thread is None:
                # Start thread which transfers data from buffer to pipe
                self.queue._buffer.clear()
                self._thread = threading.Thread(
                    target=BytesQueue._feed,
                    args=(
                        self.queue._buffer,
                        self.queue._notempty,
                        self.queue._send_bytes,
                        self.queue._wlock,
                        self.queue._writer.close,
                        self.queue._ignore_epipe,
                        self.queue._on_queue_feeder_error,
                        self.queue._sem,
                    ),
                    name="QueueFeederThread",
                )
                self._thread.daemon = True
                self._thread.start()
            self.queue._buffer.append(obj)
            self.queue._notempty.notify()

    def get_bytes(self):
        with self.queue._rlock:
            res = self.queue._recv_bytes()
        self.queue._sem.release()
        return res

    @staticmethod
    def _feed(
        buffer, notempty, send_bytes, writelock, close, ignore_epipe, onerror, queue_sem
    ):
        nacquire = notempty.acquire
        nrelease = notempty.release
        nwait = notempty.wait
        bpopleft = buffer.popleft
        sentinel = _sentinel
        if sys.platform != "win32":
            wacquire = writelock.acquire
            wrelease = writelock.release
        else:
            wacquire = None

        while 1:
            try:
                nacquire()
                try:
                    if not buffer:
                        nwait()
                finally:
                    nrelease()
                try:
                    while 1:
                        obj = bpopleft()
                        if obj is sentinel:
                            close()
                            return

                        # serialize the data before acquiring the lock
                        if wacquire is None:
                            send_bytes(obj)
                        else:
                            wacquire()
                            try:
                                send_bytes(obj)
                            finally:
                                wrelease()
                except IndexError:
                    pass
            except Exception as e:
                if ignore_epipe and getattr(e, "errno", 0) == errno.EPIPE:
                    return
                # Since this runs in a daemon thread the resources it uses
                # may be become unusable while the process is cleaning up.
                # We ignore errors which happen after the process has
                # started to cleanup.
                if is_exiting():
                    return
                else:
                    # Since the object has not been sent in the queue, we need
                    # to decrease the size of the queue. The error acts as
                    # if the object had been silently removed from the queue
                    # and this step is necessary to have a properly working
                    # queue.
                    queue_sem.release()
                    onerror(e, obj)


class SimpleBytesMemQueue:
    def __init__(self, size):
        self.queue = mp.SimpleQueue()
        self.buffer = shared_memory.SharedMemory(create=True, size=size)

    def get_bytes(self):
        with self.queue._rlock:
            res = self.queue._reader.recv_bytes()
            dsize = pickle.loads(res)
            res = self.buffer[:dsize]
        return res

    def put_bytes(self, obj):
        dsize = len(obj)

        if self.queue._wlock is None:
            # writes to a message oriented win32 pipe are atomic
            self.buffer[:] = obj
            self.queue._writer.send_bytes(pickle.dumps(dsize))
        else:
            with self.queue._wlock:
                self.buffer[:] = obj
                self.queue._writer.send_bytes(pickle.dumps(dsize))


def func_queue(parent, child):
    while True:
        data = parent.get()
        if len(data) == 0:
            break
        data = (np.frombuffer(data) + 1).tobytes()
        child.put(data)


def func_bytes_queue(parent, child):
    while True:
        data = parent.get_bytes()
        if len(data) == 0:
            break
        data = (np.frombuffer(data) + 1).tobytes()
        child.put_bytes(data)


def func_smm_queue(sm, pipe):
    while True:
        dsize = pipe.recv()
        if dsize == 0:
            break
        data = sm.buf[:dsize]
        data = (np.frombuffer(data) + 1).tobytes()
        dsize = len(data)
        sm.buf[:dsize] = data
        pipe.send(dsize)


if __name__ == "__main__":

    import time

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    import tqdm

    data_size = 10_000_000  # 100_000_000
    n_iters = 100  # int(1e2)
    logs = list()

    data = np.random.random(data_size // 2).tobytes()
    task_etime = timeit.timeit(
        stmt="(np.frombuffer(data) + 1).tobytes()",
        setup="import numpy as np; data_size = 10_000_000; data = np.random.random(data_size // 2).tobytes()",
        number=100,
    )
    task_etime = task_etime / 100

    #
    # Simple Bytes Mem Queue
    #

    with SharedMemoryManager() as smm:
        sm = smm.SharedMemory(data_size * 8)
        parent, child = mp.Pipe()
        proc = mp.Process(target=func_smm_queue, args=(sm, child,))
        proc.start()
        for i in tqdm.tqdm(np.arange(n_iters)):
            start = time.monotonic()
            data = np.random.random(data_size // 2).tobytes()
            dsize = len(data)
            sm.buf[:dsize] = data
            parent.send(len(data))
            dsize = parent.recv()
            recved = sm.buf[:dsize]
            etime = time.monotonic() - start
            logs.append(dict(kind="shared_memory", etime=etime - task_etime))
            assert np.allclose(np.frombuffer(data) + 1, np.frombuffer(recved))
        parent.send(0)
        proc.join()

    #
    # Bytes Queue
    #

    parent = BytesQueue()
    child = BytesQueue()
    proc = mp.Process(target=func_bytes_queue, args=(parent, child,))
    proc.start()
    for i in tqdm.tqdm(np.arange(n_iters)):
        start = time.monotonic()
        data = np.random.random(data_size // 2).tobytes()
        parent.put_bytes(data)
        recved = child.get_bytes()
        etime = time.monotonic() - start
        logs.append(dict(kind="bytes_queue", etime=etime - task_etime))
        assert np.allclose(np.frombuffer(data) + 1, np.frombuffer(recved))
    parent.put_bytes(b"")
    proc.join()

    #
    # Simple Bytes Queue
    #

    parent = SimpleBytesQueue()
    child = SimpleBytesQueue()
    proc = mp.Process(target=func_bytes_queue, args=(parent, child,))
    proc.start()
    for i in tqdm.tqdm(np.arange(n_iters)):
        start = time.monotonic()
        data = np.random.random(data_size // 2).tobytes()
        parent.put_bytes(data)
        recved = child.get_bytes()
        etime = time.monotonic() - start
        logs.append(dict(kind="simple_bytes_queue", etime=etime - task_etime))
        assert np.allclose(np.frombuffer(data) + 1, np.frombuffer(recved))
    parent.put_bytes(b"")
    proc.join()

    #
    # Simple Queue
    #

    parent = mp.SimpleQueue()
    child = mp.SimpleQueue()
    proc = mp.Process(target=func_queue, args=(parent, child,))
    proc.start()
    for i in tqdm.tqdm(np.arange(n_iters)):
        start = time.monotonic()
        data = np.random.random(data_size // 2).tobytes()
        parent.put(data)
        recved = child.get()
        etime = time.monotonic() - start
        logs.append(dict(kind="simple_queue", etime=etime - task_etime))
        assert np.allclose(np.frombuffer(data) + 1, np.frombuffer(recved))
    parent.put(b"")
    proc.join()

    #
    # Queue
    #

    parent = mp.Queue(maxsize=100)
    child = mp.Queue(maxsize=100)
    proc = mp.Process(target=func_queue, args=(parent, child,))
    proc.start()
    for i in tqdm.tqdm(np.arange(n_iters)):
        start = time.monotonic()
        data = np.random.random(data_size // 2).tobytes()
        parent.put(data)
        recved = child.get()
        etime = time.monotonic() - start
        logs.append(dict(kind="queue", etime=etime - task_etime))
        assert np.allclose(np.frombuffer(data) + 1, np.frombuffer(recved))
    parent.put(b"")
    proc.join()

    logs = pd.DataFrame(logs)
    etimes = logs.groupby("kind").mean()["etime"]
    order = etimes.index[etimes.argsort()]
    sns.boxplot(data=logs, x="kind", y="etime", order=order, showfliers=False)
    plt.show()

    embed()
    exit()
