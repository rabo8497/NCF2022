import mmap
import multiprocessing as mp
import pathlib

import numpy as np
from IPython import embed


class Fifo1:
    def __init__(self, maxsize) -> None:

        self.maxsize = maxsize
        self.buffer = [None] * maxsize
        self.first = 0
        self.last = 0
        self.n_items = 0

    def put(self, item):
        self.buffer[self.last] = item
        self.last = (self.last + 1) % self.maxsize
        self.n_items += 1
        if self.n_items > self.maxsize:
            self.n_items = self.maxsize
            self.first = (self.first + 1) % self.maxsize

    def get(self):
        if self.n_items > 0:
            item = self.buffer[self.first]
            self.first = (self.first + 1) % self.maxsize
            self.n_items -= 1
            if self.n_items < 0:
                self.n_items = 0
            return item
        else:
            return None

    def qsize(self):
        return self.n_items

    def empty(self):
        return self.n_items == 0

    def full(self):
        return self.n_items == self.maxsize

    def __repr__(self) -> str:
        return f"{self.buffer}"


class Fifo2:
    def __init__(self, maxsize) -> None:

        self.maxsize = maxsize
        self.item_size = [0 for _ in range(self.maxsize)]
        self.buffer = [bytearray(128) for _ in range(self.maxsize)]
        self.first = 0
        self.last = 0
        self.n_items = 0

    def put(self, item):
        item_size = len(item)
        self.item_size[self.last] = item_size
        buffer = self.buffer[self.last]
        if len(buffer) < item_size:
            buffer = bytearray(int(item_size * 1.1))
        buffer[:item_size] = item
        self.buffer[self.last] = buffer
        self.last = (self.last + 1) % self.maxsize
        self.n_items += 1
        if self.n_items > self.maxsize:
            self.n_items = self.maxsize
            self.first = (self.first + 1) % self.maxsize

    def get(self):
        if self.n_items > 0:
            item_size = self.item_size[self.first]
            self.item_size[self.first] = 0
            item = bytes(self.buffer[self.first][:item_size])
            self.first = (self.first + 1) % self.maxsize
            self.n_items -= 1
            if self.n_items < 0:
                self.n_items = 0
            return item
        else:
            return None

    def qsize(self):
        return self.n_items

    def empty(self):
        return self.n_items == 0

    def full(self):
        return self.n_items == self.maxsize

    def __repr__(self) -> str:
        return f"{self.buffer}"


class Fifo3:
    def __init__(self, maxsize) -> None:
        # self.root = root
        self.maxsize = maxsize
        self.meta_size = 8
        self.buffer = [bytearray(self.meta_size + 128) for _ in range(self.maxsize)]
        self.first = 0
        self.last = 0
        self.n_items = 0

    def to_bytes(self, item_size):
        return (item_size).to_bytes(self.meta_size, byteorder="little", signed=False)

    def from_bytes(self, int_byte):
        return int.from_bytes(int_byte, byteorder="little", signed=False)

    def put(self, item):
        item_size = len(item)
        buffer = self.buffer[self.last]
        if len(buffer) < (self.meta_size + item_size):
            buffer = bytearray(self.meta_size + int(item_size * 1.1))
        buffer[: self.meta_size] = self.to_bytes(item_size)
        buffer[self.meta_size : item_size] = item
        self.buffer[self.last] = buffer
        self.last = (self.last + 1) % self.maxsize
        self.n_items += 1
        if self.n_items > self.maxsize:
            self.n_items = self.maxsize
            self.first = (self.first + 1) % self.maxsize

    def get(self):
        if self.n_items > 0:
            item_size = self.from_bytes(self.buffer[self.first][: self.meta_size])
            item = bytes(self.buffer[self.first][self.meta_size : item_size])
            self.first = (self.first + 1) % self.maxsize
            self.n_items -= 1
            if self.n_items < 0:
                self.n_items = 0
            return item
        else:
            return None

    def qsize(self):
        return self.n_items

    def empty(self):
        return self.n_items == 0

    def full(self):
        return self.n_items == self.maxsize

    def __repr__(self) -> str:
        return f"{self.buffer}"


class Fifo4:
    def __init__(self, maxsize, root) -> None:
        self.root = root
        self.maxsize = maxsize
        self.meta_size = 8
        self.buffer = [mmap.mmap(-1, self.meta_size + 128) for _ in range(self.maxsize)]
        self.first = 0
        self.last = 0
        self.n_items = 0

    def to_bytes(self, item_size):
        return (item_size).to_bytes(self.meta_size, byteorder="little", signed=False)

    def from_bytes(self, int_byte):
        return int.from_bytes(int_byte, byteorder="little", signed=False)

    def put(self, item):
        item_size = len(item)
        buffer = self.buffer[self.last]
        if len(buffer) < (self.meta_size + item_size):
            buffer = mmap.mmap(-1, self.meta_size + int(item_size * 1.1))
        buffer[: self.meta_size] = self.to_bytes(item_size)
        buffer[self.meta_size : self.meta_size + item_size] = item
        self.buffer[self.last] = buffer
        self.last = (self.last + 1) % self.maxsize
        self.n_items += 1
        if self.n_items > self.maxsize:
            self.n_items = self.maxsize
            self.first = (self.first + 1) % self.maxsize

    def get(self):
        if self.n_items > 0:
            item_size = self.from_bytes(self.buffer[self.first][: self.meta_size])
            item = bytes(
                self.buffer[self.first][self.meta_size : self.meta_size + item_size]
            )
            self.first = (self.first + 1) % self.maxsize
            self.n_items -= 1
            if self.n_items < 0:
                self.n_items = 0
            return item
        else:
            return None

    def qsize(self):
        return self.n_items

    def empty(self):
        return self.n_items == 0

    def full(self):
        return self.n_items == self.maxsize

    def __repr__(self) -> str:
        return f"{self.buffer}"


class Fifo5:
    def __init__(self, maxsize, root: pathlib.Path) -> None:
        self.root = root
        self.maxsize = maxsize
        self.uint_size = 8
        root.mkdir(parents=True, exist_ok=True)

        fos = list()
        for i in range(self.maxsize + 1):
            buff = root / f"{i}.mem"
            if not buff.exists():
                buff.write_bytes(bytearray(self.uint_size + 128))
            f = buff.open("r+b")
            fos.append(f)

        self.buffer = list()
        for i, f in enumerate(fos):
            self.buffer.append(mmap.mmap(f.fileno(), self.uint_size + 128))

        meta = self.buffer[-1]
        self.first = self.from_bytes(meta[: self.uint_size])
        self.last = self.from_bytes(meta[self.uint_size : 2 * self.uint_size])
        self.n_items = self.from_bytes(meta[2 * self.uint_size : 3 * self.uint_size])

    def to_bytes(self, item_size):
        return (item_size).to_bytes(self.uint_size, byteorder="little", signed=False)

    def from_bytes(self, int_byte):
        return int.from_bytes(int_byte, byteorder="little", signed=False)

    def put(self, item):
        item_size = len(item)
        buffer = self.buffer[self.last]
        if len(buffer) < (self.uint_size + item_size):
            buffer.resize(self.meta_size + int(item_size * 1.1))
            # buffer = mmap.mmap(-1, self.meta_size + int(item_size * 1.1))
        buffer[: self.uint_size] = self.to_bytes(item_size)
        buffer[self.uint_size : self.uint_size + item_size] = item
        self.buffer[self.last] = buffer
        self.last = (self.last + 1) % self.maxsize
        self.n_items += 1
        if self.n_items > self.maxsize:
            self.n_items = self.maxsize
            self.first = (self.first + 1) % self.maxsize

    def get(self):
        if self.n_items > 0:
            item_size = self.from_bytes(self.buffer[self.first][: self.uint_size])
            item = bytes(
                self.buffer[self.first][self.uint_size : self.uint_size + item_size]
            )
            self.first = (self.first + 1) % self.maxsize
            self.n_items -= 1
            if self.n_items < 0:
                self.n_items = 0
            return item
        else:
            return None

    def qsize(self):
        return self.n_items

    def empty(self):
        return self.n_items == 0

    def full(self):
        return self.n_items == self.maxsize

    def __repr__(self) -> str:
        return f"{self.buffer}"


def print_id(l):
    print(id(l))
    print(l)


if __name__ == "__main__":

    import time

    import tqdm

    q0 = mp.Queue(maxsize=100)
    q1 = Fifo1(maxsize=100)
    q2 = Fifo2(maxsize=100)
    q3 = Fifo3(maxsize=100)
    q4 = Fifo4(maxsize=100, root="")
    q5 = Fifo5(maxsize=100, root=pathlib.Path("temp"))
    qs = [q0, q1, q2, q3, q4, q5]

    vs = [None] * len(qs)
    stime = np.zeros(len(qs))
    gtime = np.zeros(len(qs))

    n_exprs = 10

    for i in tqdm.tqdm(np.arange(1e4)):
        for j in range(n_exprs):
            value = np.random.random(1000).tobytes()

            for k, q in enumerate(qs):
                start = time.monotonic()
                q.put(value)
                stime[k] += (time.monotonic() - start) / n_exprs

        for j in range(n_exprs):
            for k, q in enumerate(qs):
                start = time.monotonic()
                vs[k] = q.get()
                gtime[k] += (time.monotonic() - start) / n_exprs

            try:
                for k in range(1, len(qs)):
                    assert vs[0] == vs[k]
            except:
                # embed()
                # exit()
                pass

    print(stime / stime.min())
    print(gtime / gtime.min())
    ttime = stime + gtime
    print(ttime / ttime.min())

    embed()
    exit()

