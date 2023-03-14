import mmap
from IPython import embed
from pathlib import Path


class SharedMemory:
    def __init__(self, pipe, mmap_path, n_items, item_size, init) -> None:
        self.pipe = pipe
        self.n_items = n_items
        self.item_size = item_size
        if init:
            with open(mmap_path, "wb") as f:
                f.write(b"0" * (self.n_items * self.item_size))
        self.f = open(mmap_path, "r+b")
        self.buff = mmap.mmap(self.f.fileno(), self.n_items * self.item_size)

    def close(self):
        self.buff.close()
        self.f.close()

    def set(self, idx, data):
        assert len(data) <= self.item_size
        self.buff[(self.n_items * idx) : (self.n_items * idx) + len(data)] = data
        self.pipe.send(len(data))

    def get(self, idx):
        size = self.pipe.recv()
        return self.buff[(self.n_items * idx) : (self.n_items * idx) + size]


class Pipe:
    def __init__(self, path, n_items, item_size, init) -> None:
        self.n_items = n_items
        self.int_size = 8
        self.item_size = item_size
        self.block_size = self.int_size + self.n_items

        mem_size = self.n_items * (self.int_size + self.item_size)

        # 초기화
        path = Path(path)
        if init:
            path.write_bytes(b"\x00" * mem_size)

        # 버퍼 생성
        self.f = path.open("r+b")
        self.buff = mmap.mmap(self.f.fileno(), mem_size)

    def close(self):
        self.buff.close()
        self.f.close()

    def send(self, idx, data):
        data_size = len(data)
        assert data_size <= self.item_size
        # ext.cset(self.buff, self.n_items, idx, data, len(data))
        size_start = self.block_size * idx
        size_end = size_start + self.int_size
        data_start = self.block_size * idx + self.int_size
        data_end = data_start + data_size
        while True:
            size = encode_int(self.buff[size_start:size_end])
            if size == 0:
                break
        self.buff[data_start:data_end] = data
        self.buff[size_start:size_end] = decode_int(data_size)

    def recv(self, idx):
        # return ext.cget(self.buff, self.n_items, idx)
        size_start = self.block_size * idx
        size_end = size_start + self.int_size
        while True:
            data_size = encode_int(self.buff[size_start:size_end])
            if data_size != 0:
                break
        data_start = self.block_size * idx + self.int_size
        data_end = data_start + data_size
        data = self.buff[data_start:data_end]
        self.buff[size_start:size_end] = b"\x00" * self.int_size
        return data


def decode_int(value):
    return (value).to_bytes(8, byteorder="little", signed=False)


def encode_int(bytes_array):
    return int.from_bytes(bytes(bytes_array), byteorder="little", signed=False)

