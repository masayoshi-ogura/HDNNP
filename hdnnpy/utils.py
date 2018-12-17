# coding: utf-8

import pickle
from pprint import pprint as pretty_print
import sys

from mpi4py import MPI as MPI4PY


INT_MAX = 2147483647


class MPI:
    comm = MPI4PY.COMM_WORLD
    rank = MPI4PY.COMM_WORLD.Get_rank()
    size = MPI4PY.COMM_WORLD.Get_size()


def mkdir(path):
    if MPI.rank == 0:
        path.mkdir(parents=True, exist_ok=True)


def pprint(data=None, flush=True, **options):
    if data is None:
        data = ''
    if isinstance(data, list) or isinstance(data, dict):
        pretty_print(data, **options)
    else:
        if 'stream' in options:
            options['file'] = options.pop('stream')
        print(data, **options)
    if flush:
        sys.stdout.flush()


def recv_chunk(source, max_buf_len=256 * 1024 * 1024):
    assert max_buf_len < INT_MAX
    assert max_buf_len > 0
    data = MPI.comm.recv(source=source, tag=1)
    assert data is not None
    total_chunk_num, max_buf_len, total_bytes = data
    pickled_bytes = bytearray()

    for i in range(total_chunk_num):
        b = i * max_buf_len
        e = min(b + max_buf_len, total_bytes)
        buf = bytearray(e - b)
        MPI.comm.Recv(buf, source=source, tag=2)
        pickled_bytes[b:e] = buf

    obj = pickle.loads(pickled_bytes)
    return obj


def send_chunk(obj, dest, max_buf_len=256 * 1024 * 1024):
    assert max_buf_len < INT_MAX
    assert max_buf_len > 0
    pickled_bytes = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    total_bytes = len(pickled_bytes)
    total_chunk_num = -(-total_bytes // max_buf_len)
    MPI.comm.send(
        (total_chunk_num, max_buf_len, total_bytes), dest=dest, tag=1)

    for i in range(total_chunk_num):
        b = i * max_buf_len
        e = min(b + max_buf_len, total_bytes)
        buf = pickled_bytes[b:e]
        MPI.comm.Send(buf, dest=dest, tag=2)
