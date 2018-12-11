# -*- coding: utf-8 -*-

import pickle

from .. import settings as stg


def scatter_dataset(dataset, root=0, max_buf_len=256 * 1024 * 1024):
    """Scatter the given dataset to the workers in the communicator.

    refer to chainermn.scatter_dataset()
    change:
        broadcast dataset by split size, NOT whole size, to the workers.
        omit shuffling dataset
        use raw mpi4py method to send/recv dataset
    """
    assert 0 <= root < stg.mpi.size
    stg.mpi.comm.Barrier()

    if stg.mpi.rank == root:
        mine = None
        n_total_samples = len(dataset)
        n_sub_samples = (n_total_samples + stg.mpi.size - 1) // stg.mpi.size

        for i in range(stg.mpi.size):
            b = n_total_samples * i // stg.mpi.size
            e = b + n_sub_samples

            if i == root:
                mine = dataset.take(slice(b, e, None))
            else:
                send = dataset.take(slice(b, e, None))
                send_chunk(send, dest=i, max_buf_len=max_buf_len)
        assert mine is not None
        return mine

    else:
        recv = recv_chunk(source=root, max_buf_len=max_buf_len)
        assert recv is not None
        return recv


INT_MAX = 2147483647


def send_chunk(obj, dest, max_buf_len=256 * 1024 * 1024):
    assert max_buf_len < INT_MAX
    assert max_buf_len > 0
    pickled_bytes = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    total_bytes = len(pickled_bytes)
    total_chunk_num = -(-total_bytes // max_buf_len)
    stg.mpi.comm.send((total_chunk_num, max_buf_len, total_bytes), dest=dest, tag=1)

    for i in range(total_chunk_num):
        b = i * max_buf_len
        e = min(b + max_buf_len, total_bytes)
        buf = pickled_bytes[b:e]
        stg.mpi.comm.Send(buf, dest=dest, tag=2)


def recv_chunk(source, max_buf_len=256 * 1024 * 1024):
    assert max_buf_len < INT_MAX
    assert max_buf_len > 0
    data = stg.mpi.comm.recv(source=source, tag=1)
    assert data is not None
    total_chunk_num, max_buf_len, total_bytes = data
    pickled_bytes = bytearray()

    for i in range(total_chunk_num):
        b = i * max_buf_len
        e = min(b + max_buf_len, total_bytes)
        buf = bytearray(e - b)
        stg.mpi.comm.Recv(buf, source=source, tag=2)
        pickled_bytes[b:e] = buf

    obj = pickle.loads(pickled_bytes)
    return obj
