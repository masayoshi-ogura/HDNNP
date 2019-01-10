# coding: utf-8

"""Utility functions used in various subpackages."""

__all__ = [
    'MPI',
    'pprint',
    'pyyaml_path_constructor',
    'pyyaml_path_representer',
    'recv_chunk',
    'send_chunk',
    ]

from pathlib import Path
import pickle
from pprint import pprint as pretty_print
import sys
import textwrap

from mpi4py import MPI as MPI4PY


INT_MAX = 2147483647


class MPI:
    """MPI world communicator and aliases."""
    comm = MPI4PY.COMM_WORLD
    rank = MPI4PY.COMM_WORLD.Get_rank()
    size = MPI4PY.COMM_WORLD.Get_size()


def pprint(data=None, flush=True, **options):
    """Pretty print function.

    Args:
        data (str, optional): Data to output into stdout.
        flush (bool, optional): Flush the stream after output if True.
        **options: Other options passed to :meth:`print`.
    """
    if data is None:
        data = ''
    data = textwrap.dedent(data)
    if isinstance(data, list) or isinstance(data, dict):
        pretty_print(data, **options)
    else:
        if 'stream' in options:
            options['file'] = options.pop('stream')
        print(data, **options)
    if flush:
        sys.stdout.flush()


def pyyaml_path_constructor(loader, node):
    """Helper method to load Path tag in PyYAML."""
    value = loader.construct_scalar(node)
    return Path(value)


def pyyaml_path_representer(dumper, instance):
    """Helper method to dump :class:`~pathlib.Path` in PyYAML."""
    return dumper.represent_scalar('Path', f'{instance}')


def recv_chunk(source, max_buf_len=256 * 1024 * 1024):
    """Receive data divided into small chunks with MPI communication.

    Args:
        source (int): MPI source process that sends data.
        max_buf_len (int, optional): Maximum size of each chunk.

    Returns:
        object: Received data.
    """
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
    """Send data divided into small chunks with MPI communication.

    Args:
        obj (object): Any data to send, which can be pickled.
        dest (int): MPI destination process that receives data.
        max_buf_len (int, optional): Maximum size of each chunk.
    """
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
