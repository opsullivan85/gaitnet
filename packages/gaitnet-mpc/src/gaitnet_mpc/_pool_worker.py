"""Entry point of a VectorPool worker process, see `gaitnet_mpc.pool.VectorPool._setup_workers`.

Run as a script rather than with -m: the manager sends its sys.path first, and only
then is anything from gaitnet_mpc imported. Passing the path through PYTHONPATH instead
could exceed the environment's size limit under Isaac Sim, whose sys.path holds
hundreds of extension folders.

usage: python _pool_worker.py <socket fd> <worker id>
"""

import sys
from multiprocessing.connection import Connection


def main() -> None:
    fd, worker_id = int(sys.argv[1]), int(sys.argv[2])
    conn = Connection(fd)
    try:
        sys.path[:] = conn.recv()
        pool_cls, cls, num_objects, kwargs = conn.recv()
    except EOFError:
        return  # the manager exited before sending the worker its setup
    pool_cls._worker_loop(conn, worker_id, cls, num_objects, kwargs)


if __name__ == "__main__":
    main()
