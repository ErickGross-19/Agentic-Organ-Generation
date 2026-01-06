import time
from contextlib import contextmanager


@contextmanager
def timed_stage(name: str):
    """
    Context manager to print a simple progress message and timing
    for a pipeline stage.
    """
    print(f"[{name}] started...")
    t0 = time.time()
    try:
        yield
    finally:
        dt = time.time() - t0
        print(f"[{name}] finished in {dt:.2f} s")
