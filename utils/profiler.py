from contextlib import contextmanager
import time


@contextmanager
def profile(label):
    start_time = time.time()
    try:
        yield
    finally:
        elapsed_time = (time.time() - start_time) * 1000.
        print(f'<{label}> {elapsed_time:.4f} ms')
