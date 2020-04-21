import os

def make_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def set_batches_per_epoch(batch_size):
    """ Only powers of two please. """
    total = 16 * 1024
    return int(total / batch_size)
