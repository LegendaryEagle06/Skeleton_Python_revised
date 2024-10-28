import numpy as np

def cvpr_compare(F1, F2):
    # Compute the L1 distance between F1 and F2
    dst = np.sum(np.abs(F1 - F2))
    return dst
