import numpy as np
from scipy.spatial import distance

def cvpr_compare(F1, F2, covariance_matrix):
    # Compute Mahalanobis distance between F1 and F2
    dst = distance.mahalanobis(F1, F2, np.linalg.inv(covariance_matrix))
    return dst
