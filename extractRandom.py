import numpy as np
import cv2

def extractColorHistogram(img, bins_per_channel=8):
    # Compute histogram for each channel (R, G, B)
    hist_r = cv2.calcHist([img], [0], None, [bins_per_channel], [0, 1])
    hist_g = cv2.calcHist([img], [1], None, [bins_per_channel], [0, 1])
    hist_b = cv2.calcHist([img], [2], None, [bins_per_channel], [0, 1])
    
    # Concatenate and normalize the histograms
    hist = np.concatenate([hist_r, hist_g, hist_b]).flatten()
    hist /= hist.sum()  # Normalize to sum to 1
    
    return hist