import numpy as np
import cv2


def extractColorHistogram(img, bins_per_channel=8):
    # Convert image to uint8 if necessary
    if img.dtype != np.uint8:
        img = (img * 255).astype('uint8')

    # Compute histogram for each channel (R, G, B)
    hist_r = cv2.calcHist([img], [0], None, [bins_per_channel], [0, 256])
    hist_g = cv2.calcHist([img], [1], None, [bins_per_channel], [0, 256])
    hist_b = cv2.calcHist([img], [2], None, [bins_per_channel], [0, 256])

    # Concatenate and normalize the histograms
    hist = np.concatenate([hist_r, hist_g, hist_b]).flatten()
    hist /= (hist.sum() + 1e-5)  # Normalize to sum to 1, avoid division by zero

    return hist
