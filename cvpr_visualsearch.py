import os
import numpy as np
import scipy.io as sio
import cv2
from random import randint
from cvpr_compare import cvpr_compare
from sklearn.metrics import precision_recall_curve, confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

DESCRIPTOR_FOLDER = 'descriptors'
DESCRIPTOR_SUBFOLDER = 'globalRGBhisto'
IMAGE_FOLDER = 'MSRC_ObjCategImageDatabase_v2'

# Load all descriptors
ALLFEAT = []
ALLFILES = []
for filename in os.listdir(os.path.join(DESCRIPTOR_FOLDER, DESCRIPTOR_SUBFOLDER)):
    if filename.endswith('.mat'):
        img_path = os.path.join(DESCRIPTOR_FOLDER, DESCRIPTOR_SUBFOLDER, filename)
        img_actual_path = os.path.join(IMAGE_FOLDER, 'Images', filename.replace(".mat", ".bmp"))
        img_data = sio.loadmat(img_path)
        if 'F' in img_data:
            ALLFILES.append(img_actual_path)
            ALLFEAT.append(img_data['F'].flatten())  # Ensure F is flattened

# Convert ALLFEAT to a numpy array
ALLFEAT = np.array(ALLFEAT)

# Apply PCA for dimensionality reduction
if ALLFEAT.shape[0] > 0:
    pca = PCA(n_components=min(15, ALLFEAT.shape[1]))
    ALLFEAT = pca.fit_transform(ALLFEAT)

    # Compute the covariance matrix for Mahalanobis distance
    covariance_matrix = np.cov(ALLFEAT, rowvar=False)

    # Pick a random image as the query
    NIMG = ALLFEAT.shape[0]
    queryimg = randint(0, NIMG - 1)

    # Compute the distance between the query and all other descriptors
    dst = []
    query = ALLFEAT[queryimg]
    for i in range(NIMG):
        candidate = ALLFEAT[i]
        distance = cvpr_compare(query, candidate, covariance_matrix)
        dst.append((distance, i))

    # Sort the distances
    dst.sort(key=lambda x: x[0])

    # Show the top 15 results
    SHOW = min(15, NIMG)
    for i in range(SHOW):
        img = cv2.imread(ALLFILES[dst[i][1]])
        if img is not None:
            img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))  # Make image quarter size
            cv2.imshow(f"Result {i + 1}", img)
            cv2.setWindowTitle(f"Result {i + 1}", f"Result {i + 1}")
            cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Precision-Recall Evaluation
    relevant_images = [0 for _ in range(NIMG)]  # Assume none are relevant initially
    predicted_scores = [1 / (x[0] + 1e-5) for x in dst]  # Inverse of distance as relevance score

    precision, recall, _ = precision_recall_curve(relevant_images, predicted_scores)

    # Plot Precision-Recall curve
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()

    # Confusion Matrix Evaluation
    # Define a threshold for classification (e.g., top 10 images are considered as relevant)
    threshold = sorted(predicted_scores, reverse=True)[9]  # Score of the 10th highest-ranked image
    predicted_labels = [1 if score >= threshold else 0 for score in predicted_scores]

    # Compute the confusion matrix
    cm = confusion_matrix(relevant_images, predicted_labels)

    # Plot the confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()


# Spatial Grid Extraction Function
def extract_spatial_grid_histogram(img, bins_per_channel=8, grid_size=3):
    # Convert image to uint8 if necessary
    if img.dtype != np.uint8:
        img = (img * 255).astype('uint8')

    # Divide the image into a grid
    h, w, _ = img.shape
    grid_h, grid_w = h // grid_size, w // grid_size

    hist = []
    for row in range(grid_size):
        for col in range(grid_size):
            # Extract the grid cell
            cell = img[row * grid_h:(row + 1) * grid_h, col * grid_w:(col + 1) * grid_w]

            # Compute histogram for each channel (R, G, B)
            hist_r = cv2.calcHist([cell], [0], None, [bins_per_channel], [0, 256])
            hist_g = cv2.calcHist([cell], [1], None, [bins_per_channel], [0, 256])
            hist_b = cv2.calcHist([cell], [2], None, [bins_per_channel], [0, 256])

            # Concatenate and normalize the histograms
            cell_hist = np.concatenate([hist_r, hist_g, hist_b]).flatten()
            cell_hist /= (cell_hist.sum() + 1e-5)  # Normalize to sum to 1, avoid division by zero
            hist.extend(cell_hist)

    return np.array(hist)


# Update feature extraction to use spatial grid
for filename in os.listdir(os.path.join(DESCRIPTOR_FOLDER, DESCRIPTOR_SUBFOLDER)):
    if filename.endswith('.mat'):
        img_path = os.path.join(DESCRIPTOR_FOLDER, DESCRIPTOR_SUBFOLDER, filename)
        img_actual_path = os.path.join(IMAGE_FOLDER, 'Images', filename.replace(".mat", ".bmp"))
        img = cv2.imread(img_actual_path)
        if img is not None:
            F = extract_spatial_grid_histogram(img)
            fout = os.path.join(DESCRIPTOR_FOLDER, DESCRIPTOR_SUBFOLDER, filename)
            sio.savemat(fout, {'F': F})
