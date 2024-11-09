import os
import numpy as np
import scipy.io as sio
import cv2
import concurrent.futures
from random import randint
from cvpr_compare import cvpr_compare
from sklearn.metrics import precision_recall_curve, confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc
import math

DESCRIPTOR_FOLDER = 'descriptors'
DESCRIPTOR_SUBFOLDER = 'globalRGBhisto'
IMAGE_FOLDER = 'MSRC_ObjCategImageDatabase_v2'
CODEBOOK_SIZE = 50

# Bag of Visual Words Extraction Function
def extract_sift_descriptors(img):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return descriptors

# Create a codebook using k-means clustering
all_descriptors = []
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = []
    for filename in os.listdir(os.path.join(IMAGE_FOLDER, 'Images')):
        if filename.endswith('.bmp'):
            img_path = os.path.join(IMAGE_FOLDER, 'Images', filename)
            futures.append(executor.submit(cv2.imread, img_path, cv2.IMREAD_GRAYSCALE))
    
    for future in concurrent.futures.as_completed(futures):
        img = future.result()
        if img is not None:
            descriptors = extract_sift_descriptors(img)
            if descriptors is not None:
                all_descriptors.extend(descriptors)

all_descriptors = np.array(all_descriptors)
print(f"Clustering {len(all_descriptors)} descriptors into {CODEBOOK_SIZE} visual words...")

# Use multi-threading to speed up k-means clustering
def run_kmeans(descriptors, n_clusters, random_state):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(descriptors)
    return kmeans

with concurrent.futures.ThreadPoolExecutor() as executor:
    future_kmeans = executor.submit(run_kmeans, all_descriptors, CODEBOOK_SIZE, 42)
    kmeans = future_kmeans.result()
    codebook = kmeans.cluster_centers_

# Extract BoVW histograms for each image
def extract_bovw_histogram(img, codebook, kmeans):
    descriptors = extract_sift_descriptors(img)
    if descriptors is None:
        return np.zeros(len(codebook))
    
    labels = kmeans.predict(descriptors)
    hist, _ = np.histogram(labels, bins=np.arange(len(codebook) + 1))
    hist = hist.astype('float32')
    hist /= (hist.sum() + 1e-5)  # Normalize to sum to 1
    return hist

# Load all BoVW descriptors
ALLFEAT_BOVW = []
ALLFILES = []
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = []
    for filename in os.listdir(os.path.join(IMAGE_FOLDER, 'Images')):
        if filename.endswith('.bmp'):
            img_path = os.path.join(IMAGE_FOLDER, 'Images', filename)
            futures.append(executor.submit(cv2.imread, img_path, cv2.IMREAD_GRAYSCALE))
    
    for future, filename in zip(concurrent.futures.as_completed(futures), os.listdir(os.path.join(IMAGE_FOLDER, 'Images'))):
        img = future.result()
        if img is not None:
            bovw_hist = extract_bovw_histogram(img, codebook, kmeans)
            ALLFILES.append(os.path.join(IMAGE_FOLDER, 'Images', filename))
            ALLFEAT_BOVW.append(bovw_hist)

ALLFEAT_BOVW = np.array(ALLFEAT_BOVW)

# Pick a random image as the query
NIMG = ALLFEAT_BOVW.shape[0]
queryimg = randint(0, NIMG - 1)
query = ALLFEAT_BOVW[queryimg]

# Compute the distance between the query and all other BoVW descriptors
dst_bovw = []
seen_indices = set([queryimg])
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = []
    for i in range(NIMG):
        if i not in seen_indices:
            futures.append(executor.submit(np.linalg.norm, query - ALLFEAT_BOVW[i]))
    
    for future, i in zip(concurrent.futures.as_completed(futures), range(NIMG)):
        if i not in seen_indices:
            distance = future.result()
            dst_bovw.append((distance, i))

# Sort the distances
dst_bovw.sort(key=lambda x: x[0])

# Show the top 5 results using BoVW in a dynamic grid format
SHOW = min(5, NIMG)
print(f"Top {SHOW} results using Bag of Visual Words (BoVW):")
shown_indices = seen_indices.copy()

grid_size = math.ceil(math.sqrt(SHOW))
cell_width, cell_height = 200, 200
result_window = np.zeros((cell_height * grid_size, cell_width * grid_size, 3), dtype=np.uint8)

for i in range(SHOW):
    img_idx = dst_bovw[i][1]
    if img_idx not in shown_indices:
        img = cv2.imread(ALLFILES[img_idx])
        if img is not None:
            img = cv2.resize(img, (cell_width, cell_height))
            row, col = divmod(i, grid_size)
            result_window[row * cell_height:(row + 1) * cell_height, col * cell_width:(col + 1) * cell_width] = img
            cv2.putText(result_window, f'Result {i + 1}', (col * cell_width + 10, row * cell_height + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        shown_indices.add(img_idx)

cv2.imshow('BoVW Results', result_window)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Precision-Recall Evaluation for BoVW
scaler = MinMaxScaler()
predicted_scores = np.array([x[0] for x in dst_bovw]).reshape(-1, 1)
predicted_scores = scaler.fit_transform(predicted_scores).flatten()  # Normalize distances to [0, 1]

# Reverse scores so that higher means more relevant
predicted_scores = 1 - predicted_scores

# Randomly select 10% as relevant to simulate a balanced scenario
np.random.seed(42)  # For reproducibility
relevant_indices = np.random.choice(range(len(dst_bovw)), size=len(dst_bovw) // 10, replace=False)
relevant_images = [1 if i in relevant_indices else 0 for i in range(len(dst_bovw))]

precision, recall, _ = precision_recall_curve(relevant_images, predicted_scores)

# Plot Precision-Recall curve
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (BoVW)')
plt.show()

# Confusion Matrix Evaluation for BoVW
# Use an adaptive threshold based on the median score
threshold = np.median(predicted_scores)
predicted_labels = [1 if score >= threshold else 0 for score in predicted_scores]

# Compute the confusion matrix
cm = confusion_matrix(relevant_images, predicted_labels)

# Plot the confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (BoVW)')
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
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = []
    for filename in os.listdir(os.path.join(DESCRIPTOR_FOLDER, DESCRIPTOR_SUBFOLDER)):
        if filename.endswith('.mat'):
            img_path = os.path.join(DESCRIPTOR_FOLDER, DESCRIPTOR_SUBFOLDER, filename)
            img_actual_path = os.path.join(IMAGE_FOLDER, 'Images', filename.replace(".mat", ".bmp"))
            futures.append(executor.submit(cv2.imread, img_actual_path))
        
    for future, filename in zip(concurrent.futures.as_completed(futures), os.listdir(os.path.join(DESCRIPTOR_FOLDER, DESCRIPTOR_SUBFOLDER))):
        img = future.result()
        if img is not None:
            F = extract_spatial_grid_histogram(img)
            fout = os.path.join(DESCRIPTOR_FOLDER, DESCRIPTOR_SUBFOLDER, filename)
            sio.savemat(fout, {'F': F})
