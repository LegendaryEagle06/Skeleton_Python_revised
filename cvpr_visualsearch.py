import os
import numpy as np
import scipy.io as sio
import cv2
from random import randint
from cvpr_compare import cvpr_compare
from sklearn.metrics import precision_recall_curve
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

DESCRIPTOR_FOLDER = 'descriptors'
DESCRIPTOR_SUBFOLDER = 'globalRGBhisto'
IMAGE_FOLDER = 'MSRC_ObjCategImageDatabase_v2'

# Load all descriptors
ALLFEAT = []
ALLFILES = []
for filename in os.listdir(os.path.join(DESCRIPTOR_FOLDER, DESCRIPTOR_SUBFOLDER)):
    if filename.endswith('.mat'):
        img_path = os.path.join(DESCRIPTOR_FOLDER, DESCRIPTOR_SUBFOLDER, filename)
        img_actual_path = os.path.join(IMAGE_FOLDER, 'Images', filename).replace(".mat", ".bmp")
        img_data = sio.loadmat(img_path)
        ALLFILES.append(img_actual_path)
        ALLFEAT.append(img_data['F'][0])  # Assuming F is a 1D array

# Convert ALLFEAT to a numpy array
ALLFEAT = np.array(ALLFEAT)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=15)
ALLFEAT = pca.fit_transform(ALLFEAT)

# Pick a random image as the query
NIMG = ALLFEAT.shape[0]
queryimg = randint(0, NIMG - 1)

# Compute the distance between the query and all other descriptors
dst = []
query = ALLFEAT[queryimg]
for i in range(NIMG):
    candidate = ALLFEAT[i]
    distance = cvpr_compare(query, candidate)
    dst.append((distance, i))

# Sort the distances
dst.sort(key=lambda x: x[0])

# Show the top 15 results
SHOW = 15
for i in range(SHOW):
    img = cv2.imread(ALLFILES[dst[i][1]])
    img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))  # Make image quarter size
    cv2.imshow(f"Result {i+1}", img)
    cv2.waitKey(0)
cv2.destroyAllWindows()

# Precision-Recall Evaluation
relevant_images = [1 if i < 10 else 0 for i in range(NIMG)]  # Assuming top 10 are relevant
predicted_scores = [1 / (x[0] + 1e-5) for x in dst]  # Inverse of distance as relevance score

precision, recall, _ = precision_recall_curve(relevant_images, predicted_scores)

# Plot Precision-Recall curve
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()
