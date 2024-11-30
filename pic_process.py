import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

# Step 1: load and process data
def preprocess_image(image_path, size=(128, 128)):
    img = cv2.imread(image_path)  # read image
    img = cv2.resize(img, size)  # resize
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # change to RGB
    return img.flatten()  # flatten into a one-dimensional feature

# load image path
categories = ["aloevera", "banana", "watermelon"]
image_paths = []
for category in categories:
    for i in range(1, 100): 
        image_paths.append(f"picture/{category}{i}.jpg")

# check image path 
for path in image_paths:
    if not os.path.exists(path):
        print(f"path not exist: {path}")
        
data = np.array([preprocess_image(path) for path in image_paths])  # Convert images to feature matrix

# Step 2: K-Means clusters
kmeans = KMeans(n_clusters=3, random_state=0)  # assume 3 clusters
labels = kmeans.fit_predict(data)

# Step 3:  Data visualization
pca = PCA(n_components=2)  # Reduce dimensions to 2D
reduced_data = pca.fit_transform(data)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis')
plt.title("K-Means Clustering of Plants")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.colorbar(label='Cluster')
plt.show()

