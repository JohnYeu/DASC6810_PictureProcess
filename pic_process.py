import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os



## load images path
pircture_root = "myPicture"
image_paths = []
for root, dirs, files in os.walk(pircture_root):
    for file in files:
        if file.lower().endswith(('.jpg')): 
            image_paths.append(os.path.join(root, file)) 

print(f"Total images found: {len(image_paths)}")

## check image path 
for path in image_paths:
    if not os.path.exists(path):
        print(f"path not exist: {path}")
        
## process image by batches
# define function for process image by batches
def process_image_by_batch(img_path, size) :  
    for i in range(0, len(img_path), size):
        # get one batch of images
        batch = img_path[i:i + size]   
        
        # used for saving current batch
        data = []
        for path in batch:
                # process each image
                img = cv2.imread(path)  
                if img is not None:
                    img = cv2.resize(img, (128, 128))  
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
                    data.append(img.flatten()) 
        yield np.array(data)
        
        
   
# set picture patch size    
batch_size = 1000

# set number of cluster
cluster_num = 30

# save all data after processing by batches
processed_data = []

# process data
for batch in process_image_by_batch(image_paths, batch_size):
    processed_data.append(batch)
processed_data = np.vstack(processed_data)

# k-mean clusters
kmeans = KMeans(n_clusters = cluster_num, random_state = 0)
cluster_type = kmeans.fit_predict(processed_data)

# reduce dimension to 2D
pca = PCA(n_components=2)
data_2D = pca.fit_transform(processed_data)

# data visualization
plt.figure(figsize=(8, 6))
plt.scatter(data_2D[:, 0], data_2D[:, 1], c = cluster_type, cmap = 'viridis', s = 10)
plt.title("K-Means Clustering of Images")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.colorbar(label = "Cluster")
plt.show()


