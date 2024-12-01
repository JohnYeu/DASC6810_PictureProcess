import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
from joblib import dump

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

# Define name of clusters
cluster_names = {
    0: "aloevera",
    1: "banana",
    2: "bilimbi",
    3: "cantaloupe",
    4: "cassava",
    5: "coconut",
    6: "corn",
    7: "cucumber",
    8: "curcuma",
    9: "eggplant",
    10: "galangal",
    11: "ginger",
    12: "guava",
    13: "kale",
    14: "longbeans",
    15: "mango",
    16: "melon",
    17: "orange",
    18: "paddy",
    19: "papaya",
    20: "peperchili",
    21: "pineapple",
    22: "pomelo",
    23: "shallot",
    24: "soybeans",
    25: "spinach",
    26: "sweetpotatoes",
    27: "tobacco",
    28: "waterapple",
    29: "watermelon"
}

cluster_id = np.unique(cluster_type)

## data visualization
# create a graph with size
plt.figure(figsize=(16, 12))

# reduce dimension to 2D
pca = PCA(n_components=2)
data_2D = pca.fit_transform(processed_data)

# draw each cluster
for cluster in cluster_id:
    cluster_points = data_2D[cluster_type == cluster]
    cluster_label = cluster_names.get(cluster, f"Cluster {cluster}")
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label = cluster_label, s = 5)

# draw graph
plt.title("K-Means Clustering of Images")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend(title="Cluster Names", loc="upper right")
plt.show()

## save model for prediction
dump(pca, "bio_image_pcaModel.joblib")
dump(kmeans, "bio_image_kmeansModel.joblib")


