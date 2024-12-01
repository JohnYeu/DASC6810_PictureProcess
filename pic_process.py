import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
from joblib import dump
import json


        
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
        

if __name__ == "__main__":
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
    
    # set picture patch size    
    batch_size = 1000

    # set number of cluster
    cluster_num = 10

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
    cluster_names = {}
    for cluster in range(cluster_num):
        cluster_center = kmeans.cluster_centers_[cluster]
        distances = np.linalg.norm(processed_data - cluster_center, axis = 1)
        closest_image_index = np.argmin(distances[cluster_type == cluster])
        closest_image_path = image_paths[np.where(cluster_type == cluster)[0][closest_image_index]]
        folder_name = os.path.basename(os.path.dirname(closest_image_path))
        cluster_names[cluster] = folder_name

    # check cluster names
    print("Cluster Names:")
    for cluster_id, name in cluster_names.items():
        print(f"Cluster {cluster_id}: {name}")
        
    # save cluster names to json file for pic_predict.py
    with open("cluster_names.json", "w") as f:
        json.dump(cluster_names, f)

    ## data visualization
    # create a graph with size
    plt.figure(figsize=(16, 12))

    # reduce dimension to 2D
    pca = PCA(n_components=2)
    data_2D = pca.fit_transform(processed_data)

    # draw each cluster
    cluster_id = np.unique(cluster_type)
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


