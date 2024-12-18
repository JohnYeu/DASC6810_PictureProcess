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
                    img = cv2.resize(img, (224, 224))  
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
                    data.append(img.flatten()) 
        yield np.array(data)


def get_image_paths_array(folder):

    # Get current directory
    current_dir = os.getcwd()
    pircture_root = os.path.join(current_dir, folder)

    # Get all image paths
    image_paths = []
    for root, dirs, files in os.walk(pircture_root):
        for file in files:
            if file.lower().endswith(('.jpg')): 
                image_paths.append(os.path.join(root, file)) 
    print(f"Total images found: {len(image_paths)}")

    for path in image_paths:
        if not os.path.exists(path):
            print(f"path not exist: {path}")

    return image_paths

# Extract this part as a method, because find_k_value.py needs processed_data
def generate_images_data(image_paths):
    
    # set picture patch size    
    batch_size = 1000

    # save all data after processing by batches
    processed_data = []

    # process data
    for batch in process_image_by_batch(image_paths, batch_size):
        processed_data.append(batch)
    processed_data = np.vstack(processed_data)

    return processed_data

def define_cluster_names(cluster_type, origin_image_paths, cluster_num):

    cluster_names = {}

    for cluster in range(cluster_num):
        indices_for_this_cluster = np.where(cluster_type == cluster)[0]
        images_paths_for_this_cluster = [origin_image_paths[idx] for idx in indices_for_this_cluster]
        plant_name_count_dic = {}
        for image_path in images_paths_for_this_cluster:
            plant_name = os.path.basename(os.path.dirname(image_path))
            if plant_name in plant_name_count_dic:
                plant_name_count_dic[plant_name] += 1
            else:
                plant_name_count_dic[plant_name] = 1
        name_for_this_cluster = max(plant_name_count_dic, key = plant_name_count_dic.get)
        cluster_names[cluster] = name_for_this_cluster
    return cluster_names

def plot_cluster_scatter(data, cluster_type, cluster_name_dict):
    plt.figure(figsize=(16, 12))

    # reduce dimension to 2D
    pca = PCA(n_components=2)
    data_2D = pca.fit_transform(data)

    # draw each cluster
    cluster_id = np.unique(cluster_type)
    for cluster in cluster_id:
        cluster_points = data_2D[cluster_type == cluster]
        cluster_label = cluster_name_dict.get(cluster, f"Cluster {cluster}")
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label = cluster_label, s = 5)

    # draw graph
    plt.title("K-Means Clustering of Images")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend(title="Cluster Names", loc="upper right")
    plt.show()

    return pca


if __name__ == "__main__":

    image_paths = get_image_paths_array("pictures-train")
    processed_data = generate_images_data(image_paths)
    cluster_num = 7

    # k-mean clusters
    kmeans = KMeans(n_clusters = cluster_num, random_state = 0)
    cluster_type = kmeans.fit_predict(processed_data)


    cluster_names = define_cluster_names(cluster_type, image_paths, cluster_num)

  # check cluster names
    print("Cluster Names:")
    for cluster_id, name in cluster_names.items():
        print(f"Cluster {cluster_id}: {name}")
        
    # save cluster names to json file for pic_predict.py
    with open("cluster_names.json", "w") as f:
        json.dump(cluster_names, f)

    ## data visualization
    # create a graph with size
    pca = plot_cluster_scatter(processed_data, cluster_type, cluster_names)
    
    ## save model for prediction
    dump(pca, "bio_image_pcaModel.joblib")
    dump(kmeans, "bio_image_kmeansModel.joblib")

