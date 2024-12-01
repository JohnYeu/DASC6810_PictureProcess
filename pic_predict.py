import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from joblib import load
import os
import csv
import json

from pic_process import process_image_by_batch

## load models
def load_models():
    pca = load("bio_image_pcaModel.joblib") 
    kmeans = load("bio_image_kmeansModel.joblib")
    return pca, kmeans


if __name__ == "__main__":
    pic_pca, pic_kmeans = load_models()
    with open("cluster_names.json", "r") as f:
        cluster_names = json.load(f)
    
    ## load images path
    pircture_root = "testPicture"
    image_paths = []
    for root, dirs, files in os.walk(pircture_root):
        for file in files:
            if file.lower().endswith(('.jpg')): 
                image_paths.append(os.path.join(root, file)) 
                
    print(f"Total images found: {len(image_paths)}")
    
    # set picture patch size    
    batch_size = 1000
    
    predictions = []
    
    for batch in process_image_by_batch(image_paths, batch_size):
        
        # using k-means to prediction
        batch_predictions = pic_kmeans.predict(batch)
        predictions.extend(batch_predictions)
        

    ## save results (to csv)
    results_csv = "pediction_results.csv"
    try:
        with open(results_csv, "w", newline = "", encoding = "utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Image_Path","Cluster_ID","Cluster_Name"])
            for img_path, cluster_id in zip(image_paths, predictions):
                cluster_name = cluster_names.get(str(cluster_id), "Unknown")
                writer.writerow([img_path, cluster_id, cluster_name])
    except Exception as e:
        print(f"error happened in saving result")
            
    print(f"Predictions saved to {results_csv} successfully")
    