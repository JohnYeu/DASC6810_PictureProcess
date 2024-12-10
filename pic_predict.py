import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from joblib import load
import os
import csv
import json
from pic_process import get_image_paths_array
from visualization import plt_heat_map
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
    image_paths = get_image_paths_array("pictures-test")
                
    print(f"Total images found: {len(image_paths)}")

    # set picture patch size    
    batch_size = 1000

    predictions = []

    for batch in process_image_by_batch(image_paths, batch_size):
        
        # using k-means to prediction
        batch_predictions = pic_kmeans.predict(batch)
        predictions.extend(batch_predictions)

    # save results (to csv) and print accuarcy
    results_csv = "pediction_results.csv"
    actual_plant_names = []
    predicted_cluster_names = []

    for img_path, cluster_id in zip(image_paths, predictions):
        actual_plant_names.append(os.path.basename(os.path.dirname(img_path)))
        predicted_cluster_names.append(cluster_names.get(str(cluster_id), "Unknown"))
        
    plt_heat_map(predicted_cluster_names, actual_plant_names)

    try:
        with open(results_csv, "w", newline = "", encoding = "utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Acutal Plant Type","Cluster_ID","Predicted Plant Type", "Correct"])
            for index in range(len(actual_plant_names)):
                writer.writerow([actual_plant_names[index], predictions[index], predicted_cluster_names[index], actual_plant_names[index] == predicted_cluster_names[index]])
    except Exception as e:
        print(f"error happened in saving result")

    # Print prediction accuracy
    prediction_correct_count = 0
    for actual_lant, prediction_plant in zip(actual_plant_names, predicted_cluster_names):
        if actual_lant == prediction_plant:
            prediction_correct_count += 1
    print("--------------------------------------------------")
    print(f"Prediction Accuracy: {prediction_correct_count / len(image_paths)}")
    print("--------------------------------------------------")
            
    print(f"Predictions saved to {results_csv} successfully")
