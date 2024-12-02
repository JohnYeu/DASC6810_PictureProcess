import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
from joblib import dump
import json
from pic_process import generate_images_data


if __name__ == "__main__":

    # Get processed_data
    processed_data = generate_images_data()

    # Find K value using elbow method
    cost = []
    k_range = range(1, 20)
    for k in k_range:
        k_mean_model = KMeans(n_clusters = k, random_state = 0)
        k_mean_model.fit(processed_data)
        cost.append(k_mean_model.inertia_)
    
    # Plot cost
    plt.figure(figsize = (10, 10))
    plt.plot(k_range, cost, linestyle = "-")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Cost (Inertia)")
    plt.show()


