import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np

def plt_heat_map(actual_plant_types, predicted_plant_types):
    actual_array = np.array(actual_plant_types)
    predicted_array = np.array(predicted_plant_types)

    labels = np.unique(actual_array)
    conf_matrix = confusion_matrix(actual_array, predicted_array, labels = labels)
    conf_df = pd.DataFrame(conf_matrix, index = labels, columns = labels)
    accuracy = accuracy_score(actual_array, predicted_array)

    plt.figure(figsize = (10, 10))
    sns.heatmap(conf_df, annot = True, cmap = "Oranges", fmt = "d")
    plt.title(f"Confusion Matrix\n\nPrediction Accuracy: {accuracy: .2%}", fontsize = 15)
    plt.ylabel("Actual Plant Type")
    plt.xlabel("Predicted Plant Type")
    plt.show()