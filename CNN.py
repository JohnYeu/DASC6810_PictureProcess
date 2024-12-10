import os
import numpy as np
from keras.models import Model
from keras.applications.vgg16 import VGG16
from pic_process import get_image_paths_array
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.cluster import KMeans
from keras.layers import Input
from pic_process import get_image_paths_array, define_cluster_names, plot_cluster_scatter
from visualization import plt_heat_map

# Model output would be (7, 7, 512) by defualt
def create_model_vgg16():
    input_shape = Input(shape = (224, 224, 3))
    model = VGG16(input_tensor = input_shape, include_top = False)
    # output_pooled = AveragePooling2D((7, 7))(model.output)
    return Model(model.input, model.output)

# Resize image to (224, 224, 3)
def images_array_generator(image_paths, size):
    for index in range(0, len(image_paths), size):
        batch = []
        current_image_paths = image_paths[index : index + size]
        for image_path in current_image_paths:
            img_raw = image.load_img(image_path, target_size = (224, 224))
            img_array = image.img_to_array(img_raw)
            batch.append(img_array)
        batch_array = np.array(batch)
        yield preprocess_input(batch_array)

def extract_features(image_paths, batch_size):
    all_train_features = []

    for batch in images_array_generator(image_paths, batch_size):
        batch_features = vgg16_model.predict(batch)
        batch_features_flattened = batch_features.reshape(batch_features.shape[0], -1)
        all_train_features.append(batch_features_flattened)

    return np.vstack(all_train_features)


if __name__ == "__main__":
    train_image_paths = get_image_paths_array("pictures-train")
    test_image_paths = get_image_paths_array("pictures-test")
    vgg16_model = create_model_vgg16()
    batch_size = 600
    cluster_num = 7

    all_train_features = extract_features(train_image_paths, batch_size)
    all_test_features = extract_features(test_image_paths, batch_size)

    kmeans = KMeans(n_clusters = cluster_num, random_state = 0)
    cluster_type = kmeans.fit_predict(all_train_features)
    cluster_names = define_cluster_names(cluster_type, train_image_paths, cluster_num)

    predictions = kmeans.predict(all_test_features)

    actual_plant_names = []
    predicted_cluster_names = []

    for img_path, prediction_cluster in zip(test_image_paths, predictions):
        actual_plant_names.append(os.path.basename(os.path.dirname(img_path)))
        predicted_cluster_names.append(cluster_names[prediction_cluster])
    
    plot_cluster_scatter(all_train_features, cluster_type, cluster_names)
    plt_heat_map(actual_plant_names, predicted_cluster_names)

