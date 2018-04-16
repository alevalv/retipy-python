import glob
import os
import numpy as np
import cv2
from skimage import io
from matplotlib import pyplot as plt
from keras.models import model_from_json
from retipy import retina_grayscale

# Load model of the neuronal network
json_file = open("./resources/neural_networks_models/model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# Load weights
loaded_model.load_weights("./resources/neural_networks_models/model.h5")

base_directory = "../../DRIVE/training/"
directory = base_directory+"retina_images/"
contador = 0
for filename in sorted(glob.glob(os.path.join(directory, '*.tif'))):
    name = filename.split("\\")
    name = name[-1].split("_")[0]
    image = retina_grayscale.Retina_grayscale(filename, base_directory+"1st_manual/" + name + "_manual1.gif", base_directory+"mask/" + name + "_training_mask.gif")
    # Print image
    image.view()
    # Equalizing image
    image.equalize_histogram()
    # first step: remove center reflex of the vessel
    image.opening(3)
    # second step: Correct shadows of the image
    image.shadow_correction()
    # Third step: Homogenize the image
    image.homogenize()
    # Fourth step: Vessel enhancement
    image.vessel_enhancement()
    # Fifth step: Extract the pixels' features
    image.extract_features()
    # Load features
    features = image.fe
    # Normalizing features for the neural network
    for feature in range(0, 12):
        f_std = np.std(features[:, feature])
        f_mean = np.mean(features[:, feature])
        features[:, feature] = (features[:, feature] - f_mean) / f_std
    # Classifying pixels
    for row in range(0, features.shape[0]):
        image.segmented_image[int(features[row, 13]), int(features[row, 14])] = loaded_model.predict(features[row:row + 1, 0:7], batch_size=1)
    # Applying postprocessing to a copy of the segmented image
    segmented_image = image.segmented_image
    # A threshold is set and the values are taken to extremes
    k = 0.979
    mask1 = segmented_image >= k
    mask2 = segmented_image < k
    segmented_image[mask1] = 255
    segmented_image[mask2] = 0
    # Applying connected component algorithm
    connectivity = 4
    connected_components = cv2.connectedComponentsWithStats(segmented_image.astype(np.uint8), connectivity, cv2.CV_32S)
    # Any element that does not have a connectivity area greater than 15 is eliminated
    for x in range(1, connected_components[0]):
        mask = connected_components[1] == x
        sum = int(np.sum(connected_components[1][mask])) / x
        if (sum < 15):
            segmented_image[mask] = 0
    # Printing result
    io.imshow(segmented_image)
    plt.show()