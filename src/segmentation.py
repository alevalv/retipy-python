import glob
import os
from retipy import retina_grayscale

base_directory = "../../DRIVE/training/"
directory = base_directory+"retina_images/"
contador = 0
for filename in sorted(glob.glob(os.path.join(directory, '*.tif'))):
    name = filename.split("\\")
    name = name[-1].split("_")[0]
    image = retina_grayscale.Retina_grayscale(filename, base_directory+"1st_manual/" + name + "_manual1.gif", base_directory+"mask/" + name + "_training_mask.gif")
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
