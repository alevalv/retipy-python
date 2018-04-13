import glob
import os
from retipy import configuration, retina, tortuosity

directory = "./resources/retina_images/";
contador = 0
for filename in sorted(glob.glob(os.path.join(directory, '*.tif'))):
    name = filename.split("\\")
    name = name[-1].split("_")[0]
    image = retina.Retina("./resources/manual/" + name + "_manual1.gif","./resources/mask/" + name + "_training_mask.gif")
    # Equalizing image
    image.equalize_histogram()
    #
    image.opening(3)
    #
    image.shadow_correction()
    #
    image.homogenize()
