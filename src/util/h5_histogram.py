import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("file", help="the h5 file to be opened")
parser.add_argument("-t", "--tag-id", type=int, help="the id of the tortuosity method to validate")
args = parser.parse_args()

file = h5py.File(args.file, 'r')
window = np.array(file['windows'])
print(window.shape)
tags = np.array(file['tags'])
print(tags.shape)

tag = tags[:, args.tag_id]
plt.hist(tag, bins=np.arange(tag.min(), tag.max() + 1, 0.005))
plt.show()
