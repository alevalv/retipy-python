#!/usr/bin/env python3

import argparse
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help="csv file to read")
parser.add_argument("-c1", help="column 1")
parser.add_argument("-c2", help="column 2")
parser.add_argument("-c3", help="column 3")

args = parser.parse_args()

DATA = np.genfromtxt(args.file, delimiter=',', names=['w', 'ppw', 'ss', 'r2t', 't'])
DATA = DATA[np.lexsort(np.transpose(DATA)[::-1])]

FIG = plt.figure()

ax1 = FIG.add_subplot(111, projection='3d')

ax1.plot(DATA['w'], DATA['r2t'], DATA['t'], label='test')

ax1.set_xlabel('w')
ax1.set_ylabel('r2t')
ax1.set_zlabel('t')

plt.show()
