#!/usr/bin/env python3

import argparse
import json
from retipy import retina, tortuosity


def tortuosity_window(x1, y1, x2, y2, name, description):
    tw = {}
    tw["name"] = name
    tw["description"] = description
    tw["roi_x"] = [x1, x1, x2, x2]
    tw["roi_y"] = [y1, y2, y2, y1]
    return tw


parser = argparse.ArgumentParser()
parser.add_argument(
    "-a", "--algorithm", default="TD", help="the tortuosity algorithm to apply")
parser.add_argument(
    "-t", "--threshold", type=float, help="threshold to consider a window as tortuous")
parser.add_argument("-i", "--image-path", help="the path to the retinal image to be processed")
parser.add_argument("-w", "--window-size", default=56, type=int, help="the window size")
parser.add_argument(
    "-wcm",
    "--window-creation-method",
    default="combined",
    help="the window creation mode, can be separated or combined")

args = parser.parse_args()

# TODO: this should be able to process from a basic test, a RBG image, right now it will be on segmentation only
image = retina.Retina(None, args.image_path)
image.threshold_image()
image.reshape_by_window(args.window_size)
windows = retina.Window(image, args.window_size, min_pixels=10, method=args.window_creation_method)

evaluation = \
    {
        "uri": args.image_path,
        "data": []
    }

if args.algorithm == "TD":
    for i in range(0, windows.shape[0]):
        window = windows.windows[i, 0]
        w_pos = windows.w_pos[i]
        image = retina.Retina(window, "td")
        image.threshold_image()
        image.apply_thinning()
        vessels = retina.detect_vessel_border(image)
        processed_vessel_count = 0
        for vessel in vessels:
            if len(vessel[0]) > 10:
                processed_vessel_count += 1
                tortuosity_density = tortuosity.tortuosity_density(vessel[0], vessel[1])
                if tortuosity_density > args.threshold:
                    evaluation["data"].append(tortuosity_window(
                        w_pos[0, 0].item(), w_pos[0, 1].item(), w_pos[1, 0].item(), w_pos[1, 1].item(),
                        "High Tortuosity", "Tortuosity Density Value: {}".format(tortuosity_density)))

encoder = json.JSONEncoder()
print(encoder.encode(evaluation))
