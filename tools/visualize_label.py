import os
import sys

sys.path.append(os.getcwd())
import json
import cv2
import numpy as np
from PIL import Image
from utils.colorize import colorize
from utils.misc import interpolate_for_2d
from scipy.interpolate import interp1d


BASE_PATH = r"E:\dataset\ONCE-3dlanes\data"
GET_Y = lambda x: x*10 +530

# def interpolate_z(depth, lanes):
#     depth, lanes = np.array(depth), np.array(lanes)
#     depth_interpolation = []

#     for i in range(len(depth)):
#         valid_index = np.where(lanes[i] > 0)[0]
#         start_y, end_y = GET_Y(valid_index[0]), GET_Y(valid_index[-1])
#         depth_interpolation.append(interpolate_for_2d_z(depth[i][valid_index], np.arange(start_y, end_y + 10, 10)))

#     return np.array(depth_interpolation)

# def interpolate_x(lanes):
#     lanes = np.array(lanes)
#     lanes_interpolate = []

#     for i in range(len(lanes)):
#         valid_index = np.where(lanes[i] > 0)[0]
#         start_y, end_y = GET_Y(valid_index[0]), GET_Y(valid_index[-1])
#         lane = lanes[i][valid_index]
#         f_lane = interp1d(np.arange(start_y, end_y + 10, 10), lane)

#         lane_interpolate = []
#         for y in range(start_y, end_y + 1):
#             lane_interpolate.append([int(f_lane(y)), y])
#         lanes_interpolate.append(lane_interpolate)
#     return np.array(lanes_interpolate)

        
with open(r"depth_label/depth_test.json", "r", encoding="utf-8") as f:
    json_list = f.readlines()

for label in json_list:
    label = json.loads(label)
    image_name = label["image_name"]
    depth = label["depth"]
    lanes = label["lanes"]
    image = cv2.imread(BASE_PATH + image_name)

    depth, lanes = interpolate_for_2d(depth, lanes)
    depth_image = np.zeros(shape = image.shape[:2])
    for i in range(len(depth)):
        for j in range(len(depth[i])):
            cv2.circle(depth_image, (int(lanes[i][j][0]), int(lanes[i][j][1])), 5, int(depth[i][j]), -1)

    depth = colorize(depth_image, invalid_val=0, cmap="magma")
    depth = Image.fromarray(depth)
    depth.show()
    exit(0)
    # cv2.imshow("depth_image", cv2.resize(depth_image, None, fx=0.5, fy=0.5))
    # cv2.waitKey(0)
