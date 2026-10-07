# 获取车道线y轴的取值范围
import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import cv2
import json
from utils.misc import get_2D_points_from_3D

# 最低520.44
MIN_Y = 524.243753540299 # 3000


BASE_PATH = r"E:\dataset\ONCE-3dlanes"
DATA_PATH = BASE_PATH + r"\data" 
LABEL_PATH = BASE_PATH + r"\ONCE_3DLanes"

with open(BASE_PATH + r"\lane_depth.txt", "r") as f:
    train_image_list = f.readlines()

# with open(LABEL_PATH + r"\list\train.txt", "r") as f:
#     train_image_list = f.readlines()

# train_image_list = train_image_list[103534:]
for i in range(len(train_image_list)):
    print(i, MIN_Y)
    image_name = train_image_list[i]
    image_name = image_name.strip()
    image_path = DATA_PATH + image_name
    label_path = LABEL_PATH + r"\train" + image_name.replace(".jpg", ".json")
    
    image = cv2.imread(image_path)
    label = json.load(open(label_path, "r"))

    lanes = label["lanes"]
    calibration = np.array(label["calibration"])
    lanes_depth = np.zeros(image.shape[:2], dtype=np.float32)
    
    lanes = [get_2D_points_from_3D(lane, calibration)[1] for lane in lanes]
    for lane in lanes:
        min_y = np.min(lane[0])
        if min_y < MIN_Y:
            MIN_Y = min_y


print(f"车道线起始点的最小y值为{MIN_Y}")
    