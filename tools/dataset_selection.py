# 将3D车道线的标签在图像上可视化, 实现交互操作完成合适的数据集挑选，文件名存储在lane_depth中
import os
import sys
sys.path.append(os.getcwd())
import json

import cv2
import numpy as np
from utils.misc import draw_lanes, COLOR, get_2D_points_from_3D
from depth_lane_label_sampling import interpolate_for_sampling
from utils.colorize import colorize
from PIL import Image


# 删除标签中第i条车道线
def remove_lanes(lanes, lanes_num_list):
    for i in lanes_num_list:
        lanes[i] = "none"

    return [lane for lane in lanes if lane != "none"]


BASE_PATH = r"E:\dataset\ONCE-3dlanes"
DATA_PATH = BASE_PATH + r"\data" 
LABEL_PATH = BASE_PATH + r"\ONCE_3DLanes"

with open(LABEL_PATH + r"\list\train.txt", "r") as f:
    train_image_list = f.readlines()

# train_image_list = train_image_list[16415:]
train_image_list = train_image_list[1:]
for i in range(len(train_image_list)):
    image_name = train_image_list[i]
    image_name = image_name.strip()
    image_path = DATA_PATH + image_name
    label_path = LABEL_PATH + r"\train" + image_name.replace(".jpg", ".json")
    camera_info_path = DATA_PATH + "\\" + image_name[1:7] + "\\" + image_name[1:7] + ".json"

    print(image_name)
    print(label_path, i)

    image = cv2.imread(image_path)
    label = json.load(open(label_path, "r"))
    camera_info = json.load(open(camera_info_path, "r"))
    calibration = np.array(label["calibration"])
    camera_matrix = np.array(camera_info['calib']['cam01']['cam_intrinsic'], dtype=np.float32)
    distortion_coeffs = np.array(camera_info['calib']['cam01']['distortion'], dtype=np.float32)
    image = cv2.undistort(image, camera_matrix, distortion_coeffs)
    # 去除畸变

    lanes = label["lanes"]
    print(f"车道线条数为: {len(lanes)}")
    if len(lanes) <= 2 or len(lanes) > 6:
        continue
    lanes_depth = np.zeros(image.shape[:2], dtype=np.float32)
    image = draw_lanes(image, lanes, calibration, interpolate=True)

    # lanes = [interpolate_for_sampling(lane) for lane in lanes]
    # depths = [np.array(lane)[:, 2] for lane in lanes]
    # lanes = [get_2D_points_from_3D(lane, calibration) for lane in lanes]
    # for i in range(len(depths)):
    #     for j in range(len(depths[i])):
    #         cv2.circle(lanes_depth, (int(lanes[i][0][j]), int(lanes[i][1][j])), 5, int(depths[i][j]), -1)
    # depth = colorize(lanes_depth, invalid_val=0, cmap="magma")
    # depth = Image.fromarray(depth)
    # depth.show()
    # exit(0)
    cv2.imshow("image", cv2.resize(image, None, fx=0.5, fy=0.5))
    cv2.waitKey(0)
    continue # 是否只看数据集
    
    # 删除不符合的车道线
    num_list = input("请输入需要删除第几条车道线：")
    num_list = list(map(lambda x: int(x), num_list.split(" ") if num_list != " " else []))
    if num_list != []:
        lanes = remove_lanes(lanes, num_list)

        # 显示修改之后的车道线
        image = draw_lanes(image, lanes, calibration, color_remove=(255, 255, 255))
        cv2.imshow("image", cv2.resize(image, None, fx=0.5, fy=0.5))
        cv2.waitKey(0)

    # 判断是否需要保存
    save = input("是否需要保存：")
    if save == "y" or save == "Y":
        with open(label_path, "w") as f:
            label["lanes"] = lanes
            f.writelines (json.dumps(label))

        with open("E:\dataset\ONCE-3dlanes\lane_depth_val.txt", "r+") as f:
            label_list = f.readlines()
            print("已保存{}张图片".format(len(label_list)))
            del(label_list)

        with open("E:\dataset\ONCE-3dlanes\lane_depth_val.txt", "a+") as f:
            f.writelines(image_name + "\n")
    
    # for i in range(len(depth)):
    #         for j in range(len(depth[i])):
    #             cv2.circle(depth_image, (int(lanes[i][j][0]), int(lanes[i][j][1])), 5, int(depth[i][j]), -1)
    # depth = colorize(lanes_depth, invalid_val=0, cmap="magma")
    # depth = Image.fromarray(depth)
    # depth.show()