# 对3D车道线的标签在2D图像上采样, 并且生成消失点标签保存
import os
import sys

sys.path.append(os.getcwd())
import json

import cv2
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from tqdm import tqdm
from utils.misc import draw_lanes, get_2D_points_from_3D
from tools.vanishing_point import *


def interpolate_for_sampling(lane, smoothing=True):
    """
    @params
    - lane: np.array([N x 3]), 表示一条车道线
    - smoothing: 是否需要对车道线进行平滑处理
    @return
    - gt_lane_points_camera: 插值好的车道线，最近点为图像最下方
    """
    lane = np.array(lane)
    x = lane[:, 0]
    y = lane[:, 1]
    z = lane[:, 2]
    z_min, z_max = min(z), max(z)
    zmin, zmax = np.ceil(z_min * 1000) / 1000, np.floor(z_max * 1000) / 1000

    gt_lane_points_camera = []
    # 第一次插值，使用标签中给好的点
    if len(lane) < 4:
        f_zx = interp1d(z, x, kind='linear', fill_value='extrapolate')
        f_zy = interp1d(z, y, kind='linear', fill_value='extrapolate')
        for z in np.arange(zmin, zmax, np.around((zmax - zmin) / 6, 3)):
            z = np.around(z, 3)
            x_out = np.around(f_zx(z), 3)
            y_out = np.around(f_zy(z), 3)
            gt_lane_points_camera.append([x_out, y_out, z])
    else:
        f_zx = interp1d(z, x, kind='cubic', fill_value='extrapolate')
        f_zy = interp1d(z, y, kind='cubic', fill_value='extrapolate')
        for z in np.arange(zmin, zmin + np.around((zmax - zmin) / 2, 1), 0.01):
            z = np.around(z, 3)
            x_out = np.around(f_zx(z), 3)
            y_out = np.around(f_zy(z), 3)
            gt_lane_points_camera.append([x_out, y_out, z])
        for z in np.arange(zmin + np.around((zmax - zmin) / 2, 1), zmax, 0.1):
            z = np.around(z, 3)
            x_out = np.around(f_zx(z), 3)
            y_out = np.around(f_zy(z), 3)

            gt_lane_points_camera.append([x_out, y_out, z])
    
    # 第二次插值，补全下半部分的车道线
    n = int(len(gt_lane_points_camera) * 0.3)
    last_n_points = np.array(gt_lane_points_camera[:n])
    last_n_x = last_n_points[:, 0]
    last_n_y = last_n_points[:, 1]
    last_n_z = last_n_points[:, 2]
    f_zx = np.poly1d(np.polyfit(last_n_z, last_n_x, 1))
    f_zy = np.poly1d(np.polyfit(last_n_z, last_n_y, 1))
    z_min, z_max = 0.01, last_n_points[0][2]
    extended_lane_points_camera = []
    for z in np.arange(z_min, z_max, 0.01):
        z = np.around(z, 3)
        x_out = np.around(f_zx(z), 3)
        y_out = np.around(f_zy(z), 3)
        extended_lane_points_camera.append([x_out, y_out, z])
    gt_lane_points_camera = np.vstack([extended_lane_points_camera, gt_lane_points_camera])

    # 平滑
    if smoothing:
        gt_lane_points_camera[:, 0] = savgol_filter(gt_lane_points_camera[:, 0], 20, 3)
        gt_lane_points_camera[:, 1] = savgol_filter(gt_lane_points_camera[:, 1], 20, 3)
        gt_lane_points_camera[:, 2] = savgol_filter(gt_lane_points_camera[:, 2], 20, 3)
    
    return gt_lane_points_camera

def sampling(range_param, lane, calibration):
    """
    对3D车道线的标签进行采样, 获取2D点位置和深度
    @param
    - range_param: (H, W, delta)
    - lane: np.array([N, 3]) 车道线
    @return
    - sampling_points: [2 x N] [(x1, x2, ...), (y1, y2, ...)]
    - sampling_depth: [N] N的点的深度
    """
    H, W, delta = range_param
    cy, fy = calibration[1, 2], calibration[1, 1]
    y_divided_by_z = lane[:, 1] / lane[:, 2]
    startpoint_y = (fy * lane[-1][1] + cy * lane[-1][2]) / lane[-1][2]
    startpoint_y = int(np.ceil(startpoint_y / 10) * 10) - 1
    sampling_index = []
    
    for v in range(startpoint_y, H + delta, delta):
        temp = np.abs(y_divided_by_z - (v - cy) / fy)
        sampling_index.append(np.argmin(temp))

    sampling_depth = lane[:, 2][sampling_index]
    sampling_points = get_2D_points_from_3D(lane[sampling_index], calibration)
    sampling_points[1, :] = range(startpoint_y, H + delta, delta)

    # 过滤掉超出画面范围之内的点
    sampling_points_x = sampling_points[0]
    index = len(sampling_points_x + 1)
    if np.size(np.where(sampling_points_x < 0)) != 0:
        index = int(np.where(sampling_points_x < 0)[0][0])
    if np.size(np.where(sampling_points_x > W)) != 0:
        index = int(np.where(sampling_points_x > W)[0][0])
    sampling_points = sampling_points[:, :index - 1]
    sampling_depth = sampling_depth[:index - 1]
    
    return sampling_points, sampling_depth

def fill_lane_list(lane, fill=True):
    """
    @params
    lane: np.array[2, N] or [N, 2]
    @return
    filled_lane: 没有点的部分用-2填充
    """
    if np.shape(lane)[0] == 2:
        lane = np.array(list(zip(lane[0], lane[1])))
    if not fill:
        return lane

    get_index = lambda x:int(np.ceil((x - START) / 10))
    START = 530 - 1
    POINT_NUM = lane.shape[0]
    filled_lane = np.zeros((50)) - 2
    for i in range(POINT_NUM):
        index = get_index(lane[i][1])
        if index >= 0 or index < 50:
            filled_lane[get_index(lane[i][1])] = round(lane[i][0], 3)
    return filled_lane

def fill_depth(depth, filled_lane):
    filled_depth = np.zeros_like(filled_lane) - 2
    filled_depth[filled_lane > 0] = depth
    return filled_depth.tolist()

if __name__ == "__main__":
    BASE_PATH = r"E:\dataset\ONCE-3dlanes"
    DATA_PATH = BASE_PATH + r"\data" 
    LABEL_PATH = BASE_PATH + r"\ONCE_3DLanes"

    with open(BASE_PATH + r"\lane_depth.txt", "r") as f:
        train_image_list = f.readlines()
    
    train_image_list = train_image_list[1:]
    for i in tqdm(range(len(train_image_list))):
        image_name = train_image_list[i]
        image_name = image_name.strip()
        image_path = DATA_PATH + image_name
        label_path = LABEL_PATH + r"\train" + image_name.replace(".jpg", ".json")

        # print(image_name)
        # print(label_path, i)
        
        image = cv2.imread(image_path)
        label = json.load(open(label_path, "r"))

        lanes = label["lanes"]
        calibration = np.array(label["calibration"])
        lanes_depth = np.zeros(image.shape[:2], dtype=np.float32)
        
        lanes_interpolate = [interpolate_for_sampling(lane) for lane in lanes]
        lanes_filled = []
        lanes_without_filled = []
        lanes_depth = []
        for lane in lanes_interpolate:
            lane = np.array(lane)
            sampling_points, sampling_depth = sampling((1020, 1920, 10), lane, calibration[:, 0:3])
            for j in range(len(sampling_points[0])):
                cv2.circle(image, (round(sampling_points[0][j]), round(sampling_points[1][j])), 5, (255, 0, 0), -1)
            # print(fill_lane_list(sampling_points))
            lanes_filled.append(fill_lane_list(sampling_points))
            lanes_without_filled.append(fill_lane_list(sampling_points, False))
            lanes_depth.append(fill_depth(sampling_depth, fill_lane_list(sampling_points)))
        #  查看深度之间的关系
        #     import matplotlib.pyplot as plt
        #     plt.plot(range(len(sampling_depth)), sampling_depth)
        # plt.show()
        # cv2.imshow("image", cv2.resize(image, None, fx=0.5, fy=0.5))
        # cv2.waitKey(0)
        # exit(0)

        # #------------------------获取消失点标签------------------------#
        lanes_filled = np.array(lanes_filled)
        vp = VanishingPoint.from_lane_get_all_vp(lanes_filled, np.arange(530, 1030, 10))
        _, _, pos = VanishingPoint.get_horizon(np.array(vp))
        vanishing_points, _ = VanishingPoint.get_vp_label(lanes_filled, np.arange(530, 1030, 10), pos, W=1920)
        vanishing_points = VanishingPoint.smoothe_vp(vanishing_points, [960, 0])

        lane_depth_label = {
            "image_name": image_name,
            "lanes": lanes_filled.tolist(),
            "depth": lanes_depth,
            "vp": list(vanishing_points),
            "calibration": calibration.tolist()
        }
        # with open('depth_test.json', 'a', encoding='utf-8') as f:
        #     f.writelines(json.dumps(lane_depth_label) + '\n')

        # #------------------------可视化消失点标签重建结果------------------------#
        vanishing_points = [[vp_x, vanishing_points[0]] for vp_x in vanishing_points[1:]]
        start_points = [lane[-1] for lane in lanes_without_filled]
        end_points = [lane[0] for lane in lanes_without_filled]
        # print(start_points, end_points)

        cv2.line(image, [0, pos], [1919, pos], (51, 231, 19), 5)
        # cv2.imshow("image", cv2.resize(image, None, fx=0.5, fy=0.5))
        # cv2.waitKey(0)

        for i in range(len(start_points)):
            start_index = (1019 - start_points[i][1]) // 10
            end_index = (1019 - end_points[i][1]) // 10
            vp = vanishing_points[int(start_index):int(end_index)]
            lane = VanishingPoint.reconstruct_lane(vp, start_points[i], start=529, points_num=50)
            for lp in lane:
                cv2.circle(image, (int(lp[0]), int(lp[1])), 5, (255, 255, 0), -1)
        cv2.imshow("image", cv2.resize(image, None, fx=0.5, fy=0.5))
        cv2.waitKey(0)