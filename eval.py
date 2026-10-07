import json

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from torchvision import transforms
from tqdm import tqdm

from model.model import Depth
from utils.colorize import colorize
from utils.lane_eval import Eval
from utils.misc import fill_lane, interpolate_for_2d, COLOR, map_2D_to_3D
from tools.vanishing_point import VanishingPoint
from utils.plt_utils import draw_lanes3d

BASE_PATH = r"E:\dataset\ONCE-3dlanes\data"
LABEL_PATH = r"E:\dataset\ONCE-3dlanes\ONCE_3DLanes\val"

def read_label_from_json(json_label):
    data = json.loads(json_label)
    image_name = data["image_name"]
    image = cv2.imread(BASE_PATH + image_name)
    lanes = np.array(data["lanes"])
    depth = np.array(data["depth"])
    calibration = np.array(data["calibration"])[0:3, 0:3]

    # load 3d label
    label_3d = json.loads(open((LABEL_PATH + image_name).replace("jpg", "json"), "r").readline())
    lanes_3d_label = label_3d["lanes"]

    # return image, lanes, depth, calibration, lanes_3d_label
    return image, lanes, depth, calibration

def preprocess_image(image: np.array) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize([510, 960]), 
        # transforms.Resize([340, 640]),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])
    
    image_copy = image.copy()
    image = Image.fromarray(image[:, :, ::-1])
    image = transform(image)
    image = torch.unsqueeze(image, 0).cuda()

    return image, image_copy

@torch.no_grad()
def forward(model, image):
    points, vps, depth = model(image)
    
    # map points to original image size
    points = F.softmax(points[0], dim = 1).cpu().numpy()
    exist = points[:, 0]
    x = np.argmax(points[:, 1:961], axis=1)
    y = np.argmax(points[:, 961:1471], axis=1)
    end = np.argmax(points[:, 1471:], axis=1)
    x_map = lambda x: (x + 1) * 2 - 1
    y_map = lambda y: (y + 1) * 2 - 1
    x = [x_map(x_) for x_ in x]
    y = [y_map(y_) for y_ in y]
    end = np.array([y_map(end_) for end_ in end])
    point = np.array(list(zip(x, y))).astype(int)

    # vp
    vp_y = int(vps[0][0].sigmoid().item() * 1020)
    decode_vpx = lambda x: int(x.item() * 960 + 960)
    vps = np.array([[decode_vpx(vp), vp_y] for vp in vps[0][1:]])

    # depth
    depth = depth[0].cpu().numpy() * 50

    return exist, point, end, vps, depth

def matcher(points, point_labels):
    distance = cdist(points, point_labels)
    return linear_sum_assignment(distance)


if __name__ == "__main__":
    with open(r"depth_label\depth_test.json", "r", encoding="utf-8") as f:
        json_list = f.readlines()
    
    # 计算各种指标的总和以便求平均
    TP = 0
    GT_NUM = 0
    PRED_NUM = 0
    DISTANCE_ERROR = 0
    THRESHOLD = 0.01

    for json_label in tqdm(json_list):
        image, lanes_label, depth_label, calibration = read_label_from_json(json_label)
        image, image_copy = preprocess_image(image)

        # load model
        # model = Depth().cuda()
        model = Depth(H=510, W=960, k=1).cuda()
        model.load_state_dict(torch.load(r"log\checkpoint_ema.pth", map_location="cuda"))
        exist, point, end, vps, depth = forward(model, image)
        point, end, depth = point[exist < THRESHOLD], end[exist < THRESHOLD], depth[exist < THRESHOLD]

        # match lanes
        point_labels = np.array([fill_lane(lane, True)[-1] for lane in lanes_label])
        indices = matcher(point, point_labels)

        # visualize
        cv2.line(image_copy, (0, vps[0][1]), (1919, vps[0][1]), COLOR[4], 5)
        for vp in vps:
            cv2.circle(image_copy, vp.astype(int), 5, COLOR[3], -1)
        for p in point:
            cv2.circle(image_copy, p.astype(int), 5, COLOR[2], -1)
        
        # reconstruct lanes
        lanes = []
        for i in range(len(point)):
            start_index = (1019 - point[i][1]) // 10
            end_index = (1019 - end[i]) // 10
            vp = vps[int(start_index):int(end_index)]
            lane = VanishingPoint.reconstruct_lane(vp, point[i], start=529, points_num=50)
            lanes.append(np.array(lane))
            for lp in lane:
                cv2.circle(image_copy, (int(lp[0]), int(lp[1])), 5, COLOR[1], -1)
        
        # remove invalid depth points
        lanes = np.array([fill_lane(lane) for lane in lanes])
        for i in range(len(lanes)):
            depth[i][lanes[i] < 0] = -2

        # map 2D points to 3D points
        lanes_3D = []
        for i in range(len(lanes)):
            z = np.array(depth[i])
            lanes_3D.append(map_2D_to_3D(fill_lane(lanes[i], True), z[z > 0], calibration))
        
        lanes_3d_label = []
        for i in range(len(lanes_label)):
            z = np.array(depth_label[i])
            lanes_3d_label.append(map_2D_to_3D(fill_lane(lanes_label[i], True), z[z > 0], calibration))
        
        eval_3d = Eval(lanes_3D, lanes_3d_label)
        tp, distance_error = eval_3d.eval()
        TP += tp
        GT_NUM += len(lanes_label)
        PRED_NUM += len(lanes)
        DISTANCE_ERROR += distance_error

        ## 深度图可视化
        depth, lanes = interpolate_for_2d(depth, lanes)
        depth_image = np.zeros(shape = image_copy.shape[:2])
        for i in range(len(depth)):
            for j in range(len(depth[i])):
                cv2.circle(depth_image, (int(lanes[i][j][0]), int(lanes[i][j][1])), 5, int(depth[i][j]), -1)

        depth = colorize(depth_image, invalid_val=0, cmap="magma")
        depth = Image.fromarray(depth)
        depth.save("depth.png")
        
        cv2.imshow("image", cv2.resize(image_copy, None, fx=0.5, fy=0.5))
        cv2.waitKey(0)

        # 可视化三维车道线
        # draw_lanes3d(lanes_3D, lanes_3d_label)

    DISTANCE_ERROR /= TP
    PRECISION = TP / PRED_NUM
    RECALL = TP / GT_NUM
    F_value = 2 * PRECISION * RECALL / (PRECISION + RECALL)
    print(PRECISION, RECALL, F_value, DISTANCE_ERROR)