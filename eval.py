import argparse
import json
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from model.LVPNet import LVPNet
from tools.lane import LaneEval
from tools.vanishing_point import *
from utils.config_utils import *


def get_exist_lanes_label(lanes_label):
    label = []
    for i in range(len(lanes_label)):
        if(list(filter(lambda x: x>0, lanes_label[i])) != []):
            label.append(lanes_label[i])
    
    return label

def preprocess_image(image: np.array, cfg) -> torch.Tensor:
    """
    input: image (np.array[H, W, 3])
    return: image_tensor (torch.Tensor[1, 3, H, W]) 
            image_copy: input image (np.array[H, W, 3])
    """
    transform = transforms.Compose([
        transforms.Resize(cfg.TRAIN.input_shape), 
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])
    
    image_copy = image.copy()
    image = Image.fromarray(image[:, :, ::-1])
    image = transform(image)
    image = torch.unsqueeze(image, 0).cuda()

    return image, image_copy

@torch.no_grad()
def forward(image, model, cfg):
    H, W = cfg.TRAIN.input_shape
    k = cfg.DATASET.k
    starting_point, vps = model(image)

    # decode output
    exist = F.softmax(starting_point[0], dim = 1).cpu().numpy()
    x = F.softmax(starting_point[0][:, 1:1+k*W], dim = 1).cpu().numpy()
    y = F.softmax(starting_point[0][:, 1+k*W:1+k*(W+H)], dim = 1).cpu().numpy()
    end = F.softmax(starting_point[0][:, 1+k*(W+H):1+k*(W+2*H)], dim = 1).cpu().numpy()
    exist = exist[:, 0]
    x = np.argmax(x, axis=1)
    y = np.argmax(y, axis=1)
    end = np.argmax(end, axis=1)

    # map x, y to oringal image size
    x_map = lambda x: (x + 1) * 1280 / (k * W) - 1
    y_map = lambda y: (y + 1) * 720 / (k * H) - 1
    x = [x_map(x_) for x_ in x]
    y = [y_map(y_) for y_ in y]
    end = [y_map(end_) for end_ in end]
    point = np.array(list(zip(x, y)))

    # round predicted y
    for j in range(len(y)):
         y[j] += 9 - y[j] % 10 if (9 - y[j] % 10) <= 5 else -y[j] % 10 - 1

    # convert vp vector(N) to vps(N - 1, 2)
    vp_y = int(vps[0][0].sigmoid().item() * 720)
    decode_vpx = lambda x: int(x.item() * 640 + 640)
    vps = np.array([[decode_vpx(vp), vp_y] for vp in vps[0][1:]])

    return exist, point, end, vps

def draw_lane(vp, point):
    """
    input:  - vp(np.array[N, 2])
            - start_point(np.array[2]) 
    return: - lanes
    """
    lane = VanishingPoint.reconstruct_lane(vp, point, val=False)
    # draw points of the lane
    for j in range(len(lane)):
        if(lane[j][0] > 1280 or int(lane[j][1])>720 or lane[j][0]<0 or lane[j][1] < vps[0][1] + 10):
            lane[j][0] = -2
            continue
        cv2.circle(image_copy, (int(lane[j][0]), int(lane[j][1])), 5, (0, 255, 0), -1)
    # draw starting point
    cv2.circle(image_copy, np.array(point).astype(int), 20, (126, 255, 79), -1)
    fill_lane = np.zeros(56) - 2
    for i in range(len(lane)):
        fill_lane[int((lane[i][1]-160)/10)] = lane[i][0]
    
    return np.array(fill_lane).astype(int)


if __name__ == '__main__':
    # load config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--vis', action='store_true', help='whether to visulize predicted result')
    args = parser.parse_args()
    config = DictToClass._to_class(load_config(args.config))
    
    device = "cuda" if config.EVAL.cuda else 'cpu'

    # load model
    model = LVPNet(
        H=config.TRAIN.input_shape[0],
        W=config.TRAIN.input_shape[1],
        k=config.DATASET.k,
        lanes_num=config.MODEL.lanes_num,
        vp_length=config.MODEL.vp_length
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval().to(device)

    # load test dataset
    testset_path = os.path.join(config.DATASET.base_path, "test_label.json")
    with open(testset_path, 'r') as f:
        test_lines = f.readlines()

    # start evalute test dataset
    for i in tqdm(range(len(test_lines))):
        data = json.loads(test_lines[i])
        lanes_label = data["lanes"]
        image_path = os.path.join(config.DATASET.base_path + "\\test_set", data['raw_file'])

        label = get_exist_lanes_label(lanes_label)
        image = cv2.imread(image_path)
        image, image_copy = preprocess_image(image, config)
        exist, point, end, vps = forward(image, model, config)

        lanes = []
        for i in range(len(point)):
            # non-exist
            if(exist[i] > config.EVAL.exist_threshold):
                cv2.circle(image_copy, np.array(point[i]).astype(int), 20, (79, 55, 164), -1)
            # exist
            else:
                index = (719 - point[i][1])//10 if point[i][1] != 719 else 1
                end_index = int((719 - end[i])//10)
                vp = vps[int(index):int(end_index)]
                lanes.append(draw_lane(vp, point[i]))

        lanes = lanes[:-1] if len(lanes) == 6 else lanes
        lanes = [lane.tolist() for lane in lanes]
        y = list(np.linspace(160, 710, 56))

        if args.vis:
            print(LaneEval.bench(lanes, label, y, 0))
            cv2.imshow('predicted_image', image_copy)
            cv2.waitKey(0)
        else:
            lane_information = {
                "raw_file": data['raw_file'], 
                "h_samples": y, 
                "lanes": lanes, 
                "run_time": 0
            }
            with open('temp.json', 'a', encoding='utf-8') as f:
                f.writelines(json.dumps(lane_information) + '\n')
    
    if not args.vis:
        print(
            LaneEval.bench_one_submit(
                'temp.json', 
                testset_path
            )
        )
        os.remove('temp.json')
