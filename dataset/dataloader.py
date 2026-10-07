import os
import sys, os

sys.path.append(os.getcwd())
import json

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from dataset.transform import transform
from tqdm import tqdm
from utils.misc import fill_lane

class DepthDataset(Dataset):
    def __init__(self, file_line, input_shape, train):
        super(DepthDataset, self).__init__()
        self.length = len(file_line)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.sigma = 8
        self.file_line = file_line
        self.input_shape = input_shape
        self.train = train
        self.base_path = r"E:\dataset\ONCE-3dlanes\data"

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        # load label
        data = json.loads(self.file_line[index])
        image_name = data["image_name"]
        lanes = np.array(data["lanes"])
        depth = np.array(data["depth"])
        # calibration = np.array(data["calibration"])
        vanishing_points = np.array(data["vp"])

        # read image and augment
        y = np.arange(530, 1030, 10)
        image = Image.open(self.base_path + image_name)
        image, lanes, vanishing_points, y, depth = transform()(image, lanes, depth, vanishing_points, y, self.train)
        image = cv2.resize(image, (self.input_shape[1], self.input_shape[0]))
        image = np.transpose((np.array(image, np.float64)/255.0), [2, 0, 1])

        # normalization
        for channel in range(3):
            image[channel, :, :] = (image[channel, :, :] - self.mean[channel])/self.std[channel]

        # vp normalization
        vanishing_points[0] /= 1020
        vanishing_points[vanishing_points == 0] = 960
        vanishing_points[1:] = (vanishing_points[1:] - 960) / 960

        # depth normalization
        depth /= 50
        depth[depth < 0] = 0

        # start point vector
        lanes = [fill_lane(lane, unfill=True) for lane in lanes]
        start_point = []
        end_point = []
        for lane in lanes:
            start_point.append(lane[-1])
            end_point.append(lane[0][1])
        
        point_gt = []
        for end_point_y, (start_point_x, start_point_y) in zip(end_point, start_point):
            x = np.arange(0, 960, 1, np.float32)
            y = np.arange(0, 510, 1, np.float32)
            end = np.arange(0, 510, 1, np.float32)

            if(int(start_point_x) - 1 < 0):
                mu_x = 0
            else:
                mu_x = int(start_point_x / 2) - 1
            mu_y = int(start_point_y / 2) - 1
            mu_end = int(end_point_y /2) - 1

            # Gaussian label smoothing
            x = (np.exp(- ((x - mu_x) ** 2) / (2 * self.sigma ** 2)))/(self.sigma * np.sqrt(np.pi*2))
            y = (np.exp(- ((y - mu_y) ** 2) / (2 * self.sigma ** 2)))/(self.sigma * np.sqrt(np.pi*2))
            end = (np.exp(- ((end - mu_end) ** 2) / (2 * self.sigma ** 2)))/(self.sigma * np.sqrt(np.pi*2))
            x *= 3 
            y *= 3
            end *= 3

            point_gt.append(np.concatenate([[0], x, y, end], -1))

        return image, np.array(point_gt), np.array(depth), np.array(vanishing_points)

  
def depth_collate_fn(batch):
    TO_TENSOR = lambda x: torch.from_numpy(np.array(x)).type(torch.FloatTensor).cuda()
    images = []
    points = []
    depths = []
    vps = []       

    for image, point, depth, vp in batch:
        images.append(image)
        points.append(TO_TENSOR(point))
        depths.append(TO_TENSOR(depth))
        vps.append(vp)

    images = TO_TENSOR(images)
    vps = TO_TENSOR(vps)

    return images, points, depths, vps


# dataset test
if __name__ == '__main__':
    with open(r"depth_label/depth_train.json", "r", encoding="utf-8") as f:
        json_list = f.readlines()
    
    dataset = DepthDataset(json_list, [340, 640], True)
    # for image, point, depth, vp in tqdm(dataset):
    #     print(len(point))
    from torch.utils.data import DataLoader
    train_loader = DataLoader(dataset, shuffle=True, batch_size=4, num_workers = 0, drop_last = True, collate_fn=depth_collate_fn)
    for im, point, depth, vp in train_loader:
        point = torch.cat(point, dim=0)
        point = torch.softmax(point, 1)
        point = point.cpu().numpy()
        for p in point:
            x = np.argmax(p[1:961])
            y = np.argmax(p[961:1471])
            end = np.argmax(p[1471:1981])
            print(end * 2, x * 2, y * 2)
        exit(0)