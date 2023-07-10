import os
import sys, os

sys.path.append(os.getcwd())
import json

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from torch.utils.data.dataset import Dataset
from tools.vanishing_point import VanishingPoint
from dataset.transform import transform
from tools.lane import LaneEval


class TusimpleDataset(Dataset):
    def __init__(self, file_line, input_shape, train, *args, **kwargs):
        super(TusimpleDataset, self).__init__()
        self.length = len(file_line)
        self.file_line = file_line
        self.input_shape = input_shape
        self.train = train                                   
        self._default_parameters = {
            "base_path": "dataset\\tusimple",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "sigma": 8,
            "omega_x": 5,
            "omega_y": 5,
            "omega_end": 3,
            "k": 2
        }
        self._default_parameters.update(**kwargs)
        for name, value in self._default_parameters.items():
            setattr(self, name, value)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = json.loads(self.file_line[index])
        image_name = data['file_name']
        lanes = np.array(data['lanes'])
        vanishing_points = np.array(data['vp_label'])

        # image data augmentation
        y = np.linspace(160, 710, 56)
        image = Image.open(os.path.join(self.base_path  + "\\train_set" , image_name))
        image, lanes, vanishing_points, y = transform()(image, lanes, vanishing_points, y, self.train)
        image = cv2.resize(image, (self.input_shape[1], self.input_shape[0]))
        image =  np.transpose((np.array(image, np.float64)/255.0), [2, 0, 1])

        # image normalization (B G R)
        for channel in range(3):
            image[:, :, channel] = (image[:, :, channel] - self.mean[channel])/self.std[channel]
        
        # normalize vp_y to (0, 1), vp_x to (-1, 1)
        vanishing_points[0] /= 720
        vanishing_points[vanishing_points == 0] = 640
        vanishing_points[1:] = (vanishing_points[1:] - 640)/640
        if(vanishing_points[1]>1 or vanishing_points[1]<-1):
            vanishing_points[1] = 0
        for i in range(len(vanishing_points[1:]) - 1):
            if(vanishing_points[2+i]>1 or vanishing_points[2+i]<-1):
                vanishing_points[2+i] = vanishing_points[1+i]

        # get x, y coordinates of starting point and y coordinate of ending point
        start_point = []
        end_point = []
        for i in range(len(lanes)):
            if(len(lanes[i]) == 48):
                lanes[i] = list(np.append(np.zeros(8) - 2, np.array(lanes[i])))
            lane = list(filter(lambda x : x[0]>0 and x[0]<1280, zip(lanes[i], y)))
            start_point.append((-1, -1)) if (lane == []) else start_point.append(lane[-1])
            end_point.append(-1) if (lane == []) else end_point.append(lane[0][1])

        # order of lanes label in ("./label_data_0313.json") is wrong
        start_point = np.array(start_point).astype(int)
        tan_theta = list(map(lambda x:np.arctan(x[1]/(640 - x[0] + 1e-5)), start_point[start_point > 0].reshape(-1, 2)))
        tan_theta = list(map(lambda x:x if x>0 else x + np.pi, tan_theta))
        index = np.argsort(tan_theta)
        rank = np.array([0, 1, 2, 3, 4, 5])
        before_rank = np.array([0, 1, 2, 3, 4, 5])
        error_rank = (start_point > 0).reshape(-1, 2)[:, 0].astype(int)
        rank[error_rank > 0] = rank[error_rank > 0][index]
        lanes = np.array(lanes)
        end_point = np.array(end_point)
        # swap order by angle
        start_point[before_rank] = start_point[rank]
        end_point[before_rank] = end_point[rank]
        lanes[before_rank] = lanes[rank]

        # generate starting point label
        start_point_gt = []
        H = self.input_shape[0]
        W = self.input_shape[1]
        vector_length = 1 + self.k * (2 * H + W)
        for end_point_y, (start_point_x, start_point_y) in zip(end_point, start_point):
            exist = np.zeros((1))
            if(start_point_x == -1):
                x = np.zeros((int(self.k * W)))
                y = np.zeros((int(self.k * H)))
                end = np.zeros((int(self.k * H)))
                exist[0] = 0.9 + 0.1 / vector_length
                x = x + 0.1 / vector_length
                y = y + 0.1 / vector_length
                end = end + 0.1 / vector_length
            else:
                x = np.arange(0, int(self.k * W), 1, np.float32)
                y = np.arange(0, int(self.k * H), 1, np.float32)
                end = np.arange(0, int(self.k * H), 1, np.float32)

                if(int(start_point_x) - 1 < 0):
                    mu_x = 0
                else:
                    mu_x = int(start_point_x * W / 1280 * self.k) - 1
                mu_y = int(start_point_y * H / 720 * self.k) - 1
                mu_end = int(end_point_y * H / 720 * self.k) - 1

                # Gaussian label smoothing
                x = (np.exp(- ((x - mu_x) ** 2) / (2 * self.sigma ** 2)))/(self.sigma * np.sqrt(np.pi*2))
                y = (np.exp(- ((y - mu_y) ** 2) / (2 * self.sigma ** 2)))/(self.sigma * np.sqrt(np.pi*2))
                end = (np.exp(- ((end - mu_end) ** 2) / (2 * self.sigma ** 2)))/(self.sigma * np.sqrt(np.pi*2))

                # Laplace label smoothing
                # x = (np.exp(- (np.abs(x - mu_x)) / self.sigma))/(2 * self.sigma)
                # y = (np.exp(- (np.abs(y - mu_y)) / self.sigma))/(2 * self.sigma)
                # end = (np.exp(- (np.abs(end - mu_end)) / self.sigma))/(2 * self.sigma)
                # weights 3 3 1
                x = x * self.omega_x
                y = y * self.omega_y
                end = end * self.omega_end
            
            start_point_gt.append(np.concatenate([exist, x, y, end], -1))

        return image, np.array(start_point_gt), vanishing_points, lanes
    


def tusimple_dataset_collate(batch):
    images = []
    start_point_gts  = []
    vps = []
    lanes = []
    for img, start_point_gt, vp, lane in batch:
        images.append(img)
        start_point_gts.append(start_point_gt)
        vps.append(vp)
        lanes.append(lane)

    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    start_point_gts  = torch.from_numpy(np.array(start_point_gts)).type(torch.FloatTensor)
    vps  = torch.from_numpy(np.array(vps)).type(torch.FloatTensor)
    lanes  = torch.from_numpy(np.array(lanes)).type(torch.FloatTensor)

    return images, start_point_gts, vps, lanes


# dataloader test (visulize data augmentation)
if __name__ == '__main__':
    base_path = r"D:\Code\pytorch-auto-drive-master\dataset\tusimple"
    with open(base_path + '/train_vp.json', 'r') as f:
        train_lines = f.readlines()
    dataset = TusimpleDataset(train_lines, [360, 640], False)

    for img, start_point, vp, lanes in dataset:
        x = start_point[:, 1:1281]
        y = start_point[:, 1281:2001]
        end = start_point[:, 2001:]
        img = np.transpose(img*255, [1, 2, 0])
        img = (img).astype(np.uint8)[:, :, ::-1]
        img = cv2.resize(img, [1280, 720])

        img = (img * 0.5).astype(np.uint8)
        vp[:1] = vp[0] * 720
        cv2.line(img, [0, int(vp[0])], [1280, int(vp[0])], (255, 0, 255), 5)
        vp[1:] = vp[1:] * 640 + 640
        vp = [[i, vp[0]] for i in vp[1:]]
        [cv2.circle(img, (int(v[0]), int(v[1])), 5, (255, 0, 0), -1) for v in vp]
        
        color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
        eval_lanes = []
        lanes_label = []
        for i in range(len(x)):
            # decode x, y of startpoint and y of ending point
            _x = np.argmax(x[i])
            _y = np.argmax(y[i])
            _end = np.argmax(end[i])
            if(_x == 0 and _y == 0):
                continue
            else:
                start_index = int((719 - _y) // 10)
                end_index = int((719 - _end)//10)
                lane = VanishingPoint().reconstruct_lane(vp[start_index:end_index], [int(_x), int(_y)])

                # compute quality of label
                eval_lane = VanishingPoint().reconstruct_lane(vp[start_index:end_index], [int(_x), int(_y)], val = True)
                if(lanes.shape[1] == 48):
                    eval_lane = eval_lane[8:]
                eval_lanes.append(eval_lane)
                lanes_label.append(lanes[i])

                # reconstructed lanes label
                for j in range(len(lane)):
                    if(lane[j][0] > 0 and lane[j][1] > 0):
                        cv2.circle(img, (int(lane[j][0]), int(lane[j][1])), 5, color[i], -1)
                
                # original lanes label
                # for j in range(len(lanes[i])):
                #         cv2.circle(img, (int(lanes[i][j]), 160+10*j), 5, (255, 255, 255), -1)
                cv2.circle(img, (_x, _y), 5, color[i], -1)
        
        cv2.imshow('img', img)
        cv2.waitKey(0)

        y = list(np.linspace(160, 710, 56))
        acc = LaneEval().bench(eval_lanes, lanes_label, y, 0)
        print(acc)

    # for _, x, vp, lanes in train_loader:
    #     print(x.shape,  vp.shape)
    #     break
     

    
        
 


