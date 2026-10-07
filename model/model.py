import os
import sys
sys.path.append(os.getcwd())

import math
import timm
import torch
import torch.nn as nn
from utils.misc import AVERAGE_DEPTH
from .DSConv import DSConv


class Depth(nn.Module):
    def __init__(self, H=340, W=640, k=1.5, pred_num=20, vp_length=50):
        super().__init__()
        self.register_buffer(
            name="lane_relative_depth",
            tensor=torch.from_numpy(AVERAGE_DEPTH).type(torch.FloatTensor).div_(50)
        )
        self.downsampling = lambda x: math.ceil(x/2)
        self.H, self.W = H, W    # height, width of input image
        self.H_, self.W_ = H, W  # height, width of last feature map (H_ = 1/32 H)
        for _ in range(5):
            self.H_ = self.downsampling(self.H_)
            self.W_ = self.downsampling(self.W_)
        self.pred_num = pred_num
        self.vp_length = vp_length
        self.k = k
        self.depth_feature_num = 4

        self.backbone = timm.create_model('resnet18', pretrained=True)
        self.conv = nn.Sequential(nn.Conv2d(512, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
                                  nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU())
        
        self.reducer = nn.Sequential(nn.Conv2d(256, self.pred_num, 3, 1, 1), nn.BatchNorm2d(self.pred_num), nn.ReLU())
        self.dsconv = DSConv(self.pred_num, self.pred_num, 15, 1, 0, True, "cuda")
        self.mlp_head_point = nn.Linear(self.H_*self.W_, int(1+self.k*(self.W+2*self.H)))
        self.mlp_head_depth_feature = nn.Linear(self.H_*self.W_, 50 * self.depth_feature_num)
        self.mlp_head_depth = nn.Linear(self.depth_feature_num + 1, 1)
      
        self.reducer2  = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU())
        self.avgpool = nn.AvgPool2d((self.H_, self.W_))
        self.mlp_head_vp = nn.Linear(256, self.vp_length)
    
    def forward(self, x):
        feature = self.backbone.forward_features(x)
        feature = self.conv(feature)
    
        # vanishing point branch
        horizon_feature = self.reducer2(feature)
        horizon_feature = self.avgpool(horizon_feature)
        vp = self.mlp_head_vp(horizon_feature.view(-1, 256))

        # starting point branch
        point_feature = self.reducer(feature)
        point_feature = self.dsconv(point_feature)
        point_feature = point_feature.view(-1, self.pred_num, self.H_*self.W_)
        point = self.mlp_head_point(point_feature)

        # depth
        depth_feature = self.mlp_head_depth_feature(point_feature)
        depth_feature = depth_feature.view(depth_feature.shape[:-1] + (-1, self.depth_feature_num))
        depth_feature = torch.cat((depth_feature, self.lane_relative_depth.repeat(depth_feature.size(0), self.pred_num, 1).unsqueeze(-1)), -1)
        depth = self.mlp_head_depth(depth_feature)[..., 0]
  
        return point, vp, depth

# model test
if __name__ == '__main__':
    model = Depth(H=510, W=960, k=1).cuda()
    x = torch.randn(1, 3, 510, 960).cuda()
    starting_point, vp, depth = model(x)
    print(starting_point.shape, vp.shape, depth.shape)


