import math

import timm
import torch.nn as nn


class LVPNet(nn.Module):
    def __init__(self, H=360, W=640, k=2, lanes_num=6, vp_length=56):
        """
        - H, W: height, width of input image
        - k: splitting factor (for details, refer to SimCC (https://arxiv.org/abs/2107.03332))
        - lanes_num: maximum number of lanes
        - vp_length: length of vp vector
        """
        super().__init__()
        self.downsampling = lambda x: math.ceil(x/2)
        self.H, self.W = H, W    # height, width of input image
        self.H_, self.W_ = H, W  # height, width of last feature map (H_ = 1/32 H)
        for _ in range(5):
            self.H_ = self.downsampling(self.H_)
            self.W_ = self.downsampling(self.W_)
        self.lanes_num = lanes_num
        self.vp_length = vp_length
        self.k = k

        self.backbone = timm.create_model('resnet18', pretrained=True)
        self.conv = nn.Sequential(nn.Conv2d(512, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
                                  nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU())
        
        self.reducer  = nn.Sequential(nn.Conv2d(256, self.lanes_num, 3, 1, 1), nn.BatchNorm2d(self.lanes_num), nn.ReLU())
        self.mlp_head_point = nn.Linear(self.H_*self.W_, 1+self.k*(self.W+2*self.H))
      
        self.reducer2  = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU())
        self.avgpool = nn.AvgPool2d((self.H_, self.W_))
        self.mlp_head_vp = nn.Linear(256, self.vp_length)

        self._init_weights()
    
    def forward(self, x):
        feature = self.backbone.forward_features(x)
        feature = self.conv(feature)
    
        # vanishing point branch
        horizon_feature = self.reducer2(feature)
        horizon_feature = self.avgpool(horizon_feature)
        vp = self.mlp_head_vp(horizon_feature.view(-1, 256))

        # starting point branch
        point_feature = self.reducer(feature)
        point_feature = point_feature.view(-1, self.lanes_num, self.H_*self.W_)
        point = self.mlp_head_point(point_feature)
        
        return point, vp

    def _init_weights(self):
        for m in self.reducer:
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight.data, 0, 0.12, -0.3, 0.3)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data[:] = 1.
                m.bias.data.zero_()

        for m in self.reducer2:
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight.data, 0, 0.12, -0.3, 0.3)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data[:] = 1.
                m.bias.data.zero_()
        
        for m in self.conv:
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight.data, 0, 0.12, -0.3, 0.3)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data[:] = 1.
                m.bias.data.zero_()
        
        nn.init.trunc_normal_(self.mlp_head_point.weight.data, 0, 0.12, -0.3, 0.3)
        nn.init.trunc_normal_(self.mlp_head_vp.weight.data, 0, 0.12, -0.3, 0.3)
    

# model test
if __name__ == '__main__':
    from torchsummary import summary
    import torch
    
    # forward test
    model = LVPNet().cuda()
    x = torch.randn(1, 3, 360, 640).cuda()
    starting_point, vp = model(x)
    print(starting_point.shape, vp.shape)

    summary(model, (3, 360, 640), device='cuda')
