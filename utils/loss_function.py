import torch
import torch.nn as nn


class KLDivLoss(nn.Module):
    """loss = (target * (target.log() - x)).sum()"""
    def __init__(self):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim = 2)
        self.loss = nn.KLDivLoss(reduction = 'mean')
    
    def forward(self, x, target):
        x = self.log_softmax(x)

        return self.loss(x, target)


class VPLoss(nn.Module):
    def __init__(self, lambda_vp, cuda):
        super().__init__()
        self.vp_weights = torch.linspace(1, 1, 56)
        if cuda:
            self.vp_weights = self.vp_weights.cuda()
        self.vp_weights[0] = self.vp_weights[0].add(lambda_vp)
        self.vp_loss =  nn.L1Loss('mean')
    
    def forward(self, vp, vp_label):
        vp[..., 0] =torch.sigmoid(vp[..., 0])
        vp = vp * self.vp_weights
        vp_label = vp_label * self.vp_weights
        vp_loss = self.vp_loss(vp[vp_label!=0], vp_label[vp_label!=0])

        return vp_loss