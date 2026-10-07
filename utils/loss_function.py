from turtle import forward
from xml.etree.ElementInclude import include
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment


class KLDivLoss(nn.Module):
    """loss = (target * (target.log() - x)).sum()"""
    def __init__(self):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim = 1)
        self.loss = nn.KLDivLoss(reduction = 'mean')
    
    def forward(self, x, target):
        x = self.log_softmax(x)

        return self.loss(x, target)

class VPLoss(nn.Module):
    def __init__(self, lambda_vp):
        super().__init__()
        self.vp_weights = torch.linspace(1, 1, 50).cuda()
        self.vp_weights[0] = self.vp_weights[0].add(lambda_vp)
        self.vp_loss =  nn.L1Loss('mean')
    
    def forward(self, vp, vp_label):
        vp[..., 0] =torch.sigmoid(vp[..., 0])
        vp = vp * self.vp_weights
        vp_label = vp_label * self.vp_weights
        vp_loss = self.vp_loss(vp[vp_label!=0], vp_label[vp_label!=0])

        return vp_loss

class HungarianMatcher(nn.Module):
    def __init__(self):
        super().__init__()
        self.k = 3
        self.alpha = 0.2
        self.beta = 0.6
        self.gamma = 0.2
    
    def forward(self, depths, points, depth_targets, point_targets):
        """
        depths: [B, Q]
        points: [B, Q, 1981]
        depth_targets: [B, G]
        point_targets: [B, G, 1981]
        """
        points = torch.softmax(points, dim=2)
        prob = 1 - points[..., 0]
        points = points[..., 1:]
        sizes = [target.shape[0] for target in depth_targets]
        point_targets = torch.cat([p for p in point_targets], dim=0)
        depth_targets = torch.cat([d for d in depth_targets], dim=0)
        point_targets = point_targets[..., 1:]

        B, Q, _ = points.shape
        G = depth_targets.size(0)

        # 1. Local maxima prior
        _, max_indices = torch.nn.functional.max_pool1d(
            prob.unsqueeze(1),
            kernel_size=self.k, stride=1,
            padding=(self.k - 1) // 2, 
            return_indices=True
        )
        max_indices = max_indices.squeeze(1)
        indices = torch.arange(0, Q, dtype=prob.dtype, device=prob.device).unsqueeze(0).expand_as(max_indices)
        local_maxima = (max_indices == indices).flatten().unsqueeze(-1).expand(-1, G)
        prob = prob.flatten() # BQ
        points = points.flatten(end_dim=1)  # BQ x 1980
        depths = depths.flatten(end_dim=1)

        # 2. Compute the classification cost.
        print(prob)
        cost_label = self.alpha * prob.unsqueeze(-1).expand(-1, G)  # BQ x G

        # 3. Compute the curve sampling cost
        cost_curve = self.get_cost_curve(points, point_targets)  # BQ x G
        cost_curve = self.beta * cost_curve.clamp(min=0, max=1)
        cost_depth = self.get_cost_depth(depths, depth_targets)
        cost_depth = self.gamma * cost_depth.clamp(min=0, max=1)
        C = local_maxima * cost_label * cost_curve * cost_depth
        C = -C.view(B, Q, -1).cpu()

        indices = [linear_sum_assignment(c[i].detach().numpy()) for i, c in enumerate(C.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

    def get_cost_curve(self, points, point_targets):
        NOMALIZE = lambda x, y: x.type(torch.FloatTensor).div_(y)
        x, xt = torch.argmax(points[:, 1:961], dim=1), torch.argmax(point_targets[:, 1:961], dim=1)
        y, yt = torch.argmax(points[:, 961:1471], dim=1), torch.argmax(point_targets[:, 961:1471], dim=1)
        e, et = torch.argmax(points[:, 1471:], dim=1), torch.argmax(point_targets[:, 1471:], dim=1)

        pred =  torch.stack((NOMALIZE(x, 960), NOMALIZE(y, 510), NOMALIZE(e, 510)), dim = -1).cuda()
        target = torch.stack((NOMALIZE(xt, 960), NOMALIZE(yt, 510), NOMALIZE(et, 510)), dim=-1).cuda()
        return 1 - torch.cdist(pred, target, p=3).div_(3)

    def get_cost_depth(self, depths, depth_targets):
        return 1 - torch.cdist(depths, depth_targets, p=1).div_(50)

class HungarianPointLoss(nn.Module):
    def __init__(self, point_weight, depth_weight):
        super().__init__()
        self.point_weight = point_weight
        self.depth_weight = depth_weight
        self.matcher = HungarianMatcher()
        self.klloss = KLDivLoss()
    
    def forward(self, depths, points, depth_targets, point_targets):
        target_point_labels = self.get_non_exist_vector().repeat((points.size(0), points.size(1), 1))

        total_targets = 0
        for pt in point_targets:
            total_targets += pt.numel()
        
        if total_targets > 0:
            indices = self.matcher(depths, points, depth_targets, point_targets)
            # 只对匹配好的深度进行损失计算
            idx = self.get_permutation_idx(indices)
            
            # 找出对应的depths
            depths = depths[idx]
            depth_targets = torch.cat([t[i] for t, (_, i) in zip(depth_targets, indices)], dim=0)

            # 涉及是否存在对points和targe_point_label计算损失
            for B, (t, (i, j)) in enumerate(zip(point_targets, indices)):
                target_point_labels[B][i] = t[j]
        else:
            # TODO 暂时没有未标注车道线的图像
            ...

        # 计算深度损失和车道线损失
        depth_loss = self.depth_loss(depths, depth_targets)
        point_loss = self.point_loss(points, target_point_labels)
        loss = self.depth_weight * depth_loss + self.point_weight * point_loss
        return loss
    
    def get_non_exist_vector(self):
        x = torch.zeros((1981)).cuda()
        x[0] = x[0].add_(0.9)
        x = x.add_(1 / 1981)

        return x

    def get_permutation_idx(self, indices, target=False):
        if not target:
            batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
            image_idx = torch.cat([src for (src, _) in indices])
        else:
            batch_idx = torch.cat([torch.full_like(src, i) for i, (_, src) in enumerate(indices)])
            image_idx = torch.cat([src for (_, src) in indices])

        return batch_idx, image_idx
    
    def depth_loss(self, depths, depth_targets):
        return F.l1_loss(depths[depth_targets > 0], depth_targets[depth_targets > 0], reduction="mean")
    
    def point_loss(self, points, point_targets, non_exist_weight=0.3):
        exist = point_targets[..., 0]
        non_exist_indice, exist_indice = torch.where(exist > 0.5), torch.where(exist < 0.5)

        # 分别计算存在和不存在的点损失
        non_exist_loss = self.klloss(points[non_exist_indice], point_targets[non_exist_indice])
        exist_loss = self.klloss(points[exist_indice], point_targets[exist_indice])
        point_loss = non_exist_weight * non_exist_loss + exist_loss
        return point_loss



if __name__ == "__main__":
    d = torch.rand((4, 10, 50)).cuda()
    p = torch.rand((4, 10, 1981)).cuda()
    dt = [torch.rand((3, 50)), torch.rand((4, 50)), torch.rand((3, 50)), torch.rand((3, 50))]
    pt = [torch.rand((3, 1981)), torch.rand((4, 1981)), torch.rand((3, 1981)), torch.rand((3, 1981))]
    dt = [d.cuda() for d in dt]
    pt = [p.cuda() for p in pt]
    # 匹配测试
    hm = HungarianMatcher()
    print(hm(d, p, dt, pt))

    # 损失函数测试
    # hpl = HungarianPointLoss(0.1, 0.1)
    # print(hpl(d, p, dt, pt))
