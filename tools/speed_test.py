import time
import torch
from tqdm import tqdm
import os
import sys

# from PolyLaneNet import PolyRegression
# from LSTR.LSTR import model as lstr
# from UFLD.model import parsingNet
# from PINet import hourglass_netwok
# from LaneATT.lib.config import Config as laneatt_cfg
from model.LVPNet import StartPointDetection

MODEL_FACTORY = {}

def register_model(fn):
    def register(fn):
        model_name = fn.__name__
        MODEL_FACTORY[model_name] = fn

    return register(fn)

# line anchor
# @register_model
# def LaneATT():
#     cfg = laneatt_cfg(r'LaneATT\cfgs\laneatt_tusimple_resnet18.yml')
#     model = cfg.get_model()
#     test_parameters = cfg.get_test_parameters()
#     input_shape = (1, 3, 360, 640)

#     return model, input_shape, test_parameters

# @register_model
# def CLRNet():
#     sys.path.append(os.path.join(os.getcwd(), 'CLRNet'))
#     from clrnet.utils.config import Config
#     from clrnet.models.registry import build_net

#     cfg = Config.fromfile(r'CLRNet\configs\clrnet\clr_resnet18_tusimple.py')
#     model = build_net(cfg)
#     input_shape = (1, 3, 320, 800)

#     return model, input_shape

# # row anchor
# @register_model
# def UFLD():
#     model = parsingNet(
#         size=(288, 800), 
#         pretrained=False, 
#         backbone='18', 
#         cls_dim = (100+1, 56, 4), 
#         use_aux=False
#         )
#     input_shape = (1, 3, 288, 800)

#     return model, input_shape

# # curve
# @register_model
# def PolyLaneNet():
#     model =  PolyRegression(35, 'efficientnet-b0', True, [0, 0, 0, 0])
#     input_shape = (1, 3, 360, 640)

#     return model, input_shape

# @register_model
# def LSTR():
#     model = lstr()
#     input_shape = (1, 3, 360, 640)

#     return model, input_shape

# @register_model
# def BezierNet():
#     sys.path.append(os.path.join(os.getcwd(), 'BezierLaneNet'))
#     from BezierLaneNet.utils.args import read_config
#     from BezierLaneNet.utils.models import MODELS

#     cfg = read_config(r'BezierLaneNet\configs\lane_detection\bezierlanenet\resnet18_tusimple_aug1b.py')
#     model = MODELS.from_dict(cfg['model'])
#     model.eval()
#     input_shape = (1, 3, 360, 640)

#     return model, input_shape

# # keypoint
# @register_model
# def PINet():
#     model = hourglass_netwok.lane_detection_network()
#     input_shape = (1, 3, 256, 512)

#     return model, input_shape

# vp
@register_model
def LVPNet():
    model = StartPointDetection()
    input_shape = (1, 3, 360, 640)

    return model, input_shape


def speed_test(model_name):
    if model_name != 'LaneATT':
        model, input_shape = MODEL_FACTORY[model_name]()
    else:
        model, input_shape, params = MODEL_FACTORY[model_name]()
    torch.backends.cudnn.benchmark = True
    model = model.cuda()
    model.eval()
    x = torch.randn(input_shape).cuda()

    with torch.no_grad():
        total_time = 0
        
        # warm
        for _ in range(10):
            model(x)
        for _ in tqdm(range(100000)):
            torch.cuda.synchronize()
            t1 = time.time()
            if not model_name == 'LaneATT':
                model(x)
            else:
                model(x, **params)
            torch.cuda.synchronize()
            t2 = time.time()
            total_time += (t2 - t1)
    fps = 100000 / total_time

    return fps


if __name__ == '__main__':
    fps = speed_test('LVPNet')
    print(fps)