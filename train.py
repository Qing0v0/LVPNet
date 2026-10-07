import datetime
import os

import torch.optim as optim
from timm.utils.model_ema import ModelEmaV2
from torch.utils.data import DataLoader

from dataset.dataloader import DepthDataset, depth_collate_fn
from model.model import Depth
from utils.loss_log import LossLog
from utils.train_utils import *
from utils.fit_utils import fit_one_epoch

# set config
trainset_path = r"depth_label\depth_train.json"
valset_path = r"depth_label\depth_val.json"
input_shape = [340, 640]

batch_size = 2
epoch = 150
init_lr = 0.0005                                                
min_lr = 0.000005
momentum = 0.9
weight_decay = 0
save_period = 10

ema = 0.965

# model
model = Depth().cuda()

# load dataset
with open(trainset_path, "r", encoding="utf-8") as f:
    train_line = f.readlines()
    train_set = DepthDataset(train_line, input_shape, True)
with open(valset_path, "r", encoding="utf-8") as f:
    val_line = f.readlines()
    val_set = DepthDataset(val_line, input_shape, False)
train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size, num_workers=0, drop_last=True, collate_fn=depth_collate_fn)
val_loader = DataLoader(val_set, shuffle=True, batch_size=batch_size, num_workers=0, drop_last=True, collate_fn=depth_collate_fn)

# lr & optim
step = (len(train_line) // batch_size, len(val_line) // batch_size)
lr_scheduler_func, init_lr_fit = tune_lr(batch_size, init_lr, min_lr, epoch)
optimizer = optim.AdamW(model.parameters(), init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay)

# set ema model
ema = ModelEmaV2(model, decay=ema, device='cuda')

# loss log
time_str = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
log_dir = os.path.join('log', "log_" + str(time_str))
loss_log = LossLog(log_dir)

# start train
for epoch in range(epoch):
    set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
    fit_one_epoch(model, optimizer, epoch, train_loader, val_loader, step, loss_log)
    ema.update(model)
    torch.save(ema.module.state_dict(), os.path.join(log_dir, 'checkpoint\\ema.pth'))