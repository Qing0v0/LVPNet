import argparse
import datetime

import torch.optim as optim
from timm.utils.model_ema import ModelEmaV2
from torch.utils.data import DataLoader

from dataset.tusimple_dataloader import (TusimpleDataset,
                                         tusimple_dataset_collate)
from model.LVPNet import LVPNet
from utils.config_utils import *
from utils.fit_utils import fit_one_epoch
from utils.loss_log import LossLog
from utils.train_utils import *


# load config
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='Path to config file')
args = parser.parse_args()
config = load_config(args.config)
dataset_config = config["DATASET"]
config = DictToClass._to_class(config)

# set seed
if config.TRAIN.seed:
    seed_torch(config.TRAIN.seed)
else:
    torch.backends.cudnn.benchmark = True

# load model
model = LVPNet(
    H=config.TRAIN.input_shape[0],
    W=config.TRAIN.input_shape[1],
    k=config.DATASET.k,
    lanes_num=config.MODEL.lanes_num,
    vp_length=config.MODEL.vp_length
)
if config.TRAIN.cuda:
    model = model.cuda()

# load dataset
trainset_path = os.path.join(config.DATASET.base_path, 'train_vp.json')
valset_path = os.path.join(config.DATASET.base_path, 'val_vp.json')
with open(trainset_path,"r") as f:                                     
    train_lines = f.readlines() 
with open(valset_path,"r") as f:                          
    val_lines = f.readlines() 
train_dataset = TusimpleDataset(train_lines, config.TRAIN.input_shape, True, **dataset_config)
val_dataset   = TusimpleDataset(val_lines, config.TRAIN.input_shape, False, **dataset_config)

train_loader = DataLoader(
    train_dataset, 
    shuffle = True, 
    batch_size = config.TRAIN.batch_size, 
    num_workers = config.TRAIN.num_workers,
    drop_last = True, 
    collate_fn = tusimple_dataset_collate
)
val_loader = DataLoader(
    val_dataset,
    shuffle = True, 
    batch_size = config.TRAIN.batch_size, 
    num_workers = config.TRAIN.num_workers,
    drop_last = True, 
    collate_fn = tusimple_dataset_collate
)
step = (
    len(train_lines) // config.TRAIN.batch_size, 
    len(val_lines) // config.TRAIN.batch_size
)

# optimizer
lr_scheduler_func, init_lr_fit = tune_lr(
    config.TRAIN.batch_size, 
    config.TRAIN.init_lr, 
    config.TRAIN.min_lr, 
    config.TRAIN.epoch
)
optimizer = optim.AdamW(
    model.parameters(), 
    init_lr_fit, 
    betas = (config.TRAIN.momentum, 0.999), 
    weight_decay = config.TRAIN.weight_decay
)

# set ema model
ema = ModelEmaV2(model, decay=config.TRAIN.ema, device='cuda' if config.TRAIN.cuda else 'cpu')

# loss log
time_str = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
log_dir = os.path.join('log', "log_" + str(time_str))
loss_log = LossLog(log_dir)

# start train
for epoch in range(config.TRAIN.epoch):
    set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
    fit_one_epoch(model, optimizer, epoch, train_loader, val_loader, step, loss_log, config)
    ema.update(model)
    torch.save(ema.module.state_dict(), os.path.join(log_dir, 'checkpoint\\ema.pth'))
