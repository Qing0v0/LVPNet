import os

import torch
from tqdm import tqdm

from .loss_function import *
from .train_utils import get_lr


def fit_one_epoch(model, optimizer, epoch, train_loader, val_loader, step, loss_log):
    EPOCH = 150
    train_loss = 0
    val_loss = 0
    train_step, val_step = step

    point_loss = HungarianPointLoss(1, 0.05)
    vp_loss = VPLoss(5)

    print("Start Train")
    pbar = tqdm(
        total=train_step, 
        desc=f'Epoch {epoch + 1}/{EPOCH}', 
        postfix=dict, 
        mininterval=0.3
    )
    model.train()

    # 训练集
    for i, batch in enumerate(train_loader):
        if(i >= train_step):
            break
        image, points_label, depth_label, vp_label = batch
        with torch.no_grad():
            image = image.cuda()
            points_label = points_label
            vp_label = vp_label.cuda()
            depth_label = depth_label
        optimizer.zero_grad()

        # get loss
        points, vp, depth = model(image)
        loss = point_loss(depth, points, depth_label, points_label) + 0.05 * vp_loss(vp, vp_label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pbar.set_postfix(**{
            'total_loss': train_loss / (i + 1), 
            'lr': get_lr(optimizer)}
        )
        pbar.update(1)
    pbar.close()
    print('Finish Train')

    # 验证集
    print('Start Validation')
    pbar = tqdm(
        total=val_step, 
        desc=f'Epoch {epoch + 1}/{EPOCH}', 
        postfix=dict, 
        mininterval=0.3
    )
    model.eval()

    for i, batch in enumerate(val_loader):
        if(i >= train_step):
            break
        image, points_label, depth_label, vp_label = batch
        with torch.no_grad():
            image = image.cuda()
            points_label = points_label
            vp_label = vp_label.cuda()
            depth_label = depth_label

        # get loss
        points, vp, depth = model(image)
        loss = point_loss(depth, points, depth_label, points_label) + 0.05 * vp_loss(vp, vp_label)
        val_loss += loss.item()
        pbar.set_postfix(**{'val_loss'  : val_loss / (i + 1),
                            'lr'        : get_lr(optimizer)})
        pbar.update(1)
    pbar.close()
    print('Finish Validation')

    train_loss_per_step = train_loss/ train_step
    val_loss_per_step = val_loss/ val_step
    log_dir = loss_log.get_log_dir()
    loss_log.append_loss(epoch + 1, train_loss_per_step, val_loss_per_step)
    print('Epoch:'+ str(epoch+1) + '/' + str(EPOCH))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (train_loss_per_step, val_loss_per_step))

    if (len(loss_log.val_losses) <= 1 or 
        val_loss_per_step <= min(loss_log.val_losses)):
        torch.save(
            model.state_dict(), 
            os.path.join(log_dir, "checkpoint\\best_epoch_weights.pth")
        )