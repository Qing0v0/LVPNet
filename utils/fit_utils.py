import os

import torch
from tqdm import tqdm

from .loss_function import *
from .train_utils import get_lr


def fit_one_epoch(model, optimizer, epoch, train_loader, val_loader, step, loss_log, cfg):
    train_loss = 0
    val_loss = 0
    train_step, val_step = step

    starting_point_loss = KLDivLoss()
    vp_loss = VPLoss(cfg.TRAIN.lambda_vp, cfg.TRAIN.cuda)

    # Start Train
    print("Start Train")
    pbar = tqdm(
        total=train_step, 
        desc=f'Epoch {epoch + 1}/{cfg.TRAIN.epoch}', 
        postfix=dict, 
        mininterval=0.3
    )
    model.train()

    for i, batch in enumerate(train_loader):
        if(i >= train_step):
            break
        image, starting_point_label, vp_label, lanes = batch
        with torch.no_grad():
            if cfg.TRAIN.cuda:
                image = image.cuda()
                starting_point_label = starting_point_label.cuda()
                vp_label = vp_label.cuda()
                lanes = lanes.cuda()
        optimizer.zero_grad()

        # get loss
        starting_point, vp = model(image)
        loss = cfg.TRAIN.lambda_1 * starting_point_loss(starting_point, starting_point_label) +\
               cfg.TRAIN.lambda_2 * vp_loss(vp, vp_label)
        loss.backward()
        optimizer.step()
    
        train_loss      += loss.item()
        pbar.set_postfix(**{
            'total_loss': train_loss / (i + 1), 
            'lr': get_lr(optimizer)}
        )
        pbar.update(1)
    pbar.close()
    print('Finish Train')


    # Start Validation
    print('Start Validation')
    pbar = tqdm(
        total=val_step, 
        desc=f'Epoch {epoch + 1}/{cfg.TRAIN.epoch}', 
        postfix=dict, 
        mininterval=0.3
    )
    model.eval()

    for i, batch in enumerate(val_loader):
        if(i >= val_step):
            break
        image, starting_point_label, vp_label, lanes = batch
        with torch.no_grad():
            if cfg.TRAIN.cuda:
                image = image.cuda()
                starting_point_label = starting_point_label.cuda()
                vp_label = vp_label.cuda()
                lanes = lanes.cuda()
            
            starting_point, vp = model(image)
            loss = cfg.TRAIN.lambda_1 * starting_point_loss(starting_point, starting_point_label) +\
                   cfg.TRAIN.lambda_2 * vp_loss(vp, vp_label)

            val_loss += loss.item()
            pbar.set_postfix(**{'val_loss'  : val_loss / (i + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)
    pbar.close()
    print('Finish Validation')

    # append loss to log
    train_loss_per_step = train_loss/ train_step
    val_loss_per_step = val_loss/ val_step
    log_dir = loss_log.get_log_dir()
    loss_log.append_loss(epoch + 1, train_loss_per_step, val_loss_per_step)
    print('Epoch:'+ str(epoch+1) + '/' + str(cfg.TRAIN.epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (train_loss_per_step, val_loss_per_step))
    
    # save checkpoint
    if (epoch + 1) % cfg.TRAIN.save_period == 0 or epoch + 1 == cfg.TRAIN.epoch:
        torch.save(
            model.state_dict(), 
            os.path.join(
                log_dir, 
                "checkpoint\\epoch{}-loss{:.3f}-valloss{:.3f}".format(epoch+1, train_loss_per_step, val_loss_per_step)
            )
        )

    if (len(loss_log.val_losses) <= 1 or 
        (val_loss_per_step <= min(loss_log.val_losses) and cfg.TRAIN.save_best_epoch)):
        torch.save(
            model.state_dict(), 
            os.path.join(log_dir, "checkpoint\\best_epoch_weights.pth")
        )

    if cfg.TRAIN.save_last_epoch:
        torch.save(
            model.state_dict(), 
            os.path.join(log_dir, "checkpoint\\last_epoch_weights.pth")
        ) 