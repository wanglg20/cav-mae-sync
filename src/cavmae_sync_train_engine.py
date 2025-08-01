# -*- coding: utf-8 -*-
# @Time    : 6/10/21 11:00 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : traintest.py

# not rely on supervised feature

import sys
import os
import datetime
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
from utils.metrics import *
import time
import torch
from torch import nn
import numpy as np
import pickle
from torch.cuda.amp import autocast,GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import wandb

def resume_training(audio_model, optimizer, exp_dir, device):
    """
    Resume training from the latest checkpoint if available.

    Args:
        audio_model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        exp_dir (str): The directory where model checkpoints are saved.
        device (torch.device): The device to load the model and optimizer onto.

    Returns:
        int: The starting epoch for training.
    """
    checkpoint_path = os.path.join(exp_dir, "models", "best_audio_model.pth")
    optimizer_path = os.path.join(exp_dir, "models", "best_optim_state.pth")
    if os.path.exists(checkpoint_path) and os.path.exists(optimizer_path):    
        # Load optimizer state
        optimizer_state = torch.load(optimizer_path, map_location=device)
        optimizer.load_state_dict(optimizer_state)
        # Extract the last saved epoch
        ckpt_list = os.listdir(os.path.join(exp_dir, "models"))
        epochs = [int(f.split('.')[-2]) for f in ckpt_list if f.endswith('.pth') and f.startswith('audio_model')]
        start_epoch = max(epochs) if epochs else 0
        # # Load model weights
        checkpoint_path = os.path.join(exp_dir, "models", f"audio_model.{start_epoch}.pth")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        audio_model.load_state_dict(checkpoint)
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        print(f"Resumed training from epoch {start_epoch}")
    else:
        print("No checkpoint found. Starting training from scratch.")
        start_epoch = 1

    return start_epoch


def train(audio_model, train_loader, test_loader, args, local_rank):
    device = torch.device(f'cuda:{local_rank}')
    audio_model.to(device)
    audio_model = DDP(audio_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    print('running on ' + str(device))
    torch.set_grad_enabled(True)

    batch_time, per_sample_time, data_time, per_sample_data_time, per_sample_dnn_time = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    loss_all_meter, loss_a_meter, loss_v_meter, loss_c_meter = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    progress = []

    best_epoch, best_loss = 0, np.inf
    global_step, epoch = 0, 0
    start_time = time.time()
    exp_dir = args.exp_dir
    audio_model = audio_model.to(device)
    trainables = [p for p in audio_model.parameters() if p.requires_grad]
    print('Total parameter number is : {:.3f} M'.format(sum(p.numel() for p in audio_model.parameters()) / 1e6))
    print('Total trainable parameter number is : {:.3f} M'.format(sum(p.numel() for p in trainables) / 1e6))
    optimizer = torch.optim.Adam(trainables, args.lr, weight_decay=5e-7, betas=(0.95, 0.999))

    # use adapt learning rate scheduler, for preliminary experiments only, should not use for formal experiments
    if args.lr_adapt == True:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=args.lr_patience, verbose=True)
        print('Override to use adaptive learning rate scheduler.')
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(args.lrscheduler_start, 1000, args.lrscheduler_step)),gamma=args.lrscheduler_decay)
        print('The learning rate scheduler starts at {:d} epoch with decay rate of {:.3f} every {:d} epoches'.format(args.lrscheduler_start, args.lrscheduler_decay, args.lrscheduler_step))

    print('now training with {:s}, learning rate scheduler: {:s}'.format(str(args.dataset), str(scheduler)))

    # #optional, save epoch 0 untrained model, for ablation study on model initialization purpose
    # torch.save(audio_model.state_dict(), "%s/models/audio_model.%d.pth" % (exp_dir, epoch))
    if args.resume:
        epoch = resume_training(audio_model, optimizer=optimizer, exp_dir=args.exp_dir, device=device)
        global_step = (epoch - 1) * len(train_loader) 

    epoch += 1
    scaler = GradScaler()
    
    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")
    result = np.zeros([args.n_epochs, 10])  # for each epoch, 10 metrics to record
    audio_model.train()
    while epoch < args.n_epochs + 1:
        train_loader.sampler.set_epoch(epoch)
        test_loader.sampler.set_epoch(epoch)
        begin_time = time.time()
        end_time = time.time()
        audio_model.train()
        print('---------------')
        print(datetime.datetime.now())
        print("current #epochs=%s, #steps=%s" % (epoch, global_step))
        print('current masking ratio is {:.3f} for both modalities; audio mask mode {:s}'.format(args.masking_ratio, args.mask_mode))


        args.n_print_steps = 50
        for i, (a_input, v_input, _, _, _) in enumerate(train_loader):
            
            B = a_input.size(0)
            a_input = a_input.to(device, non_blocking=True)
            v_input = v_input.to(device, non_blocking=True)

            data_time.update(time.time() - end_time)
            per_sample_data_time.update((time.time() - end_time) / a_input.shape[0])
            dnn_start_time = time.time()

            with autocast():
                loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, mask_a, mask_v, c_acc, recon_a, recon_v, cls_a, cls_v = audio_model(a_input, v_input, args.masking_ratio, args.masking_ratio, mae_loss_weight=args.mae_loss_weight, contrast_loss_weight=args.contrast_loss_weight, mask_mode=args.mask_mode)
                # this is due to for torch.nn.DataParallel, the output loss of 4 gpus won't be automatically averaged, need to be done manually
                loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, c_acc = loss.sum(), loss_mae.sum(), loss_mae_a.sum(), loss_mae_v.sum(), loss_c.sum(), c_acc.mean()

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # loss_av is the main loss
            loss_all_meter.update(loss.item(), B)
            loss_a_meter.update(loss_mae_a.item(), B)
            loss_v_meter.update(loss_mae_v.item(), B)
            loss_c_meter.update(loss_c.item(), B)
            batch_time.update(time.time() - end_time)
            per_sample_time.update((time.time() - end_time)/a_input.shape[0])
            per_sample_dnn_time.update((time.time() - dnn_start_time)/a_input.shape[0])

            # log:
            if args.use_wandb and local_rank == 0:
                wandb.log({
                    'train loss_all': loss.item(),
                    'train vision mae loss': loss_mae_v.item(),
                    'train audio mae loss': loss_mae_a.item(),
                    'train contra loss': loss_c.item(),
                    'train c_acc': c_acc.item(),
                    'train loss mae': loss_mae.item(),
                    'lr': optimizer.param_groups[0]['lr'],
                    'iters': (epoch - 1) * len(train_loader) + i,
                    'epoch': epoch
                })
            print_step = global_step % args.n_print_steps == 0
            early_print_step = epoch == 0 and global_step % (args.n_print_steps // 10) == 0
            print_step = print_step or early_print_step
            #print_step = True # for debugging purpose, always print
            if print_step and global_step != 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Per Sample Total Time {per_sample_time.avg:.5f}\t'
                  'Per Sample Data Time {per_sample_data_time.avg:.5f}\t'
                  'Per Sample DNN Time {per_sample_dnn_time.avg:.5f}\t'
                  'Train Total Loss {loss_av_meter.val:.4f}\t'
                  'Train MAE Loss Audio {loss_a_meter.val:.4f}\t'
                  'Train MAE Loss Visual {loss_v_meter.val:.4f}\t'
                  'Train Contrastive Loss {loss_c_meter.val:.4f}\t'
                  'Train Contrastive Acc {c_acc:.3f}\t'.format(
                   epoch, i, len(train_loader), per_sample_time=per_sample_time, per_sample_data_time=per_sample_data_time,
                      per_sample_dnn_time=per_sample_dnn_time, loss_av_meter=loss_all_meter, loss_a_meter=loss_a_meter, loss_v_meter=loss_v_meter, loss_c_meter=loss_c_meter, c_acc=c_acc), flush=True)
                if np.isnan(loss_all_meter.avg):
                    print("training diverged...")
                    return

            end_time = time.time()
            global_step += 1

        print('start validation')
        eval_loss_av, eval_loss_mae, eval_loss_mae_a, eval_loss_mae_v, eval_loss_c, eval_c_acc = validate(audio_model, test_loader, args, local_rank=local_rank)

        print("Eval Audio MAE Loss: {:.6f}".format(eval_loss_mae_a))
        print("Eval Visual MAE Loss: {:.6f}".format(eval_loss_mae_v))
        print("Eval Total MAE Loss: {:.6f}".format(eval_loss_mae))
        print("Eval Contrastive Loss: {:.6f}".format(eval_loss_c))
        print("Eval Total Loss: {:.6f}".format(eval_loss_av))
        print("Eval Contrastive Accuracy: {:.6f}".format(eval_c_acc))

        print("Train Audio MAE Loss: {:.6f}".format(loss_a_meter.avg))
        print("Train Visual MAE Loss: {:.6f}".format(loss_v_meter.avg))
        print("Train Contrastive Loss: {:.6f}".format(loss_c_meter.avg))
        print("Train Total Loss: {:.6f}".format(loss_all_meter.avg))

        # train audio mae loss, train visual mae loss, train contrastive loss, train total loss
        # eval audio mae loss, eval visual mae loss, eval contrastive loss, eval total loss, eval contrastive accuracy, lr
        result[epoch-1, :] = [loss_a_meter.avg, loss_v_meter.avg, loss_c_meter.avg, loss_all_meter.avg, eval_loss_mae_a, eval_loss_mae_v, eval_loss_c, eval_loss_av, eval_c_acc, optimizer.param_groups[0]['lr']]
        np.savetxt(exp_dir + '/result.csv', result, delimiter=',')
        print('validation finished')

        if eval_loss_av < best_loss:
            best_loss = eval_loss_av
            best_epoch = epoch

        if best_epoch == epoch:
            torch.save(audio_model.state_dict(), "%s/models/best_audio_model.pth" % (exp_dir))
            torch.save(optimizer.state_dict(), "%s/models/best_optim_state.pth" % (exp_dir))

        if args.save_model == True and epoch % args.save_interval == 0:
            torch.save(audio_model.state_dict(), "%s/models/audio_model.%d.pth" % (exp_dir, epoch))

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(-eval_loss_av)
        else:
            scheduler.step()

        print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))

        finish_time = time.time()
        print('epoch {:d} training time: {:.3f}'.format(epoch, finish_time-begin_time))

        epoch += 1

        batch_time.reset()
        per_sample_time.reset()
        data_time.reset()
        per_sample_data_time.reset()
        per_sample_dnn_time.reset()
        loss_all_meter.reset()
        loss_a_meter.reset()
        loss_v_meter.reset()
        loss_c_meter.reset()

def validate(audio_model, val_loader, args, local_rank):
    batch_time = AverageMeter()
    # if not isinstance(audio_model, nn.DataParallel):
    #     audio_model = nn.DataParallel(audio_model)
    #audio_model = audio_model.to(device)
    audio_model.eval()

    end = time.time()
    device = torch.device(f'cuda:{local_rank}')
    A_loss, A_loss_mae, A_loss_mae_a, A_loss_mae_v, A_loss_c, A_c_acc = [], [], [], [], [], []
    with torch.no_grad():
        for i, (a_input, v_input, _, _, _) in enumerate(val_loader):
            a_input = a_input.to(device)
            v_input = v_input.to(device)
            with autocast():
                loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, mask_a, mask_v, c_acc = audio_model(a_input, v_input, args.masking_ratio, args.masking_ratio, mae_loss_weight=args.mae_loss_weight, contrast_loss_weight=args.contrast_loss_weight, mask_mode=args.mask_mode)
                loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, c_acc = loss.sum(), loss_mae.sum(), loss_mae_a.sum(), loss_mae_v.sum(), loss_c.sum(), c_acc.mean()
            
            A_loss.append(loss.to('cpu').detach())
            A_loss_mae.append(loss_mae.to('cpu').detach())
            A_loss_mae_a.append(loss_mae_a.to('cpu').detach())
            A_loss_mae_v.append(loss_mae_v.to('cpu').detach())
            A_loss_c.append(loss_c.to('cpu').detach())
            A_c_acc.append(c_acc.to('cpu').detach())
            batch_time.update(time.time() - end)
            end = time.time()

        loss = np.mean(A_loss)
        loss_mae = np.mean(A_loss_mae)
        loss_mae_a = np.mean(A_loss_mae_a)
        loss_mae_v = np.mean(A_loss_mae_v)
        loss_c = np.mean(A_loss_c)
        c_acc = np.mean(A_c_acc)

    return loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, c_acc