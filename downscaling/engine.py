import os
import numpy as np
import math
import sys
from typing import Iterable, Optional
import torch
from dataset import score
import utils
from scipy.special import softmax
from dataset import utils_data, score
from einops import rearrange
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from aurora import Batch, Metadata
from aurora.normalisation import normalise_surf_var, normalise_atmos_var, unnormalise_surf_var, unnormalise_atmos_var
from datetime import timedelta, datetime
import pandas as pd


def train_one_epoch_downscale(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None, start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None,  patch_size = None, 
                    lat = None, lon = None, level = None, static_vars = None, surf_vars=None, upper_vars=None, use_ours=False, total_step=None):
    
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    optimizer.zero_grad()

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        surface, upper, target_surface, target_upper, time_points  = batch
        B, T, V, C, H, W = upper.shape
        B, V, C, H1, W1 = target_upper.shape
        surface = torch.stack([F.interpolate(surface[:,i], (H1,W1), mode="bilinear") for i in range(T)], dim=1)
        upper = torch.stack([F.interpolate(upper[:,i].reshape(B, -1, H, W), (H1,W1), mode="bilinear") for i in range(T)], dim=1).reshape(B, T, V, C, H1, W1)
        time_points = [pd.Timestamp(point.item(), unit='s') for point in time_points[1]]

        batch = Batch(
            surf_vars={
                "2t": surface[:,:2,0],
                "10u": surface[:,:2,1],
                "10v": surface[:,:2,2],
            },
            static_vars = static_vars,
            atmos_vars={
                "t": upper[:,:2,0],
                "u": upper[:,:2,1],
                "v": upper[:,:2,2],
                "q": upper[:,:2,3],
                "z": upper[:,:2,4],
            },
            metadata=Metadata(
                lat=lat,
                lon=lon,
                time=time_points,
                atmos_levels=level,
            ),
        ).to(device)
        
        
        surf_preds, upper_preds = model(batch)
        del batch
        
        target_surface = torch.stack([normalise_surf_var(target_surface[:,i], var, None) for i, var in enumerate(surf_vars)], dim=1).to(device)
        target_upper = torch.stack([normalise_atmos_var(target_upper[:,i], var, level) for i, var in enumerate(upper_vars)], dim=1).to(device)
        
        loss_sur = F.mse_loss(surf_preds, target_surface)
        loss_upp = F.mse_loss(upper_preds, target_upper)
        loss = 0.25 * loss_sur + loss_upp
        
        loss_value = loss.item()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss /= update_freq
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(data_iter_step + 1) % update_freq == 0, use_ours=use_ours, weight=0.2*(1-it/total_step), k_value=0.001)
        if (data_iter_step + 1) % update_freq == 0:
            optimizer.zero_grad()
            
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def validation_one_epoch_downscale(data_loader, model, device, lat = None, lon = None, level = None, static_vars = None, surf_vars=None, upper_vars=None, mode="ours"):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 20, header):

        surface, upper, target_surface, target_upper, time_points  = batch
        B, T, V, C, H, W = upper.shape
        B, V, C, H1, W1 = target_upper.shape
        surface = torch.stack([F.interpolate(surface[:,i], (H1,W1), mode="bilinear") for i in range(T)], dim=1)
        upper = torch.stack([F.interpolate(upper[:,i].reshape(B, -1, H, W), (H1,W1), mode="bilinear") for i in range(T)], dim=1).reshape(B, T, V, C, H1, W1)
        time_points = [pd.Timestamp(point.item(), unit='s') for point in time_points[1]]
        
        batch = Batch(
            surf_vars={
                "2t": surface[:,:2,0],
                "10u": surface[:,:2,1],
                "10v": surface[:,:2,2],
            },
            static_vars = static_vars,
            atmos_vars={
                "t": upper[:,:2,0],
                "u": upper[:,:2,1],
                "v": upper[:,:2,2],
                "q": upper[:,:2,3],
                "z": upper[:,:2,4],
            },
            metadata=Metadata(
                lat=lat,
                lon=lon,
                time=time_points,
                atmos_levels=level,
            ),
        ).to(device)
        # compute output
        with torch.inference_mode():
            pred_surface, pred_upper = model(batch)
            
        target_surface = torch.stack([normalise_surf_var(target_surface[:,i], var, None) for i, var in enumerate(surf_vars)], dim=1).to(device)
        target_upper = torch.stack([normalise_atmos_var(target_upper[:,i], var, level) for i, var in enumerate(upper_vars)], dim=1).to(device)
        
        loss_surface = F.mse_loss(pred_surface, target_surface)
        loss_upper = F.mse_loss(pred_upper, target_upper)
        loss = 0.5 * loss_surface + 0.5 * loss_upper
        
        target_surface = torch.stack([unnormalise_surf_var(target_surface[:,i], var, None) for i, var in enumerate(surf_vars)], dim=1)
        target_upper = torch.stack([unnormalise_atmos_var(target_upper[:,i], var, level) for i, var in enumerate(upper_vars)], dim=1)
        
        pred_surface = torch.stack([unnormalise_surf_var(pred_surface[:,i], var, None) for i, var in enumerate(surf_vars)], dim=1)
        pred_upper = torch.stack([unnormalise_atmos_var(pred_upper[:,i], var, level) for i, var in enumerate(upper_vars)], dim=1)
        
        
        valid_sur = ["2t", "10u", "10v"]
        valid_upp = ["t", "z"]
        valid_level = [850, 500]
        
        time_str = datetime.strftime(time_points[0], '%Y%m%d%H')
        
        
        rmse_surface, rmse_upper, mean_bias_surface, mean_bias_upper = score.evaluate_rmse_downscale(pred_surface, pred_upper, target_surface, target_upper, torch.linspace(90-1.40625/2, -90+1.40625/2, 128))

        metric_logger.update(valid_loss=loss.item())
        batch_size = surface.shape[0]
        
        for var in valid_sur:
            metric_logger.meters[f'rmse_{var}'].update(rmse_surface[surf_vars.index(var)], n=batch_size)
            metric_logger.meters[f'mean_bias_{var}'].update(mean_bias_surface[surf_vars.index(var)], n=batch_size)
        for var, l in zip(valid_upp, valid_level):
            metric_logger.meters[f'rmse_{var}{l}'].update(rmse_upper[upper_vars.index(var),level.index(l)], n=batch_size)
            metric_logger.meters[f'mean_bias_{var}{l}'].update(mean_bias_upper[upper_vars.index(var),level.index(l)], n=batch_size)
        
        # count=count+1
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print()
    print("Metric:")
    print(metric_logger)
    print()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}