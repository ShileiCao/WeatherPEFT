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
from datetime import timedelta
import pandas as pd

def train_one_epoch_postprocess(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None, start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None,  
                    lat = None, lon = None, level = None, static_vars = None, surf_vars=None, upper_vars=None, model_name="Aurora", criterion=None,
                    out_surf_vars = None, out_upper_vars = None, out_upper_level = None,use_ours=False, total_step=None):
    
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

        surface, upper, surface_scale, surface_targets, upper_scale, upper_targets, time_points = batch
         
        B, T, V, C, H, W = upper.shape
        _, _, V1, _, _ = surface.shape
        time_points = [pd.Timestamp(point.item(), unit='ns') for point in time_points[0]]
        
        sfc_weight = [14, 7, 7]
        pl_weight = [8, 0.1]
        
    
        batch = Batch(
            surf_vars={
                var:surface[:,:,i] for i, var in enumerate(surf_vars)
            },
            static_vars = static_vars,
            atmos_vars={
                var:upper[:,:,i] for i, var in enumerate(upper_vars)
            },
            metadata=Metadata(
                lat=lat,
                lon=lon,
                time=time_points,
                atmos_levels=level,
            ),
        ).to(device)

        pred_surface, pred_upper = model(batch)
        del batch
        surface_scale, surface_targets, upper_scale, upper_targets = surface_scale.to(device), surface_targets.to(device), upper_scale.to(device), upper_targets.to(device)
        loss_surs = 0.0
        loss_upps = 0.0
        
        
        if out_surf_vars:
            for i, var in enumerate(out_surf_vars):
                mu_surface = pred_surface[:, i*2] * surface_scale[:,1,i] + surface_scale[:,0,i]
                sigma_surface = torch.exp(pred_surface[:, i*2+1]) * surface_scale[:,1,i]
                loss_sur = criterion(mu_surface, sigma_surface, surface_targets[:,i]) * sfc_weight[i]
                metric_logger.meters[f"CRPS_{var}"].update(loss_sur.item(), n=B)
                loss_surs += loss_sur

        if out_upper_vars:
            for i, var in enumerate(out_upper_vars):
                mu_upper = pred_upper[:,i*2, level.index(out_upper_level[i])] * upper_scale[:,1,i] + upper_scale[:,0,i]
                sigma_upper = torch.exp(pred_upper[:, i*2+1, level.index(out_upper_level[i])]) * upper_scale[:,1,i]
                loss_upp = criterion(mu_upper, sigma_upper, upper_targets[:,i]) * pl_weight[i]
                metric_logger.meters[f'CRPS_{var}{str(out_upper_level[i])}'].update(loss_upp.item(), n=B)
                loss_upps += loss_upp

        loss = loss_surs + loss_upps
                
        loss_value = loss.item()
        if math.isnan(loss_value) or math.isinf(loss_value):
            print(f"Loss is NaN or Inf at {time_points[0]}")

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss /= update_freq
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(data_iter_step + 1) % update_freq == 0,use_ours=use_ours, weight=0.2*(1-it/total_step), k_value=0.001)
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

def validation_one_epoch_postprocess(data_loader, model, device, lat = None, lon = None, level = None, criterion1=None, criterion2=None,
                                   static_vars = None, surf_vars=None, upper_vars=None, model_name="Aurora", surface_efis= None, upper_efis = None,
                                   out_surf_vars = None, out_upper_vars = None, out_upper_level = None,):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'

    # switch to evaluation mode
    model.eval()
    
    for batch in metric_logger.log_every(data_loader, 20, header):
        
        surface, upper, surface_scale, surface_targets, upper_scale, upper_targets, time_points = batch
        
        B, T, V, C, H, W = upper.shape
        time_points = [pd.Timestamp(point.item(), unit='ns') for point in time_points[0]]
        loss_sur = []
        loss_upp = []
        efi_sur = []
        efi_upp = []
        
        batch = Batch(
            surf_vars={
                var:surface[:,:,i] for i, var in enumerate(surf_vars)
            },
            static_vars = static_vars,
            atmos_vars={
                var:upper[:,:,i] for i, var in enumerate(upper_vars)
            },
            metadata=Metadata(
                lat=lat,
                lon=lon,
                time=time_points,
                atmos_levels=level,
            ),
        ).to(device)
        with torch.inference_mode():
            pred_surface, pred_upper = model(batch)
        
        surface_scale, surface_targets, upper_scale, upper_targets = surface_scale.to(device), surface_targets.to(device), upper_scale.to(device), upper_targets.to(device)
        
        if out_surf_vars:
            for i, var in enumerate(out_surf_vars):
                mu_surface = pred_surface[:, i*2] * surface_scale[:,1,i] + surface_scale[:,0,i]
                sigma_surface = torch.exp(pred_surface[:, i*2+1]) * surface_scale[:,1,i]
                crps = criterion1(mu_surface, sigma_surface, surface_targets[:,i])
                loss_sur.append(crps)
                test_loss_efi = []
                for j in range(len(time_points)):
                    # try:
                    date = time_points[j]
                    ds_efi = surface_efis[i]
                    efi_tensor = torch.as_tensor(ds_efi.sel(time=date)["efi"].values)[:-1].to(device)
                    loss_efi = criterion2(mu_surface[j], sigma_surface[j], surface_targets[j,i], efi_tensor)
                    test_loss_efi.append(loss_efi.item())
                    # except KeyError:
                    #     pass
                efi_sur.append(np.mean(test_loss_efi))
                
        if out_upper_vars:
            for i, var in enumerate(out_upper_vars):
                mu_upper = pred_upper[:,i*2, level.index(out_upper_level[i])] * upper_scale[:,1,i] + upper_scale[:,0,i]
                sigma_upper = torch.exp(pred_upper[:, i*2+1, level.index(out_upper_level[i])]) * upper_scale[:,1,i]

                loss_upp.append(criterion1(mu_upper, sigma_upper, upper_targets[:,i]))
                test_loss_efi = []
                for j in range(len(time_points)):

                    date = time_points[j]
                    ds_efi = upper_efis[i]
                    efi_tensor = torch.as_tensor(ds_efi.sel(time=date)["efi"].values)[:-1].to(device)
                    loss_efi = criterion2(mu_upper[j], sigma_upper[j], upper_targets[j,i], efi_tensor)
                    test_loss_efi.append(loss_efi.item())

                efi_upp.append(np.mean(test_loss_efi))
                    
                    
                    
        B = surface.shape[0]
        for i, var in enumerate(out_surf_vars):
            metric_logger.meters[f'CRPS_{var}'].update(loss_sur[i].item(), n=B)
            metric_logger.meters[f'EECRPS_{var}'].update(efi_sur[i].item(), n=B)
            
        
        for i, var in enumerate(out_upper_vars):
            metric_logger.meters[f'CRPS_{var}{str(out_upper_level[i])}'].update(loss_upp[i].item(), n=B)
            metric_logger.meters[f'EECRPS_{var}{str(out_upper_level[i])}'].update(efi_upp[i].item(), n=B)
            
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print()
    print("Metric:")
    print(metric_logger)
    print()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
