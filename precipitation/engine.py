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
from evaluate import evaluate_pre
import xarray as xr
from datetime import timedelta, datetime



def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None, start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, out_surf_vars = None,
                    lat = None, lon = None, level = None, static_vars = None,surf_vars=None, upper_vars=None, model_name="Aurora", use_ours=False, total_step=None, noise_weight=None, k_value=None):
    model.train(True)
    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
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

        surface, upper, time_points  = batch
        time_points = [pd.Timestamp(point.item(), unit='s') for point in time_points[1]]
        target_surface = normalise_surf_var(surface[:,2:,4], "tp", None)

        batch = Batch(
            surf_vars={
                var:surface[:,:2,i] for i, var in enumerate(surf_vars)
            },
            static_vars = static_vars,
            atmos_vars={
                var:upper[:,:2,i] for i, var in enumerate(upper_vars)
            },
            metadata=Metadata(
                lat=lat,
                lon=lon,
                time=time_points,
                atmos_levels=level,
            ),
        ).to(device)
        
        
        output, _ = model(batch)
        del batch

        loss = criterion(output, target_surface.to(device))
        
        loss_value = loss.item()
        if math.isnan(loss_value) or math.isinf(loss_value):
            print(f"Loss is NaN or Inf at {time_points[0]}")


        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss /= update_freq
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(data_iter_step + 1) % update_freq == 0, use_ours=use_ours, weight=noise_weight*(1-it/total_step), k_value=k_value)

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

@torch.no_grad()
def validation_one_epoch(data_loader, model, device, lat = None, lon = None, level = None, static_vars = None, surf_vars=None, upper_vars=None, model_name="Aurora", out_surf_vars = None, mode="ours"):
    criterion = nn.L1Loss()          
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'

    # switch to evaluation mode
    model.eval()
    
    times = []
    outputs = []
    for batch in metric_logger.log_every(data_loader, 20, header):
        surface, upper, time_points  = batch
        time_points = [pd.Timestamp(point.item(), unit='s') for point in time_points[1]]
        times.append(time_points)
        
        target_surface = normalise_surf_var(surface[:,2:,4], "tp", None)
        
        batch = Batch(
            surf_vars={
                var:surface[:,:2,i] for i, var in enumerate(surf_vars)
            },
            static_vars = static_vars,
            atmos_vars={
                var:upper[:,:2,i] for i, var in enumerate(upper_vars)
            },
            metadata=Metadata(
                lat=lat,
                lon=lon,
                time=time_points,
                atmos_levels=level,
            ),
        ).to(device)
        
        with torch.inference_mode():
            output, _ = model(batch) 
        loss = criterion(output, target_surface.to(device)) 

        
        loss_value = loss.item()
        metric_logger.update(valid_loss=loss_value)
        
        output = unnormalise_surf_var(output,"tp", None).cpu().numpy()
        target_surface = unnormalise_surf_var(target_surface,"tp", None).cpu().numpy()
        output = np.clip(output, a_min=0.0, a_max=None)
        time_str = datetime.strftime(time_points[0], '%Y%m%d%H')
        outputs.append(output[:,np.newaxis])
        

    time_points = pd.to_datetime(np.concatenate(times, 0)).values
    lat = torch.linspace(90, -90, 721)[126:366].numpy()
    lon = torch.linspace(0, 360, 1441)[:-1][296:536].numpy()
    output_all = np.concatenate(outputs, 0)
    for i, var in enumerate(out_surf_vars): 
        outputs = output_all[:,:,i]
        prediction_timedelta = np.array([np.timedelta64((i+1)*12, 'h')])        
        data_array = xr.DataArray(
            data=outputs,
            dims=["time", "prediction_timedelta", "latitude", "longitude"],
            coords={
                "time": time_points,
                "prediction_timedelta" : prediction_timedelta,
                "latitude": lat,
                "longitude": lon
            },
            attrs={
                "description": "Model prediction for tp",
                "units": "mm"  # 根据实际情况填写
            }
        )
        dataset = xr.Dataset(
            {
                "total_precipitation_6hr": data_array
            },
            attrs={
                "description": "Prediction results from Aurora model",
            }
        )
        
        seeps, acc, rmse = evaluate_pre(dataset)

        metric_logger.meters[f"{var}_seeps"].update(seeps.item())
        metric_logger.meters[f"{var}_acc"].update(acc.item())
        metric_logger.meters[f"{var}_rmse"].update((rmse*1e2).item())
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print()
    print("Metric:")
    print(metric_logger)
    print()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
