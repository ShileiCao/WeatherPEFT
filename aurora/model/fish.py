import torch
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
from tqdm import tqdm


def calculate_the_importance_label_dynamic(cuda_device, grad_type, optimizer=None):
    gradients_dict = {}
    for fish_params in optimizer.param_groups:
        for fish_param, fish_name in zip(fish_params['params'], fish_params['names']):
            # if 'backbone' in fish_name and 'bias' in fish_name:
            gradients_dict[fish_name] = torch.zeros_like(fish_param).to(cuda_device)

    if grad_type == "absolute":
        grad_method = torch.abs
    elif grad_type == "square":
        grad_method = torch.square

    for fish_params in optimizer.param_groups:
        for fish_param, fish_name in zip(fish_params['params'], fish_params['names']):
            # if 'backbone' in fish_name and 'bias' in fish_name:
            gradients_dict[fish_name] += grad_method(fish_param.grad).data

    return gradients_dict


def create_mask_gradient_dynamic(keep_ratio, grad_type='absolute', optimizer=None, weight=None):
    original_device = "cuda" if torch.cuda.is_available() else "cpu"
    cuda_device = "cuda" if torch.cuda.is_available() else "cpu"
    importance_method = calculate_the_importance_label_dynamic
    gradients = importance_method(cuda_device, grad_type, optimizer=optimizer)

    # add sizes and aggregate tensors
    sizes = {}
    tensors = []

    classifier_size = 0
    all_params_size = 0

    classifier_mask_dict = {}

    for k, v in gradients.items():
        # don't count classifier layer, they should be all trainable
        if "classifier" in k:
            classifier_size += torch.prod(torch.tensor(v.shape)).item()
            classifier_mask_dict[k] = torch.ones_like(v).to(original_device)
        else:
            sizes[k] = v.shape
            tensors.append(v.view(-1))

        all_params_size += torch.prod(torch.tensor(v.shape)).item()

    tensors = torch.cat(tensors, 0)

    keep_num = int(all_params_size * keep_ratio) - classifier_size

    assert keep_num > 0

    # random noise
    tensors = tensors / tensors.max()
    tensors_noise = tensors * weight * torch.rand_like(tensors)

    top_pos = torch.topk(tensors_noise, keep_num)[1]

    masks = torch.zeros_like(tensors_noise, device=cuda_device)

    masks[top_pos] = 1

    assert masks.long().sum() == len(top_pos)

    mask_dict = {}

    now_idx = 0
    for k, v in sizes.items():
        end_idx = now_idx + torch.prod(torch.tensor(v))
        mask_dict[k] = masks[now_idx: end_idx].reshape(v).to(original_device)
        now_idx = end_idx

    assert now_idx == len(masks)

    # Add the classifier's mask to mask_dict
    mask_dict.update(classifier_mask_dict)

    # Print the parameters for checking
    classifier_size = 0
    all_params_size = 0
    pretrain_weight_size = 0
    
    for k, v in mask_dict.items():
        if "classifier" in k:
            classifier_size += (v == 1).sum().item()
        else:
            pretrain_weight_size += (v == 1).sum().item()

        all_params_size += torch.prod(torch.tensor(v.shape)).item()
    
    # print(pretrain_weight_size, classifier_size, all_params_size)
    # print(f"trainable parameters: {(pretrain_weight_size + classifier_size) / all_params_size * 100} %")

    return mask_dict