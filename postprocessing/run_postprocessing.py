import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
from functools import partial
from pathlib import Path

from timm.models import create_model
from optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner

from engine import train_one_epoch_postprocess, validation_one_epoch_postprocess
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import  multiple_samples_collate
import utils
from aurora import Aurora
from dataset import utils_data
from torch import nn
import xarray as xr

from aurora.batch import interpolate_numpy
from datetime import timedelta
import multiprocessing as mp

from dataset.score import CrpsGaussianLoss, EECRPSGaussianLoss


    
def get_args():
    parser = argparse.ArgumentParser('Finetuning for post-processing', add_help=False)
    parser.add_argument('--model', default='Aurora', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--val_batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=100, type=int)

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--layer_decay', type=float, default=0.75)

    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')


    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)
    parser.add_argument('--short_side_size', type=int, default=224)
    parser.add_argument('--test_num_segment', type=int, default=5)
    parser.add_argument('--test_num_crop', type=int, default=3)
    

    # Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--train_start_date', default='', help='train_start_date')
    parser.add_argument('--train_end_date', default='', help='train_end_date')
    parser.add_argument('--val_start_date', default='', help='val_start_date')
    parser.add_argument('--val_end_date', default='', help='val_end_date')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--use_checkpoint', action='store_true')
    parser.set_defaults(use_checkpoint=False)
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')

    # Dataset parameters
    parser.add_argument('--var', default='T2M', type=str,help='target variable')
    parser.add_argument('--nb_classes', default=400, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=6, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    
    parser.add_argument('--full', action='store_true', dest='full')
    parser.set_defaults(pin_mem=False)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--mode', default='full', help='fine_tuning mode')

    ds_init = None

    return parser.parse_args(), ds_init


def main(args, ds_init):
    utils.init_distributed_mode(args)

    if ds_init is not None:
        utils.create_ds_config(args)

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    # print(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True


    static_vars_ds = xr.open_dataset("../aux_data/static.nc", engine="netcdf4")
    
    lat = torch.linspace(90, -90, 361)
    lon = torch.linspace(0, 360, 721)[:-1]
    level = (500, 850)
    static_vars = {
            # The static variables are constant, so we just get them for the first time.
            "lsm": torch.from_numpy(interpolate_numpy(static_vars_ds["lsm"].values[0], lat=static_vars_ds.latitude.values, lon=static_vars_ds.longitude.values,
                                        lat_new=lat, lon_new=lon)).float(),
            "z": torch.from_numpy(interpolate_numpy(static_vars_ds["z"].values[0], lat=static_vars_ds.latitude.values, lon=static_vars_ds.longitude.values,
                                        lat_new=lat, lon_new=lon)).float(),
            "slt": torch.from_numpy(interpolate_numpy(static_vars_ds["slt"].values[0], lat=static_vars_ds.latitude.values, lon=static_vars_ds.longitude.values,
                                        lat_new=lat, lon_new=lon)).float(),
        }
    
    surf_vars = ('SSTK', 'TCW', 'TCWV', 'CP', 'MSL', 'TCC', 'U10M', 'V10M', 'T2M', 'TP', 'SKT') 
    upper_vars = ["Z", "T", "Q", "W", "D", "U", "V"]
    
    out_surf_vars = []
    out_upper_vars = []
    out_upper_level = []
    
    out_surf_vars = ["T2M", "T2M_std", 'U10M', 'U10M_std', 'V10M', 'V10M_std']
    out_upper_vars = [ "T", "T_std", "Z", "Z_std"]
    out_upper_level = [850, 500]
    
        
    criterion1 = CrpsGaussianLoss()
    criterion2 = EECRPSGaussianLoss()
    surface_efis = []
    upper_efis = []
    
    args.data_path = "../datasets/post-process/process"
    
    if out_surf_vars:
        for var in out_surf_vars[::2]:
            surface_efis.append(xr.open_dataset(os.path.join(args.data_path, f"efi_{var}.nc")))
        
    if out_upper_vars:
        for i, var in enumerate(out_upper_vars[::2]):
            upper_efis.append(xr.open_dataset(os.path.join(args.data_path, f"efi_{var}{str(out_upper_level[i])}.nc")))
            
    if args.full:
        T = 10
        print("using full ensemble dataset")
    else:
        T = 2
        
    dataset_train = utils_data.Dataset_Postprocessing(data_path=args.data_path, start_date=args.train_start_date, end_date=args.train_end_date, full=args.full, surface=surf_vars, upper=upper_vars, levels=level, target_surface=out_surf_vars[::2], target_upper=out_upper_vars[::2], target_level=out_upper_level)
    dataset_val = utils_data.Dataset_Postprocessing(data_path=args.data_path, start_date=args.val_start_date, end_date=args.val_end_date, val=True, full=args.full, surface=surf_vars, upper=upper_vars, levels=level, target_surface=out_surf_vars[::2], target_upper=out_upper_vars[::2], target_level=out_upper_level)
    dataset_test = None
    sampler_test = None
    
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    
    print("Sampler_train = %s" % str(sampler_train))
    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        # sampler_test = torch.utils.data.DistributedSampler(
        #     dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None


    collate_func = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=collate_func,
    )

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size= args.val_batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    else:
        data_loader_val = None

    if dataset_test is not None:
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, sampler=sampler_test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    else:
        data_loader_test = None
    
    in_channels = (len(surf_vars)+len(upper_vars)*len(level)) * T
    out_channels = (len(out_surf_vars) + len(out_upper_vars)) * 2

    use_ours = True
    model = Aurora(use_ours=True, autocast=True,
                    surf_vars = surf_vars,
                    atmos_vars = upper_vars,
                    timestep = timedelta(hours=0),     
                    out_surf_vars = out_surf_vars,
                    out_atmos_vars = out_upper_vars,
                    task = "postprocess",
                    max_history_size = T,
                    ours_prompt_length=5
                )



    model.configure_activation_checkpointing()
    print("load pretrain model")
    model.load_checkpoint_local("../aurora-0.25-pretrained.ckpt", strict=False)
        
    model.to(device)
    model.train()

    model_ema = None

    model_without_ddp = model

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    args.lr = args.lr * total_batch_size / 256
    args.min_lr = args.min_lr * total_batch_size / 256
    args.warmup_lr = args.warmup_lr * total_batch_size / 256
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

    num_layers = model_without_ddp.get_num_layers()
    if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None


    skip_weight_decay_list = model.no_weight_decay()

    if args.distributed:

        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    optimizer = create_optimizer(
        args, model_without_ddp, skip_list=skip_weight_decay_list,
        get_num_layer=assigner.get_layer_id if assigner is not None else None, 
        get_layer_scale=assigner.get_scale if assigner is not None else None)
    loss_scaler = NativeScaler()

    print("Use step level LR scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))   

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

    if args.eval:
        print("Start evaluating")
        test_stats = validation_one_epoch_postprocess(data_loader_val, model, device,lat = lat, lon = lon, level = level, criterion1 = criterion1, criterion2= criterion2,
                                                        static_vars = static_vars, surf_vars=surf_vars, upper_vars=upper_vars, model_name=args.model, surface_efis = surface_efis,
                                                        upper_efis = upper_efis, out_surf_vars = out_surf_vars[::2], out_upper_vars = out_upper_vars[::2], out_upper_level = out_upper_level,)
        test_log = {**{f'val_{k}': v for k, v in test_stats.items()}}
        
        copy_log = " &"+" &".join([str(round(v, 3)) for k, v in test_stats.items() if k!="valid_loss"])
        
        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps("Eval:") + "\n")
                f.write(json.dumps(test_log) + "\n")
                f.write(json.dumps(copy_log) + "\n" + "\n")
        # torch.distributed.barrier()
        # if global_rank == 0:
        exit(0)
        
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    train_time_only = 0
    total_step = args.epochs * num_training_steps_per_epoch
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        train_start_time = time.time()
        train_stats = train_one_epoch_postprocess(
            model, data_loader_train, optimizer,
            device, epoch, loss_scaler, args.clip_grad, 
            log_writer=log_writer, start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq, criterion = criterion1,
            lat = lat, lon = lon, level = level, static_vars = static_vars, surf_vars=surf_vars, upper_vars=upper_vars, model_name=args.model,
            out_surf_vars = out_surf_vars[::2], out_upper_vars = out_upper_vars[::2], out_upper_level = out_upper_level,use_ours=use_ours, total_step=total_step
        )
        train_time_only += time.time() - train_start_time
        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)
                
        if data_loader_val is not None:
            test_stats = validation_one_epoch_postprocess(data_loader_val, model, device,lat = lat, lon = lon, level = level, criterion1 = criterion1, criterion2= criterion2,
                                                        static_vars = static_vars, surf_vars=surf_vars, upper_vars=upper_vars, model_name=args.model, surface_efis = surface_efis, 
                                                        upper_efis = upper_efis, out_surf_vars = out_surf_vars[::2], out_upper_vars = out_upper_vars[::2], out_upper_level = out_upper_level,)
        log_stats = {'epoch': epoch,
                    **{f'train_{k}': v for k, v in train_stats.items()}}
        test_log = {**{f'val_{k}': v for k, v in test_stats.items()}}
        
        copy_log = " &"+" &".join([str(round(v, 3)) for k, v in test_stats.items() if k!="valid_loss"])
        
        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
                f.write(json.dumps(test_log) + "\n")
                f.write(json.dumps(copy_log) + "\n" + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    
    train_time_str = str(datetime.timedelta(seconds=int(train_time_only)))
    print('Training time only {}'.format(train_time_str))
    print('Total time {}'.format(total_time_str))

    if args.output_dir and utils.is_main_process():
        if log_writer is not None:
            log_writer.flush()
        with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps('Training time only {}'.format(train_time_str) + "\n"))
            f.write(json.dumps('Total time {}'.format(total_time_str)) + "\n")

if __name__ == '__main__':
    # mp.set_start_method('spawn', force=True)
    opts, ds_init = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts, ds_init)
