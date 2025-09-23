import os
import pandas as pd
import xarray as xr
import numpy as np
from datetime import datetime, timedelta
from multiprocessing import Pool
import multiprocessing as mp
from tqdm import tqdm

# 初始化参数
nc_path='../datasets/0.25'
startDate='2021'
endDate='2022'
freq='6h'
horizon = 6
surface_variables = ["t2m","u10","v10", "msl", "tp"]
upper_variables = ["t", "u", "v", "r", "z",]
levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
surface_prefix = ["t2m.era5.", "u10.era5.", "v10.era5.", "era5.mslp-slhf-sshf.",""]
upper_prefix = ["era5.temperature.", "era5.u_component_of_wind.", "era5.v_component_of_wind.", "era5.relative_humidity.", "era5.geopotential."]
output_dir = '../datasets/0.25/china_organized'  # 定义输出目录

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 生成需要处理的时间列表
keys = list(pd.date_range(start=startDate, end=endDate, freq="1d"))[:-1]

def process_time(key):
    try:
        time_str = key.strftime('%Y%m%d%H')
        surface_datasets = [xr.open_dataset(os.path.join(nc_path, "surface", var, f'{surface_prefix[i]}{time_str[:8]}.nc')) for i, var in enumerate(surface_variables[:-1])]
        upper_datasets = [xr.open_dataset(os.path.join(nc_path, "upper", var, f'{upper_prefix[i]}{time_str[:8]}.nc')).sel(level=levels) for i, var in enumerate(upper_variables)] 
        for _ in range(24//horizon):
            # 保存为NetCDF文件
            time_str = key.strftime('%Y%m%d%H')
            output_file = os.path.join(output_dir, time_str + '.npy')
            if os.path.exists(output_file):
                key += timedelta(hours=horizon)
                continue
            input_surface_variables = []
            for i, var in enumerate(surface_variables):
                if var == "tp":
                    data = np.load(os.path.join(nc_path, "tp_6hr", f"{time_str[:10]}.npy")).astype(np.float32)[126:366,296:536]
                    input_surface_variables.append(data[np.newaxis, ...])
                else:
                    ds = surface_datasets[i]
                    data = ds[var].sel(time=key).values.astype(np.float32)[126:366,296:536]
                    input_surface_variables.append(data[np.newaxis, ...])
            # 将地表变量数据堆叠
            surface_data = np.stack(input_surface_variables, axis=0)

            # 读取高空变量
            input_upper_variables = []
            for i, var in enumerate(upper_variables):
                ds = upper_datasets[i]
                data = ds[var].sel(time=key).values.astype(np.float32)[:, 126:366,296:536]
                input_upper_variables.append(data)
            # 将高空变量数据堆叠
            upper_data = np.stack(input_upper_variables, axis=0)

            all_data = np.concatenate((surface_data, upper_data), 1)
            np.save(output_file, all_data)
            key += timedelta(hours=horizon) 

    except Exception as e:
        print(f"处理时间 {key} 时出错: {e}")

if __name__ == '__main__':
    num_processes = 111  # 您拥有的CPU数量
    with Pool(num_processes) as pool:
        pool.map(process_time, keys)


