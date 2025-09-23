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
startDate='2005'
endDate='2010'
freq='6h'
horizon = 6
output_dir = '../datasets/0.25/tp_6hr'  # 定义输出目录

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(output_dir_c):
    os.makedirs(output_dir_c)


# 生成需要处理的时间列表
keys = list(pd.date_range(start=startDate, end=endDate, freq="1d"))[:-1]


def process_time(key):

    time_str = key.strftime('%Y%m%d%H')
    tp_dataset = xr.open_dataset(os.path.join(nc_path, "surface", "total_precipitation", f'era5.total_precipitation.{time_str[0:8]}.nc'))
    time_str_last = (key-timedelta(hours=24)).strftime('%Y%m%d%H')
    tp_dataset_last = xr.open_dataset(os.path.join(nc_path, "surface", "total_precipitation", f'era5.total_precipitation.{time_str_last[0:8]}.nc'))
    for i in range(24//horizon):
        time_str = key.strftime('%Y%m%d%H')
        output_file = os.path.join(output_dir, f"{time_str}.npy")
        output_file_c = os.path.join(output_dir_c, f"{time_str}.npy")
        
        if os.path.exists(output_file):
            key += timedelta(hours=horizon)
            continue
        key_c = key
        cmorph_data=np.zeros((721,1440))
        for j in range(horizon):
            if i==0 and j>0:
                data_hour = tp_dataset_last["tp"].sel(time=key_c).values.astype(np.float32)
            else:
                data_hour = tp_dataset["tp"].sel(time=key_c).values.astype(np.float32)
            if np.isnan(data_hour).any():
                print(key_c)
            cmorph_data+=data_hour
            key_c -= timedelta(hours=1)
        key += timedelta(hours=6)
        np.save(output_file, cmorph_data)

if __name__ == '__main__':
    num_processes = 111  # 您拥有的CPU数量
    with Pool(num_processes) as pool:
        pool.map(process_time, keys)

