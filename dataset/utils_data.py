import xarray as xr
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import sys
import os

import torch
import random
from torch.utils import data

import matplotlib.pyplot as plt


sfc_vars_1 = ['SSTK', 'TCW', 'TCWV', 'CP', 'MSL', 'TCC', 'U10M', 'V10M', 'T2M', 'TP', 'SKT']
sfc_vars_2 = ['sst', 'tcw', 'tcwv', 'cp', 'msl', 'tcc', 'u10', 'v10', 't2m', 'tp', 'skt']
pl_vars_1 = ["Z", "T", "Q", "W", "D", "U", "V"]
pl_vars_2 = ["z", "t", "q", "w", "d", "u", "v"]
var_map = {}
for var1, var2 in zip(sfc_vars_1+pl_vars_1, sfc_vars_2+pl_vars_2):
    var_map[var1] = var2 

class Aurora_CDF_Dataset_china(data.Dataset):
    
    """Dataset class for the era5 upper and surface variables."""

    def __init__(self,
                 nc_path='',
                 seed=1234,
                 startDate='2010',
                 endDate='2020',
                 freq='12h',
                 horizon = 12,
                 surface = ["2m_temperature","10m_u_component_of_wind","10m_v_component_of_wind", "mean_sea_level_pressure", "total_precipitation_6hr"], 
                 upper = ["temperature", "u_component_of_wind", "v_component_of_wind", "relative_humidity", "geopotential"],
                 level = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000],
                 h = 240,
                 w = 240,
                 degree = 0.25,
                 num_points = 5,
                 evaluate = False,
                 ):
        
        """Initialize."""
        self.nc_path = nc_path
        """
        To do
        if start and end is valid date, if the date can be found in the downloaded files, length >= 0

        """
        # Prepare the datetime objects for training, validation, and test

        self.freq = int(freq[:-1])
        self.surface_variables = surface
        self.upper_variables = upper
        self.levels = level
        self.horizon = horizon
        self.h = h 
        self.w = w
        self.degree = degree
        self.num_points = num_points

        self.keys = list(pd.date_range(start=startDate, end=endDate, freq=freq))[:-1]

        random.seed(seed)
    
            

    def __getitem__(self, index):
        """Return input frames, target frames, and its corresponding time steps."""
        key = self.keys[index]
        
        input_surfaces = []
        input_uppers = []

        time_points = []
        
        for p in range(self.num_points): 
            time_str = datetime.strftime(key, '%Y%m%d%H')
            time_points.append(key.timestamp())
            data = np.load(os.path.join(self.nc_path,f"{time_str}.npy")).astype(np.float32)

            input_surface_variables = data[:,0]
            input_upper_variables = data[:,1:]    

            input_surfaces.append(input_surface_variables[np.newaxis, ...])
            input_uppers.append(input_upper_variables[np.newaxis, ...])
            key = key + timedelta(hours=self.horizon)
        
        return np.concatenate(input_surfaces,axis=0), np.concatenate(input_uppers, axis=0), time_points    

    def __len__(self):
        return len(self.keys)

    def __repr__(self):
        return self.__class__.__name__
    
class Dataset_Postprocessing(data.Dataset):
    
    def __init__(self,
                 data_path='',
                 seed=1234,
                 start_date='1998-01-01',
                 end_date='2015-12-31',
                 val = False,
                 surface = ['SSTK', 'TCW', 'TCWV', 'CP', 'MSL', 'TCC', 'U10M', 'V10M', 'T2M', 'TP', 'SKT'], 
                 upper = ["Z", "T", "Q", "W", "D", "U", "V"],
                 levels = [500, 850],
                 target_surface= ["T2M",'U10M', 'V10M'],
                 target_upper = ["Z", "T"],
                 target_level = [500, 850],
                 H = 361,
                 W = 720,
                 full = False,
                 ):
        
        """Initialize."""
        self.data_path = data_path
        self.surface_variables = surface
        self.upper_variables = upper
        self.levels = levels
        self.target_sfc_variables = target_surface
        self.target_pl_variables = target_upper
        self.target_pl_level = target_level
        self.H = 361
        self.W = 720
        self.full = full
        
        time_range = slice(start_date, end_date)
        if self.full:
            self.ensemble_path = os.path.join(data_path, "ensemble")
            self.T = 10
            self.sfc_value_range = {"sst": (260., 305.), "tcw": (0., 60.), "tcwv": (0., 60.), "cp": (0., 0.04), "msl": (97000., 1.1e5), "tcc": (0., 1.0), "u10": (-13., 11.), "v10": (-10., 15.), "t2m":(218, 304), "tp": (0., 0.07), "skt": (210., 310.)}
            self.pl_value_range = [{"z": (48200, 58000), "t": (230, 269), "q": (0., 4e-3), "w": (-0.7, 1.4), "d": (-5e-5, 8e-5), "u": (-7., 27.), "v": (-7., 7.)},
                                   {"z": (10000, 15500), "t":(240, 299), "q": (0., 1.5e-2), "w": (-1.2, 1.8), "d": (-1.9e-4, 1.6e-4), "u": (-16., 17.5), "v": (-10., 16.)}]
        else:
            self.T = 2
        self.sfc_ens_mean_normalized = xr.open_dataset(os.path.join(data_path, "ENS10_sfc_mean_normalized.nc"), engine="h5netcdf").sel(time=time_range)
        self.sfc_ens_std_normalized = xr.open_dataset(os.path.join(data_path, "ENS10_sfc_std_normalized.nc"), engine="h5netcdf").sel(time=time_range)
        self.sfc_ens_mean = xr.open_dataset(os.path.join(data_path, "ENS10_sfc_mean.nc"), engine="h5netcdf").sel(time=time_range)
        self.sfc_ens_std = xr.open_dataset(os.path.join(data_path, "ENS10_sfc_std.nc"), engine="h5netcdf").sel(time=time_range)
        
        
        self.pl_ens_mean_normalized = [xr.open_dataset(os.path.join(data_path, f"ENS10_pl_mean_{str(l)}_normalized.nc"), engine="h5netcdf").sel(time=time_range) for l in levels]   
        self.pl_ens_std_normalized = [xr.open_dataset(os.path.join(data_path, f"ENS10_pl_std_{str(l)}_normalized.nc"), engine="h5netcdf").sel(time=time_range) for l in levels]
        self.pl_ens_mean = [xr.open_dataset(os.path.join(data_path, f"ENS10_pl_mean_{str(l)}.nc"), engine="h5netcdf").sel(time=time_range) for l in levels]
        self.pl_ens_std = [xr.open_dataset(os.path.join(data_path, f"ENS10_pl_std_{str(l)}.nc"), engine="h5netcdf").sel(time=time_range) for l in levels]
        
        self.era5 = xr.open_dataset(os.path.join(data_path, "ERA5.nc"), engine="h5netcdf").sel(time=time_range)
        if val:
            self.keys = self.sfc_ens_mean.drop_sel(time="2017-01-02").time.values
            # self.keys = [self.sfc_ens_mean.sel(time="2017-12-01").time.values]
        else:
            self.keys = self.sfc_ens_mean.time.values
        self.era5_sfc_scale = {}

        random.seed(seed)

    def __getitem__(self, index):
        """Return input frames, target frames, and its corresponding time steps."""
        time_points = []
        key = self.keys[index]
        time_points.append(key.item())
        
        sfc_inputs = np.zeros((self.T, len(self.surface_variables), self.H, self.W)).astype(np.float32)
        pl_inputs = np.zeros((self.T, len(self.upper_variables), len(self.levels), self.H, self.W)).astype(np.float32)
        
        ds_targets = self.era5.sel(time=key)
        time_str = (pd.to_datetime(key)-timedelta(days=2)).strftime("%Y%m%d")
        
        sfc_targets = np.zeros((len(self.target_sfc_variables), self.H, self.W)).astype(np.float32)
        sfc_scale = np.zeros((2, len(self.target_sfc_variables), self.H, self.W)).astype(np.float32)

        if self.target_sfc_variables:
            for i in range(len(self.target_sfc_variables)):
                sfc_targets[i] = ds_targets[self.target_sfc_variables[i]].values.astype(np.float32)
                sfc_scale[0,i] = self.sfc_ens_mean[self.target_sfc_variables[i]].sel(time=key).values.astype(np.float32)
                sfc_scale[1,i] = self.sfc_ens_std[self.target_sfc_variables[i]].sel(time=key).values.astype(np.float32)
                
            
        pl_targets = np.zeros((len(self.target_pl_variables), self.H, self.W)).astype(np.float32)
        pl_scale = np.zeros((2, len(self.target_pl_variables), self.H, self.W)).astype(np.float32)
        
        if self.target_pl_variables:
            for i in range(len(self.target_pl_variables)):
                pl_targets[i] = ds_targets[self.target_pl_variables[i]].values.astype(np.float32)[0]                
                pl_scale[0,i] = self.pl_ens_mean[self.levels.index(self.target_pl_level[i])][self.target_pl_variables[i]].sel(time=key,plev=self.target_pl_level[i]*1e2).values.astype(np.float32)
                pl_scale[1,i] = self.pl_ens_std[self.levels.index(self.target_pl_level[i])][self.target_pl_variables[i]].sel(time=key,plev=self.target_pl_level[i]*1e2).values.astype(np.float32)            

        if self.full:
            sfc_ds = xr.open_dataset(os.path.join(self.ensemble_path, f"output.sfc.{time_str}.grib"), backend_kwargs={"indexpath":""}).fillna(9999.0)
            pl_ds = xr.open_dataset(os.path.join(self.ensemble_path, f"output.pl.{time_str}.grib"), backend_kwargs={"indexpath":""}).sel(isobaricInhPa=self.levels).fillna(9999.0)
            for i, var in enumerate(self.surface_variables):
                value = sfc_ds[var_map[var]].values.astype(np.float32)
                minval, maxval = self.sfc_value_range[var_map[var]]
                sfc_inputs[:,i] = (value - minval) / (maxval - minval)
            for i, var in enumerate(self.upper_variables):
                value = pl_ds[var_map[var]].values.astype(np.float32)
                for j in range(len(self.levels)):
                    minval, maxval = self.pl_value_range[j][var_map[var]]
                    pl_inputs[:,i,j] = (value[:,j] - minval) / (maxval - minval)
        else:
            for i, var in enumerate(self.surface_variables):
                sfc_inputs[1,i] = self.sfc_ens_mean_normalized[var].sel(time=key).values.astype(np.float32)
                sfc_inputs[0,i] = self.sfc_ens_std_normalized[var].sel(time=key).values.astype(np.float32)

            for i, var in enumerate(self.upper_variables):
                for j, l in enumerate(self.levels):
                    pl_inputs[1,i,j] = self.pl_ens_mean_normalized[j][var].sel(time=key).values[0].astype(np.float32)
                    pl_inputs[0,i,j] = self.pl_ens_std_normalized[j][var].sel(time=key).values[0].astype(np.float32)



        return sfc_inputs, pl_inputs, sfc_scale[:,:,:-1], sfc_targets[:,:-1], pl_scale[:,:,:-1],pl_targets[:,:-1], time_points

    def __len__(self):
        return len(self.keys)

    def __repr__(self):
        return self.__class__.__name__    
    
class Aurora_downscale(data.Dataset):
    
    """Dataset class for the era5 upper and surface variables."""

    def __init__(self,
                 nc_path='',
                 seed=1234,
                 startDate='1999',
                 endDate='2020',
                 freq='6h',
                 surface_prefix = ["2m_temperature","10m_u_component_of_wind","10m_v_component_of_wind"], 
                 upper_prefix = ["temperature", "u_component_of_wind", "v_component_of_wind", "specific_humidity", "geopotential"],
                 surface = ["t2m","u10","v10"], 
                 upper = ["t", "u", "v", "q", "z"],
                 levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000],
                 num_points = 2,
                 degree = ["5.625", "1.40625"]
                 ):
        
        """Initialize."""
        self.nc_path = nc_path
        """
        To do
        if start and end is valid date, if the date can be found in the downloaded files, length >= 0

        """
        # Prepare the datetime objects for training, validation, and test

        self.freq = int(freq[:-1])
        self.surface_variables = surface
        self.upper_variables = upper
        self.levels = levels

        self.num_points = num_points

        
        self.keys = list(pd.date_range(start=startDate, end=endDate, freq=freq))[:-2]
        
        self.lr_datasets = {}
        self.hr_datasets = {}
        
        for i, var in enumerate(self.surface_variables):
            for key in list(pd.date_range(start=startDate[:4], end=endDate[:4], freq="YE")):
                year = datetime.strftime(key, '%Y')
                self.lr_datasets[f"{var}_{year}"] = xr.open_dataset(os.path.join(self.nc_path, degree[0], surface_prefix[i], f'{surface_prefix[i]}_{year[:4]}_{degree[0]}deg.nc')) 
                self.hr_datasets[f"{var}_{year}"] = xr.open_dataset(os.path.join(self.nc_path, degree[1], surface_prefix[i], f'{surface_prefix[i]}_{year[:4]}_{degree[1]}deg.nc'))     

        for i, var in enumerate(self.upper_variables):
            for key in list(pd.date_range(start=startDate[:4], end=endDate[:4], freq="YE")):
                year = datetime.strftime(key, '%Y')
                self.lr_datasets[f"{var}_{year}"] = xr.open_dataset(os.path.join(self.nc_path, degree[0], upper_prefix[i], f'{upper_prefix[i]}_{year[:4]}_{degree[0]}deg.nc')) 
                self.hr_datasets[f"{var}_{year}"] = xr.open_dataset(os.path.join(self.nc_path, degree[1], upper_prefix[i], f'{upper_prefix[i]}_{year[:4]}_{degree[1]}deg.nc'))     
                
        random.seed(seed)

    def __getitem__(self, index):
        """Return input frames, target frames, and its corresponding time steps."""
        
        key = self.keys[index]
        input_surfaces = []
        input_uppers = []
        target_surfaces = []
        target_uppers = []
        time_points = []
        
        for _ in range(self.num_points): 
            time_points.append(key.timestamp())
            year = datetime.strftime(key, '%Y')
            input_surface_variables = []
            input_upper_variables = []            
            
            for var in self.surface_variables:
                input_surface_dataset = self.lr_datasets[f"{var}_{year}"].sel(time=key)
                value = (input_surface_dataset[var].values.astype(np.float32))[::-1, :]
                input_surface_variables.append(value[np.newaxis, ...])
                    
            for var in self.upper_variables:
                input_upper_dataset = self.lr_datasets[f"{var}_{year}"].sel(time=key)
                value = (input_upper_dataset[var].values.astype(np.float32))[:,::-1, :]
                input_upper_variables.append(value[np.newaxis, ...])

            input_surfaces.append(np.concatenate(input_surface_variables, axis=0)[np.newaxis, ...])
            input_uppers.append(np.concatenate(input_upper_variables, axis=0)[np.newaxis, ...])

            key = key + timedelta(hours=6)

        key = key - timedelta(hours=6)
        year = datetime.strftime(key, '%Y')
        for var in self.surface_variables:
            target_surface_dataset = self.hr_datasets[f"{var}_{year}"].sel(time=key)
            value = (target_surface_dataset[var].values.astype(np.float32))[::-1, :]
            target_surfaces.append(value[np.newaxis, ...])
            
        for var in self.upper_variables:
            target_upper_dataset = self.hr_datasets[f"{var}_{year}"].sel(time=key)
            value = (target_upper_dataset[var].values.astype(np.float32))[:,::-1, :]
            target_uppers.append(value[np.newaxis, ...])

        return np.concatenate(input_surfaces,axis=0), np.concatenate(input_uppers, axis=0), np.concatenate(target_surfaces,axis=0), np.concatenate(target_uppers, axis=0), time_points    

    def __len__(self):
        return len(self.keys)

    def __repr__(self):
        return self.__class__.__name__



