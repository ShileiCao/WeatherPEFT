import xarray as xr
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import torch
import random
from torch.utils import data
import os

class precipitation_dataset(data.Dataset):
    
    """Dataset class for the era5 upper and surface variables."""

    def __init__(self,
                 data_path='/share/home/liguowen/ClimaX/data/ERA5/0.25deg/daily',
                 seed=1234,
                 years=["2015",],
                 input_start_date="0201",
                 input_end_date="0301",
                 target_start_date=["0601", "0701", "0801"],
                 target_end_date= ["0630", "0731", "0831"],
                 freq='1d',
                 surface = ["tp",], 
                 surface_prefix = ["era5.total_precipitation",],
                 upper = [],
                 upper_prefix = [],
                 level = [],
                 target = "tp",
                 ):

        self.data_path = data_path
        self.surface_variables = surface
        self.upper_variables = upper
        self.levels = level
        self.years = years
        self.input_start_date=input_start_date
        self.input_end_date=input_end_date
        self.target_start_date=target_start_date
        self.target_end_date=target_end_date
        self.target = target


        self.surface_datasets = [{} for _ in range(len(years))]
        self.upper_datasets = [{} for _ in range(len(years))]
        
        for i, year in enumerate(years):
            for j, var in enumerate(surface):
                self.surface_datasets[i][var] = xr.open_dataset(os.path.join(data_path, var, f'{surface_prefix[j]}.{year}.nc'))
            for j, var in enumerate(upper):
                self.upper_datasets[i][var] = xr.open_dataset(os.path.join(data_path, var, f'{upper_prefix[j]}.{year}.nc'))

    
    
    def __getitem__(self, index):

        year = self.years[index]
        input_surface_variables = []
        target_variables = []
        for i, var in enumerate(self.surface_variables):
            input_surface_dataset = self.surface_datasets[index][var].sel(valid_time=slice(year+self.input_start_date, year+self.input_end_date))
            value = (input_surface_dataset[var].values.astype(np.float32))
            input_surface_variables.append(value)
        
        for i in range(len(self.target_start_date)):
            target_dataset = self.surface_datasets[index][self.target].sel(valid_time=slice(year+self.target_start_date[i], year+self.target_end_date[i]))
            value = (target_dataset[self.target].values.astype(np.float32))
            target_variables.append(value[np.newaxis, ...])

        return np.stack(input_surface_variables,axis=0), target_variables[0], target_variables[1], target_variables[2]
   

    def __len__(self):
        return len(self.years)

    def __repr__(self):
        return self.__class__.__name__
    
if __name__ == "__main__":
    
    years=["2015",]
    input_start_date="0201"
    input_end_date="0301"
    target_start_date=["0601", "0701", "0801"]
    target_end_date= ["0630", "0731", "0831"]
    surface = ["tp",]
    surface_prefix = ["era5.total_precipitation",]
    target = "tp"
    
    dataset = precipitation_dataset(years=years,input_start_date=input_start_date,input_end_date=input_end_date,
                                    target_start_date=target_start_date, target_end_date=target_end_date,
                                    surface = surface, target=target)
    
    surface, target_0, target_1, target_2 = dataset[0]
    
    print(surface.shape)
    print(target_0.shape)
    print(target_1.shape)
    print(target_2.shape)
    