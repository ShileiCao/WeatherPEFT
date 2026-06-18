# [ICLR2026] Task-Adaptive Parameter-Efficient Fine-Tuning for Weather Foundation Models

[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/) 

<p align="center">
  <img src="method.png"/ width="1200">
</p>

## Contents
1. [Installation Instructions](#installation-instructions)
2. [Dataset Preparation](#dataset-preparation)
3. [Execution Instructions](#execution-instructions)
4. [Acknowledgement](#acknowledgement)

## Installation Instructions
- We use Python 3.10, PyTorch 2.5.1 (CUDA 11.8 build).

```angular2

#Make sure that the WeatherPEFT is placed in the current user directory for dataset register.
cd WeatherPEFT

conda create -n WeatherPEFT python=3.10
conda activate WeatherPEFT

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
conda install -c conda-forge xarray==2025.1.2 numpy==2.1.3 pandas==2.2.3 dask netCDF4 bottleneck

pip install -r requirements.txt
```

## Dataset Preparation
Download the [constant data](https://drive.google.com/file/d/1kpUMObNisJHf-karZrMnJ4Mkjlam0jDk/view?usp=sharing), including the static variable for Aurora (Land-sea mask, Soil type, Surface-level geopotentia), precomputed statistics for normalization (mean and std), and ERA5-CH precipitation ground-truth and Climatology data for precipitation evaluation. Put them in WeatherPEFT/aux_data

1. **Downscale.** In this experiment, we downscale 5.625° ERA5 data to 1.40625° ERA5 data both at a global scale and 6-hour intervals.
The pre-processed 5.625° and 1.40625° ERA5 data data can be downloaded in [WeatherBench](https://github.com/pangeo-data/WeatherBench).
Please follow dataset structure below:s
```
    |--WeatherPEFT
        |--datasets
            |--5.625
                |--2m_temperature
                |--10m_u_component_of_wind
                |--10m_v_component_of_wind 
                |--temperature
                |--u_component_of_wind
                |--v_component_of_wind
                |--specific_humidity
                |--geopotential
            |--1.40625
                |--2m_temperature
                |--10m_u_component_of_wind
                |--10m_v_component_of_wind 
                |--temperature
                |--u_component_of_wind
                |--v_component_of_wind
                |--specific_humidity
                |--geopotential
            ...
```

2. **Ensemble Weather Forecast Post-Processing.** This experiment employs the [ENS-10](https://github.com/spcl/ens10) benchmark for global ensemble forecast post-processing, which pairs 10-member ensemble prediction (48-hour lead time) from the ECMWF Integrated Forecasting System (IFS) with ERA5 reanalysis targets at 0.5° resolution. Download the ENS10 and ERA5 [data](http://spclstorage.inf.ethz.ch/projects/deep-weather/) and follow the instructions provided by [ENS-10](https://github.com/spcl/ens10). Excute data process script in [ENS-10](https://github.com/spcl/ens10) (refer to ens10/baselines/utils and ens10/EFI) to process ERA5 and ens data and extract the EFI for additional variable U10 and V10.
Please follow dataset structure below:
```
    |--WeatherPEFT
        |--datasets
            |--post-process
                |--process
                    |--efi_T2M.nc
                    |--efi_T850.nc
                    |--efi_U10M.nc
                    |--efi_V10M.nc
                    |--efi_Z500.nc
                    |--ENS10_sfc_mean_normalized.nc
                    |--ENS10_sfc_std_normalized.nc
                    |--ENS10_sfc_mean.nc
                    |--ENS10_sfc_std.nc
                    |--ENS10_pl_mean_500_normalized.nc
                    |--ENS10_pl_mean_500.nc
                    |--ENS10_pl_std_850_normalized.nc
                    |--ENS10_pl_std_850.nc
                    |--ERA5.nc
            ...
```

3. **Regional Precipitation Forecasting.** In this experiment, we evaluate WeatherPEFT for regional precipitation forecasting, where the task is to predict the future six-hour accumulation of total precipitation (TP-6hr) based on the current weather conditions in the target region. The data is based on 12-hour interval 0.25° ERA5 data with 2010–2019 serving as the training set and 2020 as the test set, which can be downloaded from [ECMWF](https://cds.climate.copernicus.eu/datasets). Process the hourly precipitation data into 6-hour cumulative precipitation data (refer to WeatherPEFT/dataset/process_tp_6hr.py) or directly download all this data from [Weatherbench2](https://github.com/google-research/weatherbench2). Process the data to package all the variable into one at each time point over the specific region (refer to WeatherPEFT/dataset/process.py). Please follow dataset structure below:

```
    |--WeatherPEFT
        |--datasets
            |--0.25
                |--china_organized
                    |--2010010100.npy
                    |--2010010112.npy
                    |--2010010200.npy
                    |--2010010212.npy
                    ...
                    |--2020123112.npy
            ...
```

## Execution Instructions

### Pretraining Checkponit

- Download the pretraind-trained [model weights](https://huggingface.co/microsoft/aurora/blob/main/aurora-0.25-pretrained.ckpt) of Aurora and and put them in WeatherPEFT/aurora-0.25-pretrained.ckpt

### Fine-tuning

We provide three executable script in three folder (downscaling, postprocessing, and precipitation) for three corresponding downstream tasks.

For examble, using
```angular2
cd downscaling
sh script_run_downscaling.sh
```
for downscaling task.

## Acknowledgement

We thank the developers and authors of [Aurora](https://github.com/microsoft/aurora) for releasing their pre-trained model and [VideoMAE](https://github.com/MCG-NJU/VideoMAE) for releasing their helpful training framework.

## Citation

If you use WeatherPEFT in your research or wish to refer to the baseline results, please use the following BibTeX entry.

```BibTeX
@inproceedings{
cao2026taskadaptive,
title={Task-Adaptive Parameter-Efficient Fine-Tuning for Weather Foundation Models},
author={Shilei Cao and Hehai Lin and Jiashun Cheng and Yang Liu and Guowen Li and Xuehe Wang and Juepeng Zheng and Haoyuan Liang and Meng Jin and Chengwei Qin and Hong Cheng and Haohuan Fu},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=eFExhM3tKr}
}
```