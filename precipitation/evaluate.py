import xarray as xr
import pandas as pd
import numpy as np
import os


def evaluate_pre(forecast):

  def _assert_increasing(x: np.ndarray):
    if not (np.diff(x) > 0).all():
      raise ValueError(f"array is not increasing: {x}")
  def _latitude_cell_bounds(x: np.ndarray) -> np.ndarray:
    pi_over_2 = np.array([np.pi / 2], dtype=x.dtype)
    return np.concatenate([-pi_over_2, (x[:-1] + x[1:]) / 2, pi_over_2])
  def _cell_area_from_latitude(points: np.ndarray) -> np.ndarray:
    """Calculate the area overlap as a function of latitude."""
    bounds = _latitude_cell_bounds(points)
    _assert_increasing(bounds)
    upper = bounds[1:]
    lower = bounds[:-1]
    # normalized cell area: integral from lower to upper of cos(latitude)
    return np.sin(upper) - np.sin(lower)

  def get_lat_weights_ds(ds: xr.Dataset) -> xr.DataArray:
    """Computes latitude/area weights from latitude coordinate of dataset."""
    weights = _cell_area_from_latitude(np.deg2rad(ds.latitude.data))
    weights /= np.mean(weights)
    weights = ds.latitude.copy(data=weights)
    return weights

  def convert_precip_to_seeps_cat(ds):
      """Helper function for SEEPS computation. Converts values to categories."""
      # Convert to SI units [meters]
      dry_threshold = dry_threshold_mm / 1000.0
      da = ds[precip_name]
      wet_threshold_for_valid_time = wet_threshold.sel(
          dayofyear=da.valid_time.dt.dayofyear, hour=da.valid_time.dt.hour
      ).load()

      dry = da < dry_threshold
      light = np.logical_and(
          da > dry_threshold, da < wet_threshold_for_valid_time
      )
      heavy = da >= wet_threshold_for_valid_time
      result = xr.concat(
          [dry, light, heavy],
          dim=xr.DataArray(["dry", "light", "heavy"], dims=["seeps_cat"]),
      )
      # Convert NaNs back to NaNs
      result = result.astype("int").where(da.notnull())
      return result

  def get_lat_weights(lat):
    w_lat = np.cos(np.deg2rad(lat))
    return w_lat / w_lat.mean()

  def apply_time_conventions(
      forecast: xr.Dataset, by_init: bool
  ) -> xr.Dataset:
    """Apply WeatherBench2 time name conventions onto a forecast dataset."""
    forecast = forecast.copy()
    if 'prediction_timedelta' in forecast.coords:
      forecast = forecast.rename({'prediction_timedelta': 'lead_time'})
      if by_init:
        # Need to rename time dimension because different from time dimension in
        # truth dataset
        forecast = forecast.rename({'time': 'init_time'})
        valid_time = forecast.init_time + forecast.lead_time
        forecast.coords['valid_time'] = valid_time
        assert not hasattr(
            forecast, 'time'
        ), f'Forecast should not have time dimension at this point: {forecast}'
      else:
        init_time = forecast.time - forecast.lead_time
        forecast.coords['init_time'] = init_time
    return forecast

  root = "../aux_data"

  obs_path = os.path.join(root, "era5_tp_6hr_china.nc")
  climatology_path = os.path.join(root, "climatology")

  startDate = "2020-01-01"
  endDate = "2021-01-01"
  precip_name = 'total_precipitation_6hr'
  latitude_slice = slice(58.5, -1.25, -1) 
  longitude_slice = slice(74,133.75)

  obs = xr.open_dataset(obs_path)
  climatology = xr.open_dataset(os.path.join(climatology_path, "total_precipitation_6hr.nc"))

  lat = np.linspace(-90, 90, 721)
  latitude = xr.DataArray(
      data=lat,
      dims=["latitude"],
      coords={
          "latitude": lat,
      },
  )
  latitude = xr.Dataset(
      {
          "latitude": latitude
      },
      attrs={
          "description": "latitude",
      }
  )
  weights = get_lat_weights_ds(latitude).sel(latitude=latitude_slice)
  weight = weights.values[np.newaxis, :, np.newaxis]
  forecast = apply_time_conventions(forecast, by_init=True)
  p1 = xr.open_dataset(os.path.join(climatology_path, "total_precipitation_6hr_seeps_dry_fraction.nc"))[f"{precip_name}_seeps_dry_fraction"].mean(("hour", "dayofyear"))
  wet_threshold = xr.open_dataset(os.path.join(climatology_path, "total_precipitation_6hr_seeps_threshold.nc"))[f"{precip_name}_seeps_threshold"]
  scoring_matrix = [
          [xr.zeros_like(p1), 1 / (1 - p1), 4 / (1 - p1)],
          [1 / p1, xr.zeros_like(p1), 3 / (1 - p1)],
          [
              1 / p1 + 3 / (2 + p1),
              3 / (2 + p1),
              xr.zeros_like(p1),
          ],
      ]

  das = []
  for mat in scoring_matrix:
    das.append(xr.concat(mat, dim=xr.DataArray(["dry", "light", "heavy"], dims=["truth_cat"])))
  das = xr.concat(das, dim=xr.DataArray(["dry", "light", "heavy"], dims=["forecast_cat"]))

  max_p1 = 0.85
  min_p1 = 0.1
  mask = (p1.values<max_p1) & (p1.values>min_p1)
  dry_threshold_mm = 0.1


  forecast_time = forecast.isel(lead_time=0) 

  obs_time = obs.sel(time=forecast_time.valid_time)

  time_selection = dict(dayofyear=forecast_time["valid_time"].dt.dayofyear)
  time_selection["hour"] = forecast_time["valid_time"].dt.hour
  climatology_time = climatology.sel(time_selection).total_precipitation_6hr

  error = (forecast_time.total_precipitation_6hr - obs_time.total_precipitation_6hr)**2
  rmse = error.weighted(weights).mean(skipna=True).values
  anom_forecast = forecast_time.total_precipitation_6hr.values - climatology_time.values
  anom_obs = obs_time.total_precipitation_6hr.values - climatology_time.values

  weighted_acc = np.sum((anom_forecast * anom_obs) * weight, (-2,-1))/ np.sqrt(np.sum((weight * anom_forecast**2), (-2,-1)) * np.sum((weight * anom_obs**2), (-2,-1)))


  forecast_cat = convert_precip_to_seeps_cat(forecast_time)
  truth_cat = convert_precip_to_seeps_cat(obs_time)

  out = (
        forecast_cat.rename({"seeps_cat": "forecast_cat"})
        * truth_cat.rename({"seeps_cat": "truth_cat"})
    )

  scoring_matrix_time = 0.5 * das
  seeps = xr.dot(out, scoring_matrix_time, dims=("forecast_cat", "truth_cat"))
  seeps = seeps.where(p1 < max_p1, np.nan)
  seeps = seeps.where(p1 > min_p1, np.nan)
  seeps = seeps.weighted(weights).mean(skipna=True).values

  
  return seeps, weighted_acc.mean(), rmse**0.5