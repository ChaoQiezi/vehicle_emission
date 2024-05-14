# @Author   : ChaoQiezi
# @Time     : 2024/5/7  20:31
# @FileName : dead_code.py
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to ...
"""

import glob
import os.path
import pytz
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime
from osgeo import gdal, osr
import pandas as pd
import netCDF4 as nc
import numpy as np
from utils.utils import data_glt


# 准备
# nc_path = r'I:\InnovationAndEntrepreneurshipOfCollegeStudents\PollutionFromDiesel\Data\GRIDS\D02\NCEMIS_RAWOBD.nc'
in_dir = r'I:\InnovationAndEntrepreneurshipOfCollegeStudents\PollutionFromDiesel\Data\GRIDS\D02'
geo_path = r'I:\InnovationAndEntrepreneurshipOfCollegeStudents\PollutionFromDiesel\Data\GRIDS\D02\GRIDCRO2D'
out_dir = r'I:\InnovationAndEntrepreneurshipOfCollegeStudents\PollutionFromDiesel\Data\time_series'
start_date = datetime(2022, 9, 30, 8)  # 原是2022/9/30-0(UTC), 现为北京时间
end_date = datetime(2022, 11, 1, 7)
time_series = pd.date_range(start_date, end_date, freq='H')
no_data_value = -9999
# time_series = pd.date_range(start_date, end_date, freq='H', tz='UTC')  # 默认UTC
# time_series = time_series.tz_convert('Asia/Shanghai')  # UTC转北京时间

# 获取经纬度信息
with nc.Dataset(geo_path, 'r') as f:
    lon, lat = np.asarray(f['LON'][:].squeeze()), np.asarray(f['LAT'][:].squeeze())
nc_paths = glob.glob(os.path.join(in_dir, 'NCEMIS_*.nc'))
for nc_path in nc_paths:
    correction_name = os.path.basename(nc_path).split('_')[-1].split('.')[0]
    with nc.Dataset(nc_path) as f:
        CO = f['CO'][:].squeeze().filled(np.nan)
        NOx = f['NOx'][:].squeeze().filled(np.nan)
        PM2p5 = f['PM2p5'][:].squeeze().filled(np.nan)
        VOC = f['VOC'][:].squeeze().filled(np.nan)
    time_len, rows, cols = CO.shape
    pollutants = {
        'CO': CO,
        'NOx': NOx,
        'PM2p5': PM2p5,
        'VOC': VOC
    }

    df = []
    for pollutant_name, pollutant in pollutants.items():
        # 按日求和
        da = xr.DataArray(pollutant, coords=[time_series, np.arange(rows), np.arange(cols)], dims=['time', 'x', 'y'])
        da = da.sel(time=da['time'].dt.month.isin(10))
        da_sum_daily = da.groupby('time.date').sum()
        da_sum_daily = da_sum_daily.sum(dim=['x', 'y'])
        df.append(da_sum_daily.to_dataframe(name=pollutant_name))

        # 按白天黑夜求和
        da = xr.DataArray(
            data=pollutant,
            dims=['time', 'y', 'x'],
            coords=dict(
                x=('x', np.arange(cols)),
                y=('y', np.arange(rows)),
                time=('time', time_series),
                D_N=('time', time_series.map(lambda x: 'Day' if (x.hour >=8) & (x.hour <= 20) else 'Night'))
            )
        )
        da = da.sel(time=da['time'].dt.month.isin(10))  # 选择10月份数据集
        da_sum_day_night = da.groupby('D_N').sum()
        # 输出
        for group_name, group_data in da_sum_day_night.groupby('D_N'):
            out_name = '{}_{}_{}.tiff'.format(pollutant_name, group_name, correction_name)
            out_path = os.path.join(out_dir, 'TIFFs', out_name)
            data_glt(out_path, [group_data.fillna(no_data_value)], lon, lat, 0.02)

        # # 按24小时分小时求和
        if correction_name == 'GFXOBD':
            out_gfx_dir = os.path.join(out_dir, 'TIFFs', correction_name)
            if not os.path.exists(out_gfx_dir): os.makedirs(out_gfx_dir)
            da_mean_hourly = da.groupby('time.hour').mean().fillna(no_data_value)
            # 输出tiff文件
            for da_mean in da_mean_hourly:
                out_name = '{}_{:02}_{}_hourly.tiff'.format(pollutant_name, da_mean.hour, correction_name)
                out_path = os.path.join(out_gfx_dir, out_name)
                data_glt(out_path, [da_mean], lon, lat, 0.02, windows_size=9)

        print('已处理: {}-{}'.format(pollutant_name, correction_name))

    df = pd.concat(df, axis=1)
    df.to_csv(os.path.join(out_dir, '{}_sum_daily.csv'.format(correction_name)))
print('程序结束')