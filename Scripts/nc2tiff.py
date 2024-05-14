# @Author   : ChaoQiezi
# @Time     : 2024/5/7  17:13
# @FileName : nc2tiff.py
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to 提取
"""

import glob
import os.path
import numpy as np
import netCDF4 as nc  # 读取NC为文件
from utils.utils import data_glt




# 准备
nc_dir = r'I:\InnovationAndEntrepreneurshipOfCollegeStudents\PollutionFromDiesel\Data\GRIDS\D02'  # 修改此处路径
geo_path = r'I:\InnovationAndEntrepreneurshipOfCollegeStudents\PollutionFromDiesel\Data\GRIDS\D02\GRIDCRO2D'
out_dir = r'I:\InnovationAndEntrepreneurshipOfCollegeStudents\PollutionFromDiesel\Data\vars_tiff'
mask_path = r'H:\Datasets\BASIC\china_admin_city\ChengDu.shp'
if not os.path.exists(out_dir): os.makedirs(out_dir)
ds_names = ['GSUM_CO', 'GSUM_PM25', 'GSUM_VOCs', 'GSUM_NOx']

# 获取经纬度信息
with nc.Dataset(geo_path, 'r') as f:
    lon, lat = np.asarray(f['LON'][:].squeeze()), np.asarray(f['LAT'][:].squeeze())
    # 获取

# 检索nc文件
nc_paths = glob.glob(os.path.join(nc_dir, 'NCEMIS_*.nc'))
# 迭代nc文件并提取所需数据集输出为tiff文件
for nc_path in nc_paths:
    nc_name = os.path.splitext(os.path.basename(nc_path))[0]
    with nc.Dataset(nc_path, 'r') as f:
        for ds_name in ds_names:
            ds = np.asarray(f[ds_name][:])
            out_path = os.path.join(out_dir, '{}_{}.tiff'.format(nc_name, ds_name))
            data_glt(out_path, ds, lon, lat, 0.02, windows_size=9, mask_path=mask_path)
            print('processing: {}-{}'.format(nc_name, ds_name))
print('\n程序运行完毕, 从{}查看相关tiff文件'.format(out_dir))
