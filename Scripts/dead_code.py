# @Author   : ChaoQiezi
# @Time     : 2024/5/7  20:31
# @FileName : dead_code.py
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to ...
"""

from osgeo import gdal, osr
import netCDF4 as nc
import numpy as np

# 打开netCDF文件
dataset = nc.Dataset('path_to_your_file.nc')

# 读取变量
lat = dataset.variables['LAT'][:]
lon = dataset.variables['LON'][:]
var = dataset.variables['VAR'][:]

# 创建源和目标空间参考系统
src_srs = osr.SpatialReference()
src_srs.ImportFromEPSG(4326)  # WGS84, 使用经纬度坐标

dst_srs = osr.SpatialReference()
dst_srs.ImportFromProj4("+proj=lcc +lat_1=30 +lat_2=60 +lat_0=30.2 +lon_0=105 +R=6370000 +units=m +no_defs")

# 创建坐标转换
transform = osr.CoordinateTransformation(src_srs, dst_srs)

# 初始化转换后的投影坐标数组
proj_x = np.empty_like(lon)
proj_y = np.empty_like(lat)

# 为每个点应用坐标转换
for i in range(lat.shape[0]):
    for j in range(lon.shape[1]):
        x, y, z = transform.TransformPoint(lon[i, j], lat[i, j])
        proj_x[i, j] = x
        proj_y[i, j] = y

# 使用内存驱动器创建一个内存数据集
mem_driver = gdal.GetDriverByName('MEM')
dst_ds = mem_driver.Create('', lon.shape[1], lat.shape[0], 1, gdal.GDT_Float32)

# 将变量数据写入数据集
dst_ds.GetRasterBand(1).WriteArray(var)

# 设置仿射变换和投影
dst_ds.SetGeoTransform(gdal.GCPsToGeoTransform([
    gdal.GCP(proj_x[0, 0], proj_y[0, 0], 0, 0, 0),
    gdal.GCP(proj_x[0, -1], proj_y[0, -1], 0, lon.shape[1]-1, 0),
    gdal.GCP(proj_x[-1, 0], proj_y[-1, 0], 0, 0, lat.shape[0]-1)
]))
dst_ds.SetProjection(dst_srs.ExportToWkt())

# 输出为TIFF文件
tiff_driver = gdal.GetDriverByName('GTiff')
tiff_ds = tiff_driver.CreateCopy('output.tif', dst_ds)

# 清理
dst_ds = None
tiff_ds = None
dataset.close()



# 02

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from scipy.interpolate import griddata
import netCDF4 as nc

# 打开netCDF文件
dataset = nc.Dataset(r'I:\InnovationAndEntrepreneurshipOfCollegeStudents\PollutionFromDiesel\Data\GRIDS\D02\GRIDDOT2D.nc')

# 读取数据
# lat = dataset.variables['LAT'][:]
# lon = dataset.variables['LON'][:]
# var = dataset.variables['VAR'][:]
lat = dataset.variables['LATD'][:].squeeze()
lon = dataset.variables['LOND'][:].squeeze()
var = dataset.variables['MSFV2'][:].squeeze().filled(np.nan)

# 准备插值
# 将2D lon和lat数组转换为1D数组
points = np.column_stack((lon.ravel(), lat.ravel()))
values = var.ravel()

# 创建规则网格
grid_lon = np.linspace(lon.min(), lon.max(), num=lon.shape[1])
grid_lat = np.linspace(lat.min(), lat.max(), num=lat.shape[0])
grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

# 插值到规则网格
grid_var = griddata(points, values, (grid_lon, grid_lat), method='cubic')

# 设置仿射变换和保存为TIFF
transform = from_bounds(lon.min(), lat.min(), lon.max(), lat.max(), lon.shape[1], lat.shape[0])

with rasterio.open(
    'H:\Datasets\Objects\Temp\output2.tif', 'w', driver='GTiff',
    height=lat.shape[0], width=lon.shape[1],
    count=1, dtype=grid_var.dtype,
    crs='+proj=latlong',
    transform=transform
) as dst:
    dst.write(grid_var, 1)

# 清理
dataset.close()



# 03
import rasterio
from rasterio.transform import from_origin
from pyproj import Proj, Transformer
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt

# 打开NC文件
ds = nc.Dataset('I:\InnovationAndEntrepreneurshipOfCollegeStudents\PollutionFromDiesel\Data\GRIDS\D02\GRIDDOT2D.nc', 'r')

# 读取数据
lat = ds.variables['LATD'][:].squeeze()
lon = ds.variables['LOND'][:].squeeze()
var = ds.variables['MSFV2'][:].squeeze().filled(np.nan)

# 定义Lambert Conformal Conic投影
proj_string = "+proj=lcc +lat_1=30 +lat_2=60 +lat_0=30.2 +lon_0=105 +R=6370000 +units=m"
p = Proj(proj_string)
transformer = Transformer.from_proj(Proj('epsg:4326'), p, always_xy=True)

# 将地理坐标转换为投影坐标
proj_x, proj_y = transformer.transform(lon, lat)
proj_x = proj_x.filled(np.nan)
proj_y = proj_y.filled(np.nan)


# 创建仿射变换
transform = from_origin(np.min(proj_x), np.max(proj_y), proj_x.shape[1], proj_y.shape[0])

# 创建输出TIFF文件
with rasterio.open(
    r'H:\Datasets\Objects\Temp\output3.tif', 'w', driver='GTiff',
    height=var.shape[0], width=var.shape[1],
    count=1, dtype=var.dtype,
    crs=proj_string,
    transform=transform
) as dst:
    dst.write(var, 1)

# 关闭文件
ds.close()


# 04
import glob
import os.path
import numpy as np
import netCDF4 as nc  # 读取NC为文件
from scipy.ndimage import zoom
from osgeo import gdal, osr
import matplotlib.pyplot as plt


def data_glt(out_path, src_ds, src_x, src_y, out_res, zoom_scale=6, glt_range=None, windows_size=9, **kwargs):
    """
    基于经纬度数据集对目标数据集进行GLT校正/重投影(WGS84), 并输出为TIFF文件
    :param out_path: 输出tiff文件的路径
    :param src_ds: 目标数据集
    :param src_x: 对应的横轴坐标系(对应地理坐标系的经度数据集)
    :param src_y: 对应的纵轴坐标系(对应地理坐标系的纬度数据集)
    :param out_res: 输出分辨率(单位: 度/°)
    :param zoom_scale: 放大比例(默认为6)
    :param windows_size: 插值的滑动窗口大小
    :param glt_range: GLT校正的经纬度范围
    :return: None
    """
    if glt_range:
        # lon_min, lat_max, lon_max, lat_min = -180.0, 90.0, 180.0, -90.0
        lon_min, lat_max, lon_max, lat_min = glt_range
    else:
        lon_min, lat_max, lon_max, lat_min = np.nanmin(src_x), np.nanmax(src_y), \
            np.nanmax(src_x), np.nanmin(src_y)

    zoom_lon = zoom(src_x, (zoom_scale, zoom_scale), order=0)  # 0为最近邻插值
    zoom_lat = zoom(src_y, (zoom_scale, zoom_scale), order=0)
    # # 确保插值结果正常
    # zoom_lon[(zoom_lon < -180) | (zoom_lon > 180)] = np.nan
    # zoom_lat[(zoom_lat < -90) | (zoom_lat > 90)] = np.nan
    glt_cols = np.ceil((lon_max - lon_min) / out_res).astype(int)
    glt_rows = np.ceil((lat_max - lat_min) / out_res).astype(int)

    deal_bands = []
    for src_ds_band in src_ds:
        glt_ds = np.full((glt_rows, glt_cols), np.nan)
        glt_lon = np.full((glt_rows, glt_cols), np.nan)
        glt_lat = np.full((glt_rows, glt_cols), np.nan)
        geo_x_ix, geo_y_ix = np.floor((zoom_lon - lon_min) / out_res).astype(int), \
            np.floor((lat_max - zoom_lat) / out_res).astype(int)
        glt_lon[geo_y_ix, geo_x_ix] = zoom_lon
        glt_lat[geo_y_ix, geo_x_ix] = zoom_lat
        glt_x_ix, glt_y_ix = np.floor((src_x - lon_min) / out_res).astype(int), \
            np.floor((lat_max - src_y) / out_res).astype(int)
        glt_ds[glt_y_ix, glt_x_ix] = src_ds_band
        # write_tiff('H:\\Datasets\\Objects\\ReadFY3D\\Output\\py_lon.tiff', [glt_lon],
        #            [lon_min, out_res, 0, lat_max, 0, -out_res])
        # write_tiff('H:\\Datasets\\Objects\\ReadFY3D\\Output\\py_lat.tiff', [glt_lat],
        #            [lon_min, out_res, 0, lat_max, 0, -out_res])

        # 插值
        interpolation_ds = np.full_like(glt_ds, fill_value=np.nan)
        jump_size = windows_size // 2
        for row_ix in range(jump_size, glt_rows - jump_size):
            for col_ix in range(jump_size, glt_cols - jump_size):
                if ~np.isnan(glt_ds[row_ix, col_ix]):
                    interpolation_ds[row_ix, col_ix] = glt_ds[row_ix, col_ix]
                    continue
                # 定义当前窗口的边界
                row_start = row_ix - jump_size
                row_end = row_ix + jump_size + 1  # +1 因为切片不包含结束索引
                col_start = col_ix - jump_size
                col_end = col_ix + jump_size + 1
                rows, cols = np.ogrid[row_start:row_end, col_start:col_end]
                distances = np.sqrt((rows - row_ix) ** 2 + (cols - col_ix) ** 2)
                window_ds = glt_ds[(row_ix - jump_size):(row_ix + jump_size + 1),
                            (col_ix - jump_size):(col_ix + jump_size + 1)]
                if np.sum(~np.isnan(window_ds)) == 0:
                    continue
                distances_sort_pos = np.argsort(distances.flatten())
                window_ds_sort = window_ds[np.unravel_index(distances_sort_pos, distances.shape)]
                interpolation_ds[row_ix, col_ix] = window_ds_sort[~np.isnan(window_ds_sort)][0]

        deal_bands.append(np.nan_to_num(interpolation_ds, nan=-9999))
        # print('处理波段: {}'.format(len(deal_bands)))
        # if len(deal_bands) == 6:
        #     break
    write_tiff(out_path, deal_bands, [lon_min, out_res, 0, lat_max, 0, -out_res], out_res, **kwargs)

    return glt_ds


def write_tiff(out_path, dataset, transform, out_res, nodata=np.nan, mask_path=None):
    """
    输出TIFF文件
    :param out_path: 输出文件的路径
    :param dataset: 待输出的数据
    :param transform: 坐标转换信息(形式:[左上角经度, 经度分辨率, 旋转角度, 左上角纬度, 旋转角度, 纬度分辨率])
    :param nodata: 无效值
    :return: None
    """

    # 创建文件
    if not mask_path:
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(out_path, dataset[0].shape[1], dataset[0].shape[0], len(dataset), gdal.GDT_Float32)

        # 设置基本信息
        out_ds.SetGeoTransform(transform)
        out_ds.SetProjection('WGS84')

        # 写入数据
        for i in range(len(dataset)):
            out_ds.GetRasterBand(i + 1).WriteArray(dataset[i])  # GetRasterBand()传入的索引从1开始, 而非0
            out_ds.GetRasterBand(i + 1).SetNoDataValue(-9999)
        out_ds.FlushCache()
        out_ds = None

        return

    mem_driver = gdal.GetDriverByName('MEM')  # 内存中创建
    mem_ds = mem_driver.Create('', dataset[0].shape[1], dataset[0].shape[0], len(dataset), gdal.GDT_Float32)
    # 设置基本信息
    mem_ds.SetGeoTransform(transform)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    mem_ds.SetProjection(srs.ExportToWkt())
    # 写入数据
    for i in range(len(dataset)):
        mem_ds.GetRasterBand(i + 1).WriteArray(dataset[i])  # GetRasterBand()传入的索引从1开始, 而非0
        mem_ds.GetRasterBand(i + 1).SetNoDataValue(-9999)
    mem_ds.FlushCache()
    gdal.Warp(out_path, mem_ds,
              format='GTiff', cutlineDSName=mask_path, cropToCutline=True,
              srcNodata=-9999, dstNodata=-9999, dstSRS='EPSG:4326')

# 打开netCDF文件
dataset = nc.Dataset(r'I:\InnovationAndEntrepreneurshipOfCollegeStudents\PollutionFromDiesel\Data\GRIDS\D02\GRIDDOT2D.nc')

# 读取数据
lat = dataset.variables['LATD'][:].squeeze().filled(np.nan)
lon = dataset.variables['LOND'][:].squeeze().filled(np.nan)
var = dataset.variables['MSFV2'][:].squeeze().filled(np.nan)
a = data_glt(r'H:\Datasets\Objects\Temp\output4.tif', [var], lon, lat, out_res=0.02)