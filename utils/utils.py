# @Author   : ChaoQiezi
# @Time     : 2024/5/13  22:23
# @FileName : utils.py
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to ...
"""

import glob
import os.path
import numpy as np
import netCDF4 as nc  # 读取NC为文件
from scipy.ndimage import zoom
from osgeo import gdal, osr

# 进行GLT校正
def data_glt(out_path, src_ds, src_x, src_y, out_res, zoom_scale=6, glt_range=None, windows_size=7, **kwargs):
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


# 输出为tiff文件
def write_tiff(out_path, dataset, transform, out_res, nodata=-9999, mask_path=None):
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
            out_ds.GetRasterBand(i + 1).SetNoDataValue(nodata)
        out_ds.FlushCache()

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
        mem_ds.GetRasterBand(i + 1).SetNoDataValue(nodata)
    mem_ds.FlushCache()
    gdal.Warp(out_path, mem_ds,
              format='GTiff', cutlineDSName=mask_path, cropToCutline=True,
              srcNodata=nodata, dstNodata=nodata, dstSRS='EPSG:4326')