# @Author   : ChaoQiezi
# @Time     : 2024/5/14  14:42
# @FileName : plot_emission.py
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to 批量绘制空间分布图
"""

import geopandas as gpd
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from rasterio.plot import show

# # 初稿
# shp_path = r'H:\Datasets\BASIC\china_admin_city\ChengDu.shp'
# tiff_path = r'I:\InnovationAndEntrepreneurshipOfCollegeStudents\PollutionFromDiesel\Data\time_series\TIFFs\CO_Day_GFXOBD.tiff'
# gdf = gpd.read_file(shp_path)
# with rasterio.open(tiff_path) as src:
#     band = src.read(1)
#     band_transform = src.transform
#     nodata_value = src.nodata
#     band[band == nodata_value] = np.nan
#
# fig, ax = plt.subplots(figsize=(10, 10))
# show(band, transform=band_transform, ax=ax, cmap='viridis')
# gdf.plot(ax=ax, facecolor='none', edgecolor='green')
# plt.show()

#  稿2
import geopandas as gpd
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# 读取SHP文件
shp_path = r'H:\Datasets\BASIC\china_admin_city\ChengDu.shp'
gdf = gpd.read_file(shp_path)

# 读取GeoTIFF文件
tiff_path = r'I:\InnovationAndEntrepreneurshipOfCollegeStudents\PollutionFromDiesel\Data\time_series\TIFFs\CO_Day_GFXOBD.tiff'
with rasterio.open(tiff_path) as src:
    band = src.read(1)
    band_transform = src.transform
    nodata_value = src.nodata
    band[band == nodata_value] = np.nan

# 创建一个子图
fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={'projection': ccrs.PlateCarree()})

# 显示GeoTIFF数据
show(band, transform=band_transform, ax=ax, cmap='viridis', title="Pollution Distribution in Chengdu")

# 绘制SHP文件数据
gdf.plot(ax=ax, facecolor='none', edgecolor='green', linewidth=1.5)

# 添加自然地理特征
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')

# 添加经纬度网格
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--', fontsize=20)
gl.top_labels = False
gl.right_labels = False

# 添加图例
cbar = plt.colorbar(ax.images[0], ax=ax, orientation='vertical', shrink=0.5, pad=0.05)
cbar.set_label('Pollution Concentration')

# 设置标题
plt.title('Pollution Distribution in Chengdu', fontsize=15)

# 显示图像
plt.tight_layout()
plt.show()


# 稿3
import geopandas as gpd
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# 设置字体
plt.rcParams['font.family'] = 'Times New Roman'

# 读取SHP文件路径（假设路径已经正确设置）
shp_path = r'H:\Datasets\BASIC\china_admin_city\ChengDu.shp'

# 读取GeoTIFF文件
tiff_path = r'I:\InnovationAndEntrepreneurshipOfCollegeStudents\PollutionFromDiesel\Data\time_series\TIFFs\CO_Day_GFXOBD.tiff'
with rasterio.open(tiff_path) as src:
    band = src.read(1)
    band_transform = src.transform
    nodata_value = src.nodata
    band[band == nodata_value] = np.nan

# 创建一个子图
fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={'projection': ccrs.PlateCarree()})

# 显示GeoTIFF数据
show(band, transform=band_transform, ax=ax, cmap='plasma')

# 如果SHP文件路径有效，读取并绘制SHP文件数据
try:
    gdf = gpd.read_file(shp_path)
    gdf.plot(ax=ax, facecolor='none', edgecolor='green', linewidth=1.5)
except Exception as e:
    print(f"Error reading SHP file: {e}")

# 添加自然地理特征
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')

# 添加经纬度网格
gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 16, 'color': 'black', 'weight': 'bold'}
gl.ylabel_style = {'size': 16, 'color': 'black', 'weight': 'bold'}

# 添加图例
cbar = plt.colorbar(ax.images[0], ax=ax, orientation='vertical', shrink=0.5, pad=0.05)
cbar.set_label('Pollution Concentration', size=18, weight='bold')

# 设置标题
plt.title('Pollution Distribution in Chengdu', fontsize=26, weight='bold')

# 显示图像
out_path = r'I:\InnovationAndEntrepreneurshipOfCollegeStudents\PollutionFromDiesel\Doc\图\月排放量\temp.png'
plt.tight_layout()
plt.savefig(out_path)
plt.show()

