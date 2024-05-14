# @Author   : ChaoQiezi
# @Time     : 2024/5/14  14:42
# @FileName : plot_emission.py
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to 批量绘制空间分布图
"""

import glob
import os.path

from utils.utils import plot_var


# 准备
in_dir = r'I:\InnovationAndEntrepreneurshipOfCollegeStudents\PollutionFromDiesel\Data\time_series\TIFFs'
out_dir = r'I:\InnovationAndEntrepreneurshipOfCollegeStudents\PollutionFromDiesel\Doc\图\月排放量'
shp_path = r'H:\Datasets\BASIC\china_admin_city\ChengDu.shp'
img_paths = glob.glob(os.path.join(in_dir, '*.tiff'))

# 绘制
for img_path in img_paths:
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    pollutant_name, D_N, correct_name = img_name.split('_')
    out_path = os.path.join(out_dir, img_name + '.png')
    plot_var(img_path, out_path, shp_path=shp_path,
             title_name='{}-{}-{}'.format(pollutant_name, D_N, correct_name[:3]),
             cbar_name='{} emission (t)'.format(pollutant_name))

    print('已绘制: {}'.format(img_name))
print('程序结束')



#
# # 读取GeoTIFF文件
# with rasterio.open(tiff_path) as src:
#     band = src.read(1)
#     band_transform = src.transform
#     nodata_value = src.nodata
#     band[band == nodata_value] = np.nan
#
# # 创建一个子图
# fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={'projection': ccrs.PlateCarree()})
#
# # 显示GeoTIFF数据
# show(band, transform=band_transform, ax=ax, cmap='plasma')
#
# # 如果SHP文件路径有效，读取并绘制SHP文件数据
# try:
#     gdf = gpd.read_file(shp_path)
#     gdf.plot(ax=ax, facecolor='none', edgecolor='green', linewidth=1.5)
# except Exception as e:
#     print(f"Error reading SHP file: {e}")
#
# # 添加经纬度网格
# gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
# gl.top_labels = False
# gl.right_labels = False
# gl.xlabel_style = {'size': 16, 'color': 'black', 'weight': 'bold'}
# gl.ylabel_style = {'size': 16, 'color': 'black', 'weight': 'bold'}
#
# # 添加图例
# cbar = plt.colorbar(ax.images[0], ax=ax, orientation='vertical', shrink=0.5, pad=0.05)
# cbar.set_label('Pollution Concentration', size=18, weight='bold')
#
# # 设置标题
# plt.title('Pollution Distribution in Chengdu', fontsize=26, weight='bold')
#
# # 显示图像
# out_path = r'I:\InnovationAndEntrepreneurshipOfCollegeStudents\PollutionFromDiesel\Doc\图\月排放量\temp.png'
# plt.tight_layout()
# plt.savefig(out_path)
# plt.show()

