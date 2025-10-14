import os
import numpy as np
import rasterio
import geopandas as gpd
from rasterio.transform import from_origin
import joblib

# 加载训练好的模型
model_paths = [
   r'F:\01项目相关\2024\湖山铀矿\成矿预测\output_dir\航放_航磁_地质_遥感\lgbm.pkl',
   r'F:\01项目相关\2024\湖山铀矿\成矿预测\output_dir\航放_航磁_地质_遥感\balanced_lgbm.pkl',
   r'F:\01项目相关\2024\湖山铀矿\成矿预测\output_dir\航放_航磁_地质_遥感\rf.pkl',
   r'F:\01项目相关\2024\湖山铀矿\成矿预测\output_dir\航放_航磁_地质_遥感\balanced_rf.pkl',   
   r'F:\01项目相关\2024\湖山铀矿\成矿预测\output_dir\航放_航磁_地质_遥感\catboost.pkl',
   r'F:\01项目相关\2024\湖山铀矿\成矿预测\output_dir\航放_航磁_地质_遥感\balanced_catboost.pkl',
   r'F:\01项目相关\2024\湖山铀矿\成矿预测\output_dir\航放_航磁_地质_遥感\stacking.pkl',

   
]

# 加载研究区边界
research_area = gpd.read_file(r'..\05_Geography\EPL_Husab_Mine.shp')

# 提供TIFF文件所在文件夹路径
folder_path = r'F:\01项目相关\2024\湖山铀矿\成矿预测\归一化航放2'
folder_path_2 = r'F:\01项目相关\2024\湖山铀矿\成矿预测\归一化航磁2'
folder_path_3 = r'F:\01项目相关\2024\湖山铀矿\成矿预测\归一化地质'
folder_path_4 = r'F:\01项目相关\2024\湖山铀矿\成矿预测\归一化遥感'

tif_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.tif')]
tif_files += [os.path.join(folder_path_2, f) for f in os.listdir(folder_path_2) if f.endswith('.tif')]
tif_files += [os.path.join(folder_path_3, f) for f in os.listdir(folder_path_3) if f.endswith('.tif')]
tif_files += [os.path.join(folder_path_4, f) for f in os.listdir(folder_path_4) if f.endswith('.tif')]


# 打印所有tif文件路径
for tif_file in tif_files:
    print(tif_file)

feature_names = [os.path.splitext(os.path.basename(tif))[0] for tif in tif_files]

# 加载第一个因子文件获取范围和分辨率
with rasterio.open(tif_files[0]) as src:
    bounds = src.bounds
    resolution = 20  # 每隔20米
    width = int((bounds.right - bounds.left) / resolution)
    height = int((bounds.top - bounds.bottom) / resolution)
    transform = from_origin(bounds.left, bounds.top, resolution, resolution)

# 生成网格点坐标
x_coords = np.linspace(bounds.left, bounds.right, width)
y_coords = np.linspace(bounds.top, bounds.bottom, height)
grid_x, grid_y = np.meshgrid(x_coords, y_coords)

# 将网格点转换为二维坐标数组
grid_points = np.array([(x, y) for x, y in zip(grid_x.ravel(), grid_y.ravel())])

# 从TIFF文件中提取因子值
def extract_factors_from_grid(grid, tifs):
    data = []
    for tif_file in tifs:
        with rasterio.open(tif_file) as src:
            values = [val[0] for val in src.sample(grid)]
            data.append(values)
    return np.array(data).T

# 提取因子值
factor_values = extract_factors_from_grid(grid_points, tif_files)

# 对每个模型进行预测并将结果保存为GeoTIFF
for model_path in model_paths:
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    model = joblib.load(model_path)
    probabilities = model.predict_proba(factor_values)[:, 1]  # 获取属于正类的概率

    # 重塑概率为网格的形状
    probability_grid = probabilities.reshape(height, width)

    # 保存为GeoTIFF
    model_folder_name = os.path.basename(os.path.dirname(model_path))
    output_tif_path = f'F:\\01项目相关\\2024\\湖山铀矿\\成矿预测\\output_dir\\husab_mine_proba_{model_folder_name}_{model_name}.tif'
    with rasterio.open(
        output_tif_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype='float32',
        crs=src.crs,  # 使用因子文件的CRS
        transform=transform
    ) as dst:
        dst.write(probability_grid, 1)

    print(f"概率TIFF文件已保存到: {output_tif_path}")
