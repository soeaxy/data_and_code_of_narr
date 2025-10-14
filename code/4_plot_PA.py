import rasterio
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

# Use Times New Roman for all plot text
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'
from sklearn.metrics import auc

# === Step 1. 各模型预测结果路径 ===
model_tifs = {
    "Balanced CatBoost": r"E:\01项目相关\2024\湖山铀矿\成矿预测\output_dir\cut_husab_balanced_catboost.tif",
    "Balanced RF": r"E:\01项目相关\2024\湖山铀矿\成矿预测\output_dir\cut_husab_balanced_rf.tif",
    "Balanced LGBM": r"E:\01项目相关\2024\湖山铀矿\成矿预测\output_dir\cut_husab_balanced_lgbm.tif",
    "CatBoost": r"E:\01项目相关\2024\湖山铀矿\成矿预测\output_dir\cut_husab_catboost.tif",
    "Random Forest": r"E:\01项目相关\2024\湖山铀矿\成矿预测\output_dir\cut_husab_rf.tif",
    "LightGBM": r"E:\01项目相关\2024\湖山铀矿\成矿预测\output_dir\cut_husab_lgbm.tif"
}

# === Step 2. 读取第一个栅格获取基础信息 ===
ref_model = list(model_tifs.keys())[0]
with rasterio.open(model_tifs[ref_model]) as src:
    transform = src.transform
    crs_raster = src.crs
    nodata = src.nodata
    bounds_raster = src.bounds
    shape_raster = src.shape

print(f"✅ Reference raster loaded ({ref_model})")
print(f"Raster CRS: {crs_raster}")
print(f"Raster shape: {shape_raster}")
print(f"Raster bounds: {bounds_raster}")

# === Step 3. 读取矿化点并匹配CRS ===
points = gpd.read_file(r"E:\01项目相关\2024\湖山铀矿\成矿预测\工程文件\Shps\combined_samples.shp")
print(f"Points CRS before transform: {points.crs}")
if points.crs != crs_raster:
    print("⚠️ CRS mismatch detected — transforming points to raster CRS...")
    points = points.to_crs(crs_raster)

points = points[points['Class'] == 1]
print(f"Total mineralized points: {len(points)}")

# === Step 4. 将矿化点映射到栅格索引 ===
cols, rows = zip(*[~transform * (x, y) for x, y in zip(points.geometry.x, points.geometry.y)])
rows, cols = np.array(rows).astype(int), np.array(cols).astype(int)

valid = (rows >= 0) & (rows < shape_raster[0]) & (cols >= 0) & (cols < shape_raster[1])
rows, cols = rows[valid], cols[valid]
print(f"Valid points within raster: {len(rows)} / {len(points)} ({len(rows)/len(points)*100:.1f}%)")

# === Step 5. 可视化点与参考栅格对齐 ===
with rasterio.open(model_tifs[ref_model]) as src:
    ref_prob = src.read(1)
    ref_prob = np.where(ref_prob == nodata, np.nan, ref_prob)

plt.figure(figsize=(8,6))
plt.imshow(ref_prob, cmap='viridis', origin='upper')
plt.scatter(cols, rows, color='red', s=10, label='Mineralized Points')
plt.title(f"Alignment Check ({ref_model})")
plt.legend()
plt.xlabel("Column Index")
plt.ylabel("Row Index")
plt.tight_layout()
plt.show()

# === Step 6. 定义计算P–A曲线的函数 ===
def compute_pa_curve(prob, rows, cols):
    flat_prob = prob.flatten()
    flat_prob = flat_prob[~np.isnan(flat_prob)]
    thresholds = np.linspace(1.0, 0.0, 100)
    total_points = len(rows)
    cum_area, cum_capture = [], []
    for t in thresholds:
        mask = prob >= t
        area_ratio = np.sum(mask) / np.sum(~np.isnan(prob))
        hit_count = np.sum(mask[rows, cols])
        hit_ratio = hit_count / total_points
        cum_area.append(area_ratio)
        cum_capture.append(hit_ratio)
    pa_auc = auc(cum_area, cum_capture)
    return np.array(cum_area), np.array(cum_capture), pa_auc

# === Step 7. 批量计算6个模型的P–A曲线 ===
results = {}
for model_name, tif_path in model_tifs.items():
    print(f"\n▶ Processing model: {model_name}")
    with rasterio.open(tif_path) as src:
        prob = src.read(1)
        prob = np.where(prob == src.nodata, np.nan, prob)
    area, capture, pa_auc = compute_pa_curve(prob, rows, cols)
    results[model_name] = {'area': area, 'capture': capture, 'auc': pa_auc}
    print(f"  AUC(P–A): {pa_auc:.4f} | Range: [{np.nanmin(prob):.3f}, {np.nanmax(prob):.3f}]")

# === Step 8. 绘制6个模型的P–A对比曲线 ===
plt.figure(figsize=(9,7))
colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b']
for (model_name, res), color in zip(results.items(), colors):
    plt.plot(res['area']*100, res['capture']*100, lw=2, color=color,
             label=f"{model_name} (PA-AUC={res['auc']:.3f})")

plt.plot([0,100],[0,100],'k--',label='Random model')
plt.xlabel('Cumulative Predicted Area (%)', fontsize=12)
plt.ylabel('Cumulative Captured Mineralized Points (%)', fontsize=12)
plt.title('Prediction–Area (P–A) Curves Comparison for Six Models')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
# Save high-resolution figure (600 DPI). Adjust path/filename as needed.
save_path = r"pa_curves.png"
plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
plt.show()

print("\n✅ All 6 P–A curves computed and plotted successfully.")
