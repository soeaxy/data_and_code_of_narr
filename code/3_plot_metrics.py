import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Define the data
data = {
    "Model": [
        "LGBM", "CatBoost", "RF", "Balanced LGBM", 
        "Balanced CatBoost", "Balanced RF", "Stacking"
    ],
    "Accuracy": [0.991, 0.992, 0.990, 0.984, 0.986, 0.956, 0.992],
    "Balanced \nAccuracy": [0.966, 0.968, 0.958, 0.979, 0.981, 0.968, 0.973],
    "Recall": [0.935, 0.940, 0.919, 0.973, 0.975, 0.983, 0.950],
    "F1 Score": [0.950, 0.954, 0.944, 0.916, 0.926, 0.801, 0.953],
    "Kappa": [0.945, 0.950, 0.938, 0.907, 0.919, 0.777, 0.949],
    "MCC": [0.945, 0.950, 0.939, 0.909, 0.920, 0.794, 0.949],
}
df = pd.DataFrame(data)
df = df.set_index('Model')

# Plot
plt.figure(figsize=(10, 6))
sns.set(font="Times New Roman", font_scale=1.4)
ax = sns.heatmap(df, annot=True, cmap='YlGnBu', fmt='.3f', cbar_kws={'label': 'Score'}, annot_kws={"size": 16, "fontname": "Times New Roman"})

plt.title('Classifier Performance Metrics', fontsize=16, fontname='Times New Roman')
plt.ylabel('Model', fontsize=18, fontname='Times New Roman')
plt.xlabel(' ', fontsize=18, fontname='Times New Roman')

plt.xticks(fontsize=15, fontname='Times New Roman', rotation=45)
plt.yticks(fontsize=15, fontname='Times New Roman')

plt.tight_layout()
plt.savefig('classifier_performance_metrics.png', dpi=600, bbox_inches='tight') 
plt.show()
