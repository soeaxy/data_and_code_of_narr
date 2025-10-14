import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Model ranking data from the image
rank_data = {
    "Model": [
        "LGBM", "CatBoost", "RF", "Balanced LGBM",
        "Balanced CatBoost", "Balanced RF", "Stacking"
    ],
    "Accuracy": [3, 1, 4, 6, 5, 7, 2],
    "Balanced \nAccuracy": [6, 4, 7, 2, 1, 5, 3],
    "Recall": [6, 5, 7, 3, 2, 1, 4],
    "F1 Score": [3, 1, 4, 6, 5, 7, 2],
    "Kappa": [3, 1, 4, 6, 5, 7, 2],
    "MCC": [3, 1, 4, 6, 5, 7, 2],
}
rank_df = pd.DataFrame(rank_data)
rank_df = rank_df.set_index('Model')

# Plot the heatmap with Times New Roman and larger font
plt.figure(figsize=(10, 6))
sns.set(font="Times New Roman", font_scale=1.4)
ax = sns.heatmap(rank_df, annot=True, cmap='Oranges', fmt='d', cbar_kws={
                 'label': 'Rank'}, annot_kws={"size": 20, "fontname": "Times New Roman"})

plt.title('Model Rankings Across Performance Metrics (Lower is Better)',
          fontsize=20, fontname='Times New Roman')
plt.ylabel('Model', fontsize=20, fontname='Times New Roman')
plt.xlabel(' ', fontsize=20, fontname='Times New Roman')

plt.xticks(fontsize=15, fontname='Times New Roman')
plt.yticks(fontsize=15, fontname='Times New Roman', rotation=0)

plt.tight_layout()
plt.savefig('classifier_performance_ranks.png', dpi=600, bbox_inches='tight')
plt.show()
