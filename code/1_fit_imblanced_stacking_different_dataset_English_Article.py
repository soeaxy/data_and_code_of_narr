import os
import itertools
import joblib
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.calibration import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, roc_curve, roc_auc_score,
                             precision_recall_curve, auc, accuracy_score,
                             balanced_accuracy_score, recall_score,
                             f1_score, cohen_kappa_score, matthews_corrcoef)
from sklearn.model_selection import train_test_split
from imblearn.pipeline import make_pipeline
from imblearn.ensemble import BalancedRandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
# from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
import shap
# from pdpbox import pdp

# Utility functions
def plot_confusion_matrix(cm, classes, ax, title='Confusion matrix', cmap=plt.cm.Blues):
    ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')


def extract_factors_from_tifs(points, tifs):
    coords = [(point.x, point.y) for point in points.geometry]
    data = []
    for tif_file in tifs:
        with rasterio.open(tif_file) as src:
            values = [val[0] if val[0] is not None else np.nan for val in src.sample(coords)]
            data.append(values)
    return np.column_stack(data)


def train_and_evaluate(pipeline, X_train, y_train, X_test, y_test, model_name, output_dir, model_dir):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    # Plot and save confusion matrix
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_confusion_matrix(cm, classes=[0, 1], ax=ax, title=f'{model_name} Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png'))
    plt.close(fig)

    # Save model
    joblib.dump(pipeline, os.path.join(model_dir, f'{model_name.lower().replace(" ", "_")}.pkl'))

    # Metrics
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Balanced Accuracy': balanced_accuracy_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'Kappa': cohen_kappa_score(y_test, y_pred),
        'MCC': matthews_corrcoef(y_test, y_pred)
    }
    return metrics

# Custom Stacking Classes
class MeanStacking(BaseEstimator, RegressorMixin):
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)
        return self

    def predict(self, X):
        predictions = [model.predict(X) for model in self.models]
        return np.mean(predictions, axis=0)

class MetaLearningClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, models, meta_model):
        self.models = models
        self.meta_model = meta_model

    def fit(self, X, y):
        meta_X = np.column_stack([cross_val_predict(model, X, y, method='predict_proba') for model in self.models])
        self.meta_model.fit(meta_X, y)
        for model in self.models:
            model.fit(X, y)
        return self

    def predict(self, X):
        meta_X = np.column_stack([model.predict_proba(X) for model in self.models])
        return self.meta_model.predict(meta_X)

# Stacking Helper Functions
def create_voting_classifier(models):
    estimators = [(model_name, model) for model_name, model in models.items()]
    return VotingClassifier(estimators=estimators, voting='soft')

def create_stacking_classifier(models):
    estimators = [(model_name, model) for model_name, model in models.items()]
    final_estimator = LogisticRegression()
    return StackingClassifier(estimators=estimators, final_estimator=final_estimator)

def plot_curves(pipelines, X_test, y_test, labels, output_dir):
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    for pipeline, label in zip(pipelines, labels):
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f'{label} (ROC-AUC={roc_auc:.3f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png') , dpi=600)
    plt.close()

    plt.figure()
    for pipeline, label in zip(pipelines, labels):
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f'{label} (PR-AUC={pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'), dpi=600)
    plt.close()

def compute_and_plot_shap_values(model, X, model_name, plot_type='bar'):
    plt.rcParams['font.family'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 12

    # 使用TreeExplainer或KernelExplainer
    explainer = shap.Explainer(model.predict_proba, X)

    # 获取shap_values，注意二分类时是长度为2的列表，分别对应两个类别
    shap_values = explainer(X)

    # 合并正负样本的SHAP值
    shap_values_combined = shap_values[..., 1] - shap_values[..., 0]

    # 绘制SHAP图，将正负样本放在同一张图上
    shap.summary_plot(shap_values_combined, X, plot_type=plot_type, show=False)
    plt.tight_layout()
    plt.savefig(f'{model_name}_combined_{plot_type}_SHAP_plot.jpg', dpi=600)
    plt.close()

    return shap_values


# Main workflow
def main():
    base_dir = r'E:\01项目相关\2024\湖山铀矿\成矿预测'
    shapefile_path = os.path.join(base_dir, r'工程文件\Shps\combined_samples.shp')
    folder_paths = ['归一化航放2', '归一化航磁2', '归一化地质', '归一化遥感']
    output_dir = os.path.join(base_dir, 'Result_英文论文')
    model_dir = os.path.join(base_dir, 'output_dir\航放_航磁_地质_遥感')

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    points = gpd.read_file(shapefile_path)
    tif_files = [os.path.join(base_dir, folder, f)
                 for folder in folder_paths
                 for f in os.listdir(os.path.join(base_dir, folder)) if f.endswith('.tif')]

    feature_names = [os.path.splitext(os.path.basename(tif))[0] for tif in tif_files]
    X = pd.DataFrame(extract_factors_from_tifs(points, tif_files), columns=feature_names)
    y = points['Class']
    
    # Ensure feature alignment between training and testing data
    X = X.loc[:, ~X.columns.duplicated()]  # Remove duplicate columns if any

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    models = {
        
        'LGBM': LGBMClassifier(random_state=42),
        'CatBoost': CatBoostClassifier(random_state=42, verbose=0),
        'RF': RandomForestClassifier(random_state=42),
        'Class-weighted LGBM': LGBMClassifier(class_weight='balanced', random_state=42),
        'Class-weighted CatBoost': CatBoostClassifier(class_weights=[1, 10], random_state=42, verbose=0),
        'Balanced RF': BalancedRandomForestClassifier(random_state=42),
    }

    pipelines = {name: make_pipeline(StandardScaler(), model) for name, model in models.items()}

    # Define base learners for stacking
    estimators = [(name, pipeline) for name, pipeline in pipelines.items()]

    # Define stacking classifier
    stacking_model = StackingClassifier(
        estimators=estimators,
        final_estimator = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000),
        cv=5
    )

    # Add stacking model to pipelines
    pipelines['Stacking'] = stacking_model

    # Train stacking model and calculate SHAP and PDP values

    # Fit the stacking model
    stacking_model.fit(X_train, y_train)
    # Calculate and plot SHAP values using the utility function
    # Summarize the background data to speed up SHAP calculations
    background = shap.sample(X_train, 100)  # Use 100 samples as the background data
    # Ensure feature alignment between training and testing data
    X_train = X_train.loc[:, X_train.columns.isin(X_test.columns)]
    X_test = X_test.loc[:, X_test.columns.isin(X_train.columns)]

    # compute_and_plot_shap_values(stacking_model, background, "StackingClassifier", plot_type='bar')
    # compute_and_plot_shap_values(stacking_model, background, "StackingClassifier", plot_type='beeswarm')

    
    metrics_summary = {}
    for name, pipeline in pipelines.items():
        metrics = train_and_evaluate(pipeline, X_train, y_train, X_test, y_test, name, output_dir, model_dir)
        metrics_summary[name] = metrics

    pd.DataFrame(metrics_summary).to_csv(os.path.join(output_dir, 'model_metrics.csv'))

    plot_curves(list(pipelines.values()), X_test, y_test, list(pipelines.keys()), output_dir)


if __name__ == "__main__":
    main()