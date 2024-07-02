import pickle
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc
import shap

def load_model(filename):
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

def plot_roc_curve(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, 'b', label=f'AUC = {roc_auc:.3f}')
    ax.plot([0, 1], [0, 1], 'r--')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    return fig

def plot_feature_importance(model):
    feature_importance = pd.DataFrame({
        'feature': model.feature_names_in_,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    fig, ax = plt.subplots()
    ax.bar(feature_importance['feature'], feature_importance['importance'])
    ax.set_xticklabels(feature_importance['feature'], rotation=90)
    ax.set_xlabel('Features')
    ax.set_ylabel('Importance')
    ax.set_title('Feature Importance')
    plt.tight_layout()
    return fig

def plot_shap_summary(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X.iloc[:100])  # Using only first 100 for speed
    fig = plt.figure()
    shap.summary_plot(shap_values[1], X.iloc[:100], plot_type="bar", show=False)
    return fig
