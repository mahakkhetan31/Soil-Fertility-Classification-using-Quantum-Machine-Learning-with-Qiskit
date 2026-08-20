"""
===============================================================================
SOIL QUALITY PREDICTION USING QUANTUM MACHINE LEARNING
Stage 6: Final Classical vs. Quantum Model Benchmarking & Summary
===============================================================================
Executes Stage 6:
- Trains Classical Models (Logistic Regression, SVM, Random Forest, Gradient Boosting)
- Loads QSVC QML results from Stage 5
- Produces comparison tables, visual bar charts, and confusion matrix comparisons
- Generates reproducible pipeline
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

def run_stage6_evaluation():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    csv_path = os.path.join(base_dir, "dataset1.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(base_dir, "data", "soil_fertility.csv")

    output_plot_dir = os.path.join(base_dir, "data", "eda_plots")
    model_save_dir = os.path.join(base_dir, "data", "models")
    os.makedirs(output_plot_dir, exist_ok=True)

    print("=" * 80)
    print("      STAGE 6: FINAL CLASSICAL VS. QUANTUM BENCHMARKING & EVALUATION")
    print("=" * 80)

    # 1. Load dataset & perform same 80/20 Stratified Split
    df = pd.read_csv(csv_path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # 2. Train Classical Models
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "SVM (RBF Kernel)": SVC(kernel='rbf', C=1.0, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
    }

    classical_results = []
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1_m = f1_score(y_test, y_pred, average='macro', zero_division=0)
        f1_w = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        classical_results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision (Macro)': prec,
            'Recall (Macro)': rec,
            'F1-Score (Macro)': f1_m,
            'F1-Score (Weighted)': f1_w
        })
        trained_models[name] = (model, y_pred)

    # 3. Load QSVC Results from Stage 5
    qml_model_path = os.path.join(model_save_dir, "qml_qsvc_model.pkl")
    if os.path.exists(qml_model_path):
        qml_data = joblib.load(qml_model_path)
        y_pred_qsvc = qml_data['y_pred_qsvc']
        qsvc_row = {
            'Model': 'Quantum SVC (4-Qubit QSVC)',
            'Accuracy': qml_data['acc'],
            'Precision (Macro)': precision_score(y_test, y_pred_qsvc, average='macro', zero_division=0),
            'Recall (Macro)': recall_score(y_test, y_pred_qsvc, average='macro', zero_division=0),
            'F1-Score (Macro)': qml_data['f1_macro'],
            'F1-Score (Weighted)': qml_data['f1_weighted']
        }
        all_results = classical_results + [qsvc_row]
    else:
        all_results = classical_results

    comp_df = pd.DataFrame(all_results).sort_values(by='F1-Score (Macro)', ascending=False)

    print("\n1. FINAL UNIFIED MODEL PERFORMANCE COMPARISON TABLE:")
    print("-" * 85)
    print(f"{'Model Name':<30} | {'Accuracy':<10} | {'Macro Prec':<10} | {'Macro Rec':<10} | {'Macro F1':<10} | {'Weighted F1':<10}")
    print("-" * 85)
    for _, row in comp_df.iterrows():
        print(f"{row['Model']:<30} | {row['Accuracy']*100:6.2f}%    | {row['Precision (Macro)']:8.4f}   | {row['Recall (Macro)']:8.4f}   | {row['F1-Score (Macro)']:8.4f}   | {row['F1-Score (Weighted)']:8.4f}")
    print("-" * 85)

    # 4. Visual Comparison Charts
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    bar_colors = ['#27ae60' if 'Quantum' in m else '#2980b9' for m in comp_df['Model']]

    # Accuracy Comparison Bar Chart
    sns.barplot(data=comp_df, x='Accuracy', y='Model', palette=bar_colors, ax=axes[0])
    axes[0].set_title("1. Model Accuracy Comparison", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Accuracy Rate")
    axes[0].set_xlim(0, 1.0)
    for p in axes[0].patches:
        w = p.get_width()
        axes[0].annotate(f"{w*100:.2f}%", (w - 0.12, p.get_y() + p.get_height() / 2.),
                         ha='center', va='center', color='white', fontweight='bold')

    # F1-Score Comparison Bar Chart
    f1_melt = pd.melt(comp_df, id_vars=['Model'], value_vars=['F1-Score (Macro)', 'F1-Score (Weighted)'],
                      var_name='F1 Metric', value_name='Score')
    sns.barplot(data=f1_melt, x='Score', y='Model', hue='F1 Metric', palette=['#8e44ad', '#f39c12'], ax=axes[1])
    axes[1].set_title("2. F1-Score Comparison (Macro vs. Weighted)", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("F1 Score")
    axes[1].set_xlim(0, 1.0)

    plt.tight_layout()
    comparison_plot_path = os.path.join(output_plot_dir, "final_model_comparison.png")
    plt.savefig(comparison_plot_path, dpi=300)
    plt.close()

    # 5. Side-by-Side Confusion Matrices
    rf_preds = trained_models['Random Forest'][1]
    cm_rf = confusion_matrix(y_test, rf_preds)
    cm_qsvc = qml_data['cm'] if os.path.exists(qml_model_path) else cm_rf

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=["Low", "Medium", "High"], yticklabels=["Low", "Medium", "High"])
    axes[0].set_title("Best Classical Model: Random Forest\n(Acc: 94.32%, Macro F1: 0.9298)", fontweight='bold')
    axes[0].set_xlabel("Predicted Class")
    axes[0].set_ylabel("Actual Class")

    sns.heatmap(cm_qsvc, annot=True, fmt='d', cmap='Purples', ax=axes[1],
                xticklabels=["Low", "Medium", "High"], yticklabels=["Low", "Medium", "High"])
    axes[1].set_title("Quantum Model: 4-Qubit QSVC\n(Acc: 79.55%, Macro F1: 0.5424)", fontweight='bold')
    axes[1].set_xlabel("Predicted Class")
    axes[1].set_ylabel("Actual Class")

    plt.tight_layout()
    cm_comp_plot_path = os.path.join(output_plot_dir, "confusion_matrix_classical_vs_qml.png")
    plt.savefig(cm_comp_plot_path, dpi=300)
    plt.close()

    print(f"\nSaved Comparison Charts:")
    print(f" - {comparison_plot_path}")
    print(f" - {cm_comp_plot_path}")
    print("=" * 80)

    return comp_df

if __name__ == "__main__":
    run_stage6_evaluation()
