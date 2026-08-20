"""
===============================================================================
SOIL QUALITY PREDICTION USING QUANTUM MACHINE LEARNING
Stage 3: Classical Machine Learning Baseline & QML Feature Selection Pipeline
===============================================================================
This script executes:
- Part 1: Stratified Train/Test Split (80/20) & Leakage-Free Scaling
- Part 2: Feature Selection (Correlation, Mutual Info, Random Forest Importance)
- Part 3: Classical Model Training (Logistic Regression, SVM, Random Forest, Gradient Boosting)
- Part 4: Comprehensive Model Evaluation (Accuracy, Precision, Recall, Macro/Weighted F1, Confusion Matrices)
- Part 5: Model Selection & Comparison
- Part 6: QML Feature Subset Definition & Export
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# Set visual style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

def run_classical_baseline():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    csv_path = os.path.join(base_dir, "dataset1.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(base_dir, "data", "soil_fertility.csv")

    output_plot_dir = os.path.join(base_dir, "data", "eda_plots")
    model_save_dir = os.path.join(base_dir, "data", "models")
    os.makedirs(output_plot_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)

    print("=" * 80)
    print("      STAGE 3: CLASSICAL ML BASELINE & QML FEATURE SELECTION PIPELINE")
    print("=" * 80)

    # Load dataset
    df = pd.read_csv(csv_path)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    feature_names = list(X.columns)

    # -------------------------------------------------------------------------
    # PART 1 — TRAIN/TEST SPLIT (PREVENTING DATA LEAKAGE)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print("PART 1: TRAIN/TEST SPLIT (80% TRAIN / 20% TEST)")
    print("=" * 40)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X.values, y.values, test_size=0.20, random_state=42, stratify=y.values
    )

    print(f"Stratified Train-Test Split Completed:")
    print(f" - Training set: {X_train_raw.shape[0]} samples")
    print(f" - Testing set:  {X_test_raw.shape[0]} samples")

    # Fit Scaler ONLY on Training Set
    scaler_full = StandardScaler()
    X_train_full = scaler_full.fit_transform(X_train_raw)
    X_test_full = scaler_full.transform(X_test_raw)

    print(" - Features scaled using StandardScaler (Fitted strictly on Training data).")

    # -------------------------------------------------------------------------
    # PART 2 — FEATURE SELECTION FOR QML
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print("PART 2: MULTI-METHOD FEATURE SELECTION ANALYSIS")
    print("=" * 40)

    # Method 1: Correlation with Target
    corr_series = df.corr()['Output'].drop('Output').abs().sort_values(ascending=False)

    # Method 2: Mutual Information
    mi_scores = mutual_info_classif(X_train_raw, y_train, random_state=42)
    mi_series = pd.Series(mi_scores, index=feature_names).sort_values(ascending=False)

    # Method 3: Random Forest Feature Importance
    rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_selector.fit(X_train_raw, y_train)
    rf_imp_series = pd.Series(rf_selector.feature_importances_, index=feature_names).sort_values(ascending=False)

    # Combine metrics into a unified DataFrame
    feat_analysis = pd.DataFrame({
        'Abs Correlation': corr_series,
        'Mutual Info Score': mi_series,
        'RF Importance': rf_imp_series
    }).fillna(0)

    # Rank features across all methods
    feat_analysis['Composite Rank'] = (
        feat_analysis['Abs Correlation'].rank(ascending=False) +
        feat_analysis['Mutual Info Score'].rank(ascending=False) +
        feat_analysis['RF Importance'].rank(ascending=False)
    ) / 3.0

    feat_analysis.sort_values(by='Composite Rank', inplace=True)
    print("\nFeature Selection Summary Table (Ranked by Composite Score):")
    print(feat_analysis.round(4).to_string())

    # Selecting Top 4 Compact Features for QML Circuit (4 Qubits)
    qml_selected_features = list(feat_analysis.index[:4])
    print(f"\n---> FINAL COMPACT QML FEATURE SUBSET SELECTED (4 Features / 4 Qubits):")
    for idx, f_name in enumerate(qml_selected_features, 1):
        print(f"     Qubit {idx-1} <---> Feature '{f_name}'")

    # Get column indices for selected features
    qml_feat_indices = [feature_names.index(f) for f in qml_selected_features]
    X_train_qml = X_train_full[:, qml_feat_indices]
    X_test_qml = X_test_full[:, qml_feat_indices]

    # -------------------------------------------------------------------------
    # PART 3 & 4 — CLASSICAL MODELS TRAINING & EVALUATION
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print("PART 3 & 4: CLASSICAL MODELS TRAINING & EVALUATION")
    print("=" * 40)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Support Vector Machine (SVM)": SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    results = []
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, (m_name, model) in enumerate(models.items()):
        # Train model on full 12 features
        model.fit(X_train_full, y_train)
        y_pred = model.predict(X_test_full)

        acc = accuracy_score(y_test, y_pred)
        prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        results.append({
            'Model': m_name,
            'Accuracy': acc,
            'Precision (Macro)': prec_macro,
            'Recall (Macro)': rec_macro,
            'F1-Score (Macro)': f1_macro,
            'F1-Score (Weighted)': f1_weighted
        })

        print(f"\n--- {m_name} (Full 12 Features) ---")
        print(f"Accuracy:           {acc * 100:.2f}%")
        print(f"Macro F1-Score:     {f1_macro:.4f}")
        print(f"Weighted F1-Score:  {f1_weighted:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=["Low (0)", "Medium (1)", "High (2)"], zero_division=0))

        # Save Confusion Matrix Plot
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=["Low", "Medium", "High"], yticklabels=["Low", "Medium", "High"])
        axes[i].set_title(f"{m_name}\n(Acc: {acc*100:.1f}%, Macro F1: {f1_macro:.2f})", fontweight='bold')
        axes[i].set_xlabel("Predicted Label")
        axes[i].set_ylabel("True Label")

    plt.tight_layout()
    cm_plot_path = os.path.join(output_plot_dir, "classical_confusion_matrices.png")
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()

    # Also evaluate models on the REDUCED 4-Feature QML Subset for baseline benchmark!
    qml_subset_results = []
    print("\n" + "=" * 40)
    print("CLASSICAL MODEL BENCHMARK ON REDUCED QML SUBSET (4 FEATURES)")
    print("=" * 40)
    for m_name, model in models.items():
        model.fit(X_train_qml, y_train)
        y_pred_qml = model.predict(X_test_qml)
        acc_q = accuracy_score(y_test, y_pred_qml)
        f1_m_q = f1_score(y_test, y_pred_qml, average='macro', zero_division=0)
        f1_w_q = f1_score(y_test, y_pred_qml, average='weighted', zero_division=0)
        qml_subset_results.append({
            'Model (4-Features)': m_name,
            'Accuracy': acc_q,
            'F1-Score (Macro)': f1_m_q,
            'F1-Score (Weighted)': f1_w_q
        })
        print(f"{m_name:30s} -> Acc: {acc_q*100:.2f}% | Macro F1: {f1_m_q:.4f} | Weighted F1: {f1_w_q:.4f}")

    # -------------------------------------------------------------------------
    # PART 5 — MODEL SELECTION
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print("PART 5: MODEL SELECTION & COMPARISON TABLE")
    print("=" * 40)

    results_df = pd.DataFrame(results).sort_values(by='F1-Score (Macro)', ascending=False)
    print("\nFULL FEATURE SET (12 Features) MODEL COMPARISON TABLE:")
    print(results_df.to_string(index=False))

    qml_results_df = pd.DataFrame(qml_subset_results).sort_values(by='F1-Score (Macro)', ascending=False)
    print("\nREDUCED QML FEATURE SUBSET (4 Features) MODEL COMPARISON TABLE:")
    print(qml_results_df.to_string(index=False))

    best_model_name = results_df.iloc[0]['Model']
    best_f1 = results_df.iloc[0]['F1-Score (Macro)']
    print(f"\n---> BEST CLASSICAL BASELINE MODEL: '{best_model_name}' (Macro F1: {best_f1:.4f})")

    # -------------------------------------------------------------------------
    # PART 6 — SAVE INFORMATION FOR QML
    # -------------------------------------------------------------------------
    qml_export_path = os.path.join(model_save_dir, "qml_feature_subset.pkl")
    qml_metadata = {
        'selected_features': qml_selected_features,
        'selected_indices': qml_feat_indices,
        'scaler_full': scaler_full,
        'best_classical_model_name': best_model_name,
        'results_full': results_df.to_dict(orient='records'),
        'results_qml_subset': qml_results_df.to_dict(orient='records')
    }
    joblib.dump(qml_metadata, qml_export_path)
    print(f"\nExported QML Metadata to: '{qml_export_path}'")
    print("=" * 80)

    return {
        'feat_analysis': feat_analysis,
        'qml_selected_features': qml_selected_features,
        'results_df': results_df,
        'qml_results_df': qml_results_df,
        'best_model': best_model_name
    }

if __name__ == "__main__":
    run_classical_baseline()
