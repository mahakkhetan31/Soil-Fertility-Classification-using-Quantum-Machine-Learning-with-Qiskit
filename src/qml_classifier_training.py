"""
===============================================================================
SOIL QUALITY PREDICTION USING QUANTUM MACHINE LEARNING
Stage 5: Quantum Classifier (QSVC) Training, Evaluation, Error Analysis & Disclaimer
===============================================================================
This script executes Parts 1 to 6 of Stage 5:
- Part 1 & 2: QSVC Model Initialization & Training on 4-Qubit Quantum Kernel
- Part 3: Prediction Generation & Soil Fertility Class Name Mapping
- Part 4: Comprehensive Evaluation (Accuracy, Precision, Recall, Macro/Weighted F1, Confusion Matrix)
- Part 5: Error Analysis & Misclassification Investigation
- Part 6: Quantum Experiment Disclaimer & Technical Parameters Specification
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# Set visual style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

def run_qml_classifier_stage5():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    csv_path = os.path.join(base_dir, "dataset1.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(base_dir, "data", "soil_fertility.csv")

    output_plot_dir = os.path.join(base_dir, "data", "eda_plots")
    model_save_dir = os.path.join(base_dir, "data", "models")
    os.makedirs(output_plot_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)

    print("=" * 80)
    print("      STAGE 5: QUANTUM MACHINE LEARNING CLASSIFIER (QSVC) PIPELINE")
    print("=" * 80)

    # 1. Load dataset & final selected 4 features from Stage 3/4
    df = pd.read_csv(csv_path)
    qml_features = ['N', 'P', 'K', 'OC']
    target_col = 'Output'

    X_raw = df[qml_features].values
    y = df[target_col].values

    target_class_names = {
        0: "Less Fertile (Low)",
        1: "Fertile (Medium)",
        2: "Highly Fertile (High)"
    }

    # 2. Stratified 80/20 Train-Test Split (Same split as Classical Baseline)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"\n1. DATA SPLIT & QUANTUM PREPROCESSING:")
    print(f"   - Training Set: {X_train_raw.shape[0]} soil samples")
    print(f"   - Testing Set:  {X_test_raw.shape[0]} soil samples")

    # Fit MinMaxScaler [0, pi] ONLY on Training set (Zero Data Leakage)
    scaler_q = MinMaxScaler(feature_range=(0, np.pi))
    X_train_q = scaler_q.fit_transform(X_train_raw)
    X_test_q = scaler_q.transform(X_test_raw)

    # 3. Construct Qiskit Quantum Feature Map & Quantum Kernel
    import qiskit
    import qiskit_machine_learning
    from qiskit.circuit.library import zz_feature_map
    from qiskit.quantum_info import Statevector

    num_qubits = len(qml_features)
    feature_map = zz_feature_map(feature_dimension=num_qubits, reps=2, entanglement='linear')

    print(f"\n2. QUANTUM ARCHITECTURE SPECIFICATION:")
    print(f"   - Number of Qubits: {num_qubits} ({qml_features})")
    print(f"   - Quantum Feature Map: zz_feature_map (reps=2, entanglement='linear')")
    print(f"   - Qiskit Version: {qiskit.__version__}")
    print(f"   - Qiskit Machine Learning: {qiskit_machine_learning.__version__}")
    print(f"   - Backend Execution: Local Classical Quantum Statevector Simulator")

    # Compute Quantum Fidelity Statevector Kernel Matrices
    print("\n3. COMPUTING QUANTUM STATEVECTOR KERNEL MATRICES:")
    print(f"   - Precomputing Train Kernel Matrix K_train ({len(X_train_q)} x {len(X_train_q)})...")
    
    # Pre-compute statevectors for train and test sets
    sv_train = [Statevector(feature_map.assign_parameters(dict(zip(feature_map.parameters, x)))) for x in X_train_q]
    sv_test  = [Statevector(feature_map.assign_parameters(dict(zip(feature_map.parameters, x)))) for x in X_test_q]

    K_train = np.zeros((len(X_train_q), len(X_train_q)))
    for i in range(len(X_train_q)):
        for j in range(len(X_train_q)):
            K_train[i, j] = np.abs(sv_train[i].inner(sv_train[j])) ** 2

    print(f"   - Precomputing Test Kernel Matrix K_test ({len(X_test_q)} x {len(X_train_q)})...")
    K_test = np.zeros((len(X_test_q), len(X_train_q)))
    for i in range(len(X_test_q)):
        for j in range(len(X_train_q)):
            K_test[i, j] = np.abs(sv_test[i].inner(sv_train[j])) ** 2

    # 4. Train Quantum Kernel Support Vector Classifier (QSVC)
    print("\n4. TRAINING QUANTUM SUPPORT VECTOR CLASSIFIER (QSVC):")
    qsvc = SVC(kernel='precomputed', C=1.0, random_state=42)
    qsvc.fit(K_train, y_train)
    print("   - QSVC Training Completed Successfully!")

    # 5. Generate Predictions
    y_pred_qsvc = qsvc.predict(K_test)

    print("\n5. EXAMPLE PREDICTIONS (FIRST 10 TEST SAMPLES):")
    print("-" * 65)
    print(f"{'Sample #':<10} | {'Actual Class':<22} | {'QSVC Predicted Class':<22}")
    print("-" * 65)
    for idx in range(10):
        act_str = target_class_names[y_test[idx]]
        pred_str = target_class_names[y_pred_qsvc[idx]]
        print(f"Sample {idx+1:<3d}   | {act_str:<22} | {pred_str:<22}")
    print("-" * 65)

    # 6. Evaluation Metrics
    acc = accuracy_score(y_test, y_pred_qsvc)
    prec_macro = precision_score(y_test, y_pred_qsvc, average='macro', zero_division=0)
    rec_macro = recall_score(y_test, y_pred_qsvc, average='macro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred_qsvc, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred_qsvc, average='weighted', zero_division=0)

    print("\n6. QSVC EVALUATION METRICS:")
    print(f"   - Test Accuracy:          {acc * 100:.2f}%")
    print(f"   - Precision (Macro):      {prec_macro:.4f}")
    print(f"   - Recall (Macro):         {rec_macro:.4f}")
    print(f"   - F1-Score (Macro):       {f1_macro:.4f}")
    print(f"   - F1-Score (Weighted):    {f1_weighted:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_qsvc, target_names=["Low (0)", "Medium (1)", "High (2)"], zero_division=0))

    # Confusion Matrix Visualization
    cm = confusion_matrix(y_test, y_pred_qsvc)
    plt.figure(figsize=(7, 5.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=["Low (0)", "Medium (1)", "High (2)"],
                yticklabels=["Low (0)", "Medium (1)", "High (2)"])
    plt.title(f"QSVC Confusion Matrix (4-Qubit Quantum Kernel)\nAccuracy: {acc*100:.2f}%, Macro F1: {f1_macro:.4f}",
              fontsize=12, fontweight='bold', pad=12)
    plt.xlabel("Predicted Soil Fertility Class", fontsize=11)
    plt.ylabel("Actual Soil Fertility Class", fontsize=11)
    plt.tight_layout()

    cm_plot_path = os.path.join(output_plot_dir, "qml_qsvc_confusion_matrix.png")
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()

    # 7. Error Analysis
    print("\n7. ERROR ANALYSIS & MISCLASSIFICATION BREAKDOWN:")
    misclassified_mask = (y_test != y_pred_qsvc)
    num_errors = np.sum(misclassified_mask)
    print(f"   - Total Misclassified Test Samples: {num_errors} out of {len(y_test)} ({num_errors/len(y_test)*100:.2f}%)")

    err_df = pd.DataFrame({
        'Actual': [target_class_names[val] for val in y_test[misclassified_mask]],
        'Predicted': [target_class_names[val] for val in y_pred_qsvc[misclassified_mask]]
    })
    print("\n   Misclassification Matrix (Actual vs Predicted Errors):")
    print(pd.crosstab(err_df['Actual'], err_df['Predicted']))

    # 8. Export Stage 5 Results
    joblib.dump({
        'qsvc': qsvc,
        'acc': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'cm': cm,
        'y_test': y_test,
        'y_pred_qsvc': y_pred_qsvc
    }, os.path.join(model_save_dir, "qml_qsvc_model.pkl"))

    print(f"\nSaved Confusion Matrix Plot to: '{cm_plot_path}'")
    print(f"Exported QML Model & Metrics to: '{os.path.join(model_save_dir, 'qml_qsvc_model.pkl')}'")
    print("=" * 80)

    return {
        'acc': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'cm': cm,
        'num_errors': num_errors
    }

if __name__ == "__main__":
    run_qml_classifier_stage5()
