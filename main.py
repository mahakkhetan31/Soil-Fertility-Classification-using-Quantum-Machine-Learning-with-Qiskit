"""
===============================================================================
SOIL QUALITY PREDICTION USING QUANTUM MACHINE LEARNING
Master Execution Script - End-to-End Pipeline
===============================================================================
Run this single file using:
    python main.py

This script sequentially executes all project stages:
1. Stage 1 & 2: Exploratory Data Analysis & Preprocessing
2. Stage 3: Classical ML Baseline & QML Feature Selection
3. Stage 4: Qiskit 4-Qubit Feature Map & Quantum Kernel Setup
4. Stage 5: Quantum Support Vector Classifier (QSVC) Training & Evaluation
5. Stage 6: Classical vs. Quantum Model Comparison & Plots
"""

import sys
import os

# Add src/ to python path
src_dir = os.path.join(os.path.dirname(__file__), "src")
sys.path.append(src_dir)

from eda import run_eda
from preprocessing import run_preprocessing
from classical_ml_baseline import run_classical_baseline
from quantum_kernel_setup import run_quantum_kernel_setup
from qml_classifier_training import run_qml_classifier_stage5
from final_evaluation_and_summary import run_stage6_evaluation

def main():
    print("#" * 80)
    print("      SOIL QUALITY PREDICTION USING QML - MASTER EXECUTION PIPELINE")
    print("#" * 80)

    print("\n[STEP 1/6] Running Exploratory Data Analysis (EDA)...")
    run_eda()

    print("\n[STEP 2/6] Running Data Preprocessing & PCA...")
    run_preprocessing(n_components=6)

    print("\n[STEP 3/6] Training & Evaluating Classical ML Baseline...")
    run_classical_baseline()

    print("\n[STEP 4/6] Constructing Qiskit Feature Map & Quantum Kernel...")
    run_quantum_kernel_setup()

    print("\n[STEP 5/6] Training & Evaluating Quantum Classifier (QSVC)...")
    run_qml_classifier_stage5()

    print("\n[STEP 6/6] Generating Final Model Comparison & Visualizations...")
    run_stage6_evaluation()

    print("\n" + "#" * 80)
    print("SUCCESS: Full Project Pipeline Completed!")
    print("All generated plots & heatmaps are saved in: data/eda_plots/")
    print("#" * 80)

if __name__ == "__main__":
    main()
