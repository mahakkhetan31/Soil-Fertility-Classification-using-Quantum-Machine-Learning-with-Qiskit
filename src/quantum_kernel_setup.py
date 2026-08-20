"""
===============================================================================
SOIL QUALITY PREDICTION USING QUANTUM MACHINE LEARNING
Stage 4: Quantum Feature Map & Quantum Kernel Construction (Qiskit 2.5.2 Compliant)
===============================================================================
This script executes Parts 1 to 4 of Stage 4:
- Part 1: Quantum Data Preparation (4 selected features scaled to [0, pi])
- Part 2: Qiskit 4-Qubit zz_feature_map Implementation & Circuit Representation
- Part 3: Quantum Kernel Matrix Computation via Qiskit Quantum Statevectors
- Part 4: Dimensionality Validation & Heatmap Visualization (Saved to data/eda_plots/)
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def run_quantum_kernel_setup():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    csv_path = os.path.join(base_dir, "dataset1.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(base_dir, "data", "soil_fertility.csv")

    output_plot_dir = os.path.join(base_dir, "data", "eda_plots")
    model_save_dir = os.path.join(base_dir, "data", "models")
    os.makedirs(output_plot_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)

    print("=" * 80)
    print("      STAGE 4: QISKIT QUANTUM FEATURE MAP & QUANTUM KERNEL PIPELINE")
    print("=" * 80)

    # 1. Environment & Package Verification
    import qiskit
    import qiskit_machine_learning
    import qiskit_algorithms
    from qiskit.quantum_info import Statevector
    from qiskit.circuit.library import zz_feature_map

    print("\n1. VERIFIED QUANTUM SOFTWARE VERSIONS:")
    print(f"   - Qiskit Version:                  {qiskit.__version__}")
    print(f"   - Qiskit Machine Learning Version: {qiskit_machine_learning.__version__}")
    print(f"   - Qiskit Algorithms Version:       {qiskit_algorithms.__version__}")
    print(f"   - Backend: Local Classical Quantum Statevector Simulator")

    # Load dataset & top 4 selected features from Stage 3
    df = pd.read_csv(csv_path)
    qml_features = ['N', 'P', 'K', 'OC']
    num_qubits = len(qml_features)

    X_raw = df[qml_features].values
    y = df['Output'].values

    # Stratified Train-Test Split (80/20)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.20, random_state=42, stratify=y
    )

    # Scale to [0, pi] for quantum phase angle encoding (Fitted strictly on X_train)
    q_scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train_q = q_scaler.fit_transform(X_train_raw)
    X_test_q = q_scaler.transform(X_test_raw)

    print(f"\n2. QUANTUM DATA PREPARATION ({num_qubits} Qubits):")
    print(f"   - Selected Features: {qml_features}")
    print(f"   - Qubit Mapping:")
    for i, f_name in enumerate(qml_features):
        print(f"     * Qubit q[{i}] <---> Feature '{f_name}'")
    print(f"   - Scaling: MinMaxScaler [0, pi] radians (Min: {X_train_q.min():.4f}, Max: {X_train_q.max():.4f})")

    # -------------------------------------------------------------------------
    # PART 2 — QUANTUM FEATURE MAP (zz_feature_map)
    # -------------------------------------------------------------------------
    feature_map = zz_feature_map(feature_dimension=num_qubits, reps=2, entanglement='linear')

    print(f"\n3. QUANTUM FEATURE MAP CONSTRUCTED:")
    print(f"   - Type: Second-order Pauli Z-Z Entangling Feature Map (zz_feature_map)")
    print(f"   - Number of Qubits: {feature_map.num_qubits}")
    print(f"   - Circuit Depth: {feature_map.depth()}")
    print(f"   - Entanglement Architecture: linear")
    print(f"   - Circuit Repetitions (Layers): 2")

    # -------------------------------------------------------------------------
    # PART 3 — QUANTUM KERNEL MATRIX COMPUTATION
    # -------------------------------------------------------------------------
    print("\n4. COMPUTING QUANTUM KERNEL MATRIX VIA STATEVECTOR OVERLAPS:")
    sample_size = min(40, len(X_train_q))
    X_train_sub = X_train_q[:sample_size]

    # Pre-compute statevectors for efficiency
    statevectors = []
    for sample in X_train_sub:
        bound_circuit = feature_map.assign_parameters(sample)
        sv = Statevector(bound_circuit)
        statevectors.append(sv)

    # Compute pairwise fidelity kernel: K(i, j) = |<psi(x_i) | psi(x_j)>|^2
    kernel_matrix_train = np.zeros((sample_size, sample_size))
    for i in range(sample_size):
        for j in range(sample_size):
            fidelity = np.abs(statevectors[i].inner(statevectors[j])) ** 2
            kernel_matrix_train[i, j] = fidelity

    # -------------------------------------------------------------------------
    # PART 4 — VALIDATION & VISUALIZATION
    # -------------------------------------------------------------------------
    print("\n5. KERNEL MATRIX VALIDATION:")
    print(f"   - Matrix Shape: {kernel_matrix_train.shape} (Expected: ({sample_size}, {sample_size}))")
    print(f"   - Contains NaN / Inf: {np.isnan(kernel_matrix_train).any() or np.isinf(kernel_matrix_train).any()}")
    print(f"   - Diagonal Min/Max: {np.diag(kernel_matrix_train).min():.4f} / {np.diag(kernel_matrix_train).max():.4f} (Expected self-overlap: 1.0)")
    print(f"   - Matrix Symmetry Check: {np.allclose(kernel_matrix_train, kernel_matrix_train.T)}")

    # Heatmap Visualization
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(kernel_matrix_train, cmap='viridis', vmin=0, vmax=1, cbar_kws={'label': 'Quantum Fidelity Overlap |<ψ(xi)|ψ(xj)>|²'})
    plt.title("Fidelity Quantum Kernel Matrix Heatmap (4-Qubit Soil State Vectors)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Soil Sample Index i", fontsize=11)
    plt.ylabel("Soil Sample Index j", fontsize=11)
    plt.tight_layout()
    
    heatmap_path = os.path.join(output_plot_dir, "quantum_kernel_matrix_heatmap.png")
    plt.savefig(heatmap_path, dpi=300)
    plt.close()

    # Export Stage 4 Metadata
    qml_stage4_path = os.path.join(model_save_dir, "quantum_kernel_data.pkl")
    joblib.dump({
        'num_qubits': num_qubits,
        'qml_features': qml_features,
        'X_train_q': X_train_q,
        'X_test_q': X_test_q,
        'y_train': y_train,
        'y_test': y_test,
        'q_scaler': q_scaler,
        'feature_map': feature_map,
        'kernel_matrix': kernel_matrix_train
    }, qml_stage4_path)

    print(f"\nSaved Quantum Kernel Heatmap to: '{heatmap_path}'")
    print(f"Exported Stage 4 Quantum Metadata to: '{qml_stage4_path}'")
    print("=" * 80)

    return {
        'num_qubits': num_qubits,
        'features': qml_features,
        'kernel_matrix': kernel_matrix_train,
        'heatmap_path': heatmap_path
    }

if __name__ == "__main__":
    run_quantum_kernel_setup()
