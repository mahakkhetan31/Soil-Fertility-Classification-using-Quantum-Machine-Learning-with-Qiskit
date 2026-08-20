import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import joblib

def run_preprocessing(n_components=6, random_state=42):
    """
    Stage 2: Data Preprocessing & Dimensionality Reduction Pipeline.
    1. Loads dataset1.csv
    2. Performs Stratified Train-Test Split (80/20)
    3. Fits MinMaxScaler to scale feature values into [0, pi] for Qiskit angle encoding
    4. Applies PCA to reduce 12 features to `n_components` (default: 6) for quantum execution
    5. Saves preprocessed arrays and scaler/PCA objects into `data/preprocessed/`
    """
    base_dir = os.path.dirname(os.path.dirname(__file__))
    csv_path = os.path.join(base_dir, "dataset1.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(base_dir, "data", "soil_fertility.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    save_dir = os.path.join(base_dir, "data", "preprocessed")
    os.makedirs(save_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    feature_names = list(df.columns[:-1])

    print("=" * 65)
    print("STAGE 2: PREPROCESSING & DIMENSIONALITY REDUCTION (PCA)")
    print("=" * 65)
    print(f"Original Feature Matrix Shape: {X.shape}")
    print(f"Target Array Shape: {y.shape}")

    # 1. Train-Test Split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )
    print(f"\n1. TRAIN-TEST SPLIT (80% Train / 20% Test):")
    print(f"   - Training Set: {X_train_raw.shape[0]} samples")
    print(f"   - Testing Set:  {X_test_raw.shape[0]} samples")

    # 2. MinMaxScaler to [0, pi] for Quantum Angle Encoding
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    print(f"\n2. FEATURE SCALING:")
    print(f"   - MinMaxScaler feature range: [0, pi] (0 to {np.pi:.4f} radians)")

    # 3. Principal Component Analysis (PCA) for QML
    pca = PCA(n_components=n_components, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    var_ratio = pca.explained_variance_ratio_
    cum_var = np.cumsum(var_ratio)

    print(f"\n3. DIMENSIONALITY REDUCTION (PCA for QML):")
    print(f"   - Target Qubits / Components: {n_components}")
    print(f"   - Variance Explained per Component:")
    for i, ratio in enumerate(var_ratio, 1):
        print(f"     * PC{i}: {ratio * 100:.2f}% (Cumulative: {cum_var[i-1] * 100:.2f}%)")

    # 4. Save Artifacts
    np.save(os.path.join(save_dir, "X_train_full.npy"), X_train_scaled)
    np.save(os.path.join(save_dir, "X_test_full.npy"), X_test_scaled)
    np.save(os.path.join(save_dir, "X_train_pca.npy"), X_train_pca)
    np.save(os.path.join(save_dir, "X_test_pca.npy"), X_test_pca)
    np.save(os.path.join(save_dir, "y_train.npy"), y_train)
    np.save(os.path.join(save_dir, "y_test.npy"), y_test)

    joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))
    joblib.dump(pca, os.path.join(save_dir, "pca.pkl"))

    print(f"\n4. SAVED PREPROCESSED ARRAYS & MODELS:")
    print(f"   - Directory: {save_dir}")
    print(f"   - Full Scaled Features (12 features): X_train_full.npy, X_test_full.npy")
    print(f"   - PCA Features ({n_components} qubits/features): X_train_pca.npy, X_test_pca.npy")
    print(f"   - Target Arrays: y_train.npy, y_test.npy")
    print("=" * 65)

    return {
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'X_train_pca': X_train_pca,
        'X_test_pca': X_test_pca,
        'y_train': y_train,
        'y_test': y_test,
        'cum_variance': cum_var[-1]
    }

if __name__ == "__main__":
    run_preprocessing(n_components=6)
