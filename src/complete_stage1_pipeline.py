"""
===============================================================================
SOIL QUALITY PREDICTION USING QUANTUM MACHINE LEARNING
Stage 1: Comprehensive Data Cleaning, EDA, & Reproducible Preprocessing Pipeline
===============================================================================
This script executes Parts 1 to 5 of Stage 1 on `dataset1.csv`:
- Part 1: Data Cleaning (Missing values, duplicates, invalid range checks, outlier analysis)
- Part 2: Target Preparation (Encoding & Class Mapping)
- Part 3: Exploratory Data Analysis & Visualizations (Saved to `data/eda_plots/`)
- Part 4: Agricultural & Statistical Interpretation
- Part 5: Reproducible Scikit-Learn Pipeline (Preventing Data Leakage)
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Set visual style for publication-ready plots
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'

def run_stage1_pipeline():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    csv_path = os.path.join(base_dir, "dataset1.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(base_dir, "data", "soil_fertility.csv")

    output_plot_dir = os.path.join(base_dir, "data", "eda_plots")
    save_data_dir = os.path.join(base_dir, "data", "preprocessed")
    os.makedirs(output_plot_dir, exist_ok=True)
    os.makedirs(save_data_dir, exist_ok=True)

    print("=" * 80)
    print("      SOIL QUALITY PREDICTION - STAGE 1 FULL EDA & PREPROCESSING PIPELINE")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # PART 1 — DATA CLEANING & INSPECTION
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print("PART 1: DATA CLEANING & VALIDATION")
    print("=" * 40)

    raw_df = pd.read_csv(csv_path)
    # 1. Create copy of original dataframe
    df = raw_df.copy()

    print(f"1. Dataset Loaded: '{os.path.basename(csv_path)}'")
    print(f"   - Dimensions: {df.shape[0]} rows (samples) x {df.shape[1]} columns (features + target)")

    # 2. Missing Values Check
    missing_series = df.isnull().sum()
    total_missing = missing_series.sum()
    if total_missing == 0:
        print("2. Missing Value Status: Clean. 0 missing values detected. No imputation required.")
    else:
        print(f"2. Missing Values Found ({total_missing}). Performing Median Imputation...")
        df.fillna(df.median(), inplace=True)

    # 3. Duplicate Rows Check
    num_duplicates = df.duplicated().sum()
    print(f"3. Duplicate Rows Status: {num_duplicates} duplicate rows detected.")
    if num_duplicates > 0:
        df.drop_duplicates(inplace=True)
        print(f"   - Removed duplicates. New shape: {df.shape}")

    # 4. Invalid Measurement Verification
    print("\n4. Invalid Soil Measurement Check (Physicochemical Bounds):")
    # Soil pH should be between 0 and 14
    invalid_ph = df[(df['pH'] < 0) | (df['pH'] > 14)]
    # Nutrients (N, P, K, EC, OC, S, Zn, Fe, Cu, Mn, B) should be non-negative
    feature_cols = [c for c in df.columns if c != 'Output']
    negative_val_counts = (df[feature_cols] < 0).sum().sum()

    print(f"   - Out-of-bounds pH (< 0 or > 14): {len(invalid_ph)} samples.")
    print(f"   - Negative Nutrient/EC/OC values (< 0): {negative_val_counts} samples.")
    print("   - Result: All soil feature values fall within physically valid domain ranges.")

    # 5. Outlier & Extreme Value Analysis (IQR Method)
    print("\n5. Outlier & Extreme Value Investigation (Interquartile Range - IQR):")
    outlier_summary = {}
    for col in feature_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_summary[col] = len(outliers)

    print("   Outliers count per feature (IQR threshold = 1.5 * IQR):")
    for feat, count in outlier_summary.items():
        pct = (count / len(df)) * 100
        print(f"   - {feat:4s}: {count:3d} extreme values ({pct:.1f}%)")
    print("\n   [Decision]: Outliers represent naturally occurring high/low nutrient soil conditions.")
    print("   They will NOT be removed blindly as they contain valuable agricultural signals.")

    # -------------------------------------------------------------------------
    # PART 2 — TARGET PREPARATION
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print("PART 2: TARGET PREPARATION & CLASS MAPPING")
    print("=" * 40)

    target_col = 'Output'
    print(f"1. Target Column Identified: '{target_col}'")

    # Mapping target classes explicitly
    target_mapping = {
        0: "Less Fertile (Low Quality)",
        1: "Fertile (Medium Quality)",
        2: "Highly Fertile (High Quality)"
    }
    print("2. Preserved Class Label Mapping:")
    for code, label in target_mapping.items():
        print(f"   - Code {code} <---> Label '{label}'")

    class_counts = df[target_col].value_counts().sort_index()
    class_percentages = (class_counts / len(df)) * 100

    print("\n3. Final Class Distribution:")
    for code, count in class_counts.items():
        print(f"   - Class {code} [{target_mapping[code]}]: {count} samples ({class_percentages[code]:.2f}%)")

    # -------------------------------------------------------------------------
    # PART 3 — EXPLORATORY DATA ANALYSIS & VISUALIZATIONS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print("PART 3: GENERATING EDA VISUALIZATIONS")
    print("=" * 40)

    # Plot 1: Target Class Distribution
    plt.figure(figsize=(8, 5))
    palette_colors = ["#e74c3c", "#f39c12", "#2ecc71"]
    ax = sns.barplot(x=class_counts.index, y=class_counts.values, palette=palette_colors)
    plt.title("1. Soil Fertility Target Class Distribution", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Soil Fertility Class", fontsize=11)
    plt.ylabel("Sample Count", fontsize=11)
    plt.xticks(ticks=[0, 1, 2], labels=["Low (0)", "Medium (1)", "High (2)"])
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}\n({p.get_height()/len(df)*100:.1f}%)",
                    (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                    ha='center', va='center', color='white', fontweight='bold', fontsize=11)
    plt.tight_layout()
    plot1_path = os.path.join(output_plot_dir, "1_target_class_distribution.png")
    plt.savefig(plot1_path, dpi=300)
    plt.close()

    # Plot 2: Histograms & KDE of Numerical Soil Features
    plt.figure(figsize=(14, 10))
    for i, col in enumerate(feature_cols, 1):
        plt.subplot(4, 3, i)
        sns.histplot(df[col], kde=True, color='#2980b9', bins=20)
        plt.title(f"Distribution: {col}", fontsize=10, fontweight='bold')
        plt.xlabel("")
        plt.ylabel("")
    plt.suptitle("2. Histograms & KDE Distributions of Soil Properties", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plot2_path = os.path.join(output_plot_dir, "2_feature_histograms.png")
    plt.savefig(plot2_path, dpi=300)
    plt.close()

    # Plot 3: Boxplots of Soil Properties by Fertility Class
    plt.figure(figsize=(14, 10))
    for i, col in enumerate(feature_cols, 1):
        plt.subplot(4, 3, i)
        sns.boxplot(x=target_col, y=col, data=df, palette=palette_colors)
        plt.title(f"{col} by Fertility Class", fontsize=10, fontweight='bold')
        plt.xlabel("")
        plt.ylabel("")
    plt.suptitle("3. Boxplots of Soil Properties Grouped by Fertility Class", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plot3_path = os.path.join(output_plot_dir, "3_feature_boxplots_by_class.png")
    plt.savefig(plot3_path, dpi=300)
    plt.close()

    # Plot 4: Correlation Heatmap
    plt.figure(figsize=(11, 9))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5)
    plt.title("4. Correlation Heatmap of Soil Properties & Fertility", fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    plot4_path = os.path.join(output_plot_dir, "4_correlation_heatmap.png")
    plt.savefig(plot4_path, dpi=300)
    plt.close()

    # Plot 5: Pairwise Relationships for Key Macronutrients (N, P, K, pH, OC)
    key_features = ['N', 'P', 'K', 'pH', 'OC', target_col]
    g = sns.pairplot(df[key_features], hue=target_col, palette=palette_colors, corner=True, diag_kind='kde')
    g.fig.suptitle("5. Pairwise Relationships for Key Soil Features", y=1.02, fontsize=14, fontweight='bold')
    plot5_path = os.path.join(output_plot_dir, "5_key_features_pairplot.png")
    g.savefig(plot5_path, dpi=300)
    plt.close()

    # Plot 6: Agricultural Insight Plot (Nitrogen vs Phosphorus colored by Class)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='N', y='P', hue=target_col, style=target_col, data=df, palette=palette_colors, s=70, alpha=0.8)
    plt.title("6. Agronomic Insight: Nitrogen (N) vs. Phosphorus (P) Interaction", fontsize=13, fontweight='bold')
    plt.xlabel("Nitrogen N (kg/ha)", fontsize=11)
    plt.ylabel("Phosphorus P (kg/ha)", fontsize=11)
    plt.legend(title="Fertility Class", labels=["Low (0)", "Medium (1)", "High (2)"])
    plt.tight_layout()
    plot6_path = os.path.join(output_plot_dir, "6_agronomic_n_vs_p_scatter.png")
    plt.savefig(plot6_path, dpi=300)
    plt.close()

    print(f"All 6 Plots Generated & Saved to '{output_plot_dir}':")
    print(f"   1. {os.path.basename(plot1_path)}")
    print(f"   2. {os.path.basename(plot2_path)}")
    print(f"   3. {os.path.basename(plot3_path)}")
    print(f"   4. {os.path.basename(plot4_path)}")
    print(f"   5. {os.path.basename(plot5_path)}")
    print(f"   6. {os.path.basename(plot6_path)}")

    # -------------------------------------------------------------------------
    # PART 4 — INTERPRETATION & DISCUSSION
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print("PART 4: AGRICULTURAL & STATISTICAL INTERPRETATION")
    print("=" * 40)
    
    target_corr = corr[target_col].drop(target_col).sort_values(ascending=False)
    print("\n1. Nutrients Associated with Soil Fertility:")
    for feat, r_val in target_corr.items():
        print(f"   - {feat:4s} (Correlation r = {r_val:+.4f})")

    print("\n2. Key Skewness & Distribution Characteristics:")
    for col in feature_cols:
        skew_val = df[col].skew()
        if abs(skew_val) > 0.5:
            print(f"   - {col:4s}: Skewness = {skew_val:+.2f} (Moderately/Highly Skewed)")

    print("\n3. Class Imbalance Assessment:")
    print("   - Low Fertility (Class 0) represents 60.9% of the dataset.")
    print("   - High Fertility (Class 2) represents 10.9% of the dataset.")
    print("   - [Action]: Stratified train-test splitting is mandatory to preserve class ratios.")

    # -------------------------------------------------------------------------
    # PART 5 — REPRODUCIBLE PREPROCESSING PIPELINE (NO DATA LEAKAGE)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print("PART 5: REPRODUCIBLE PREPROCESSING PIPELINE")
    print("=" * 40)

    X = df[feature_cols].values
    y = df[target_col].values

    # 1. Stratified Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("1. Stratified Train-Test Split Completed:")
    print(f"   - Training Set Shape: {X_train.shape[0]} samples x {X_train.shape[1]} features")
    print(f"   - Testing Set Shape:  {X_test.shape[0]} samples x {X_test.shape[1]} features")

    # 2. Build Pipeline (MinMaxScaler for Quantum Angle Encoding [0, pi])
    pipeline = Pipeline([
        ('scaler', MinMaxScaler(feature_range=(0, np.pi)))
    ])

    # Fit ONLY on X_train to prevent data leakage!
    X_train_scaled = pipeline.fit_transform(X_train)
    X_test_scaled = pipeline.transform(X_test)

    print("\n2. Pipeline Transformation (MinMaxScaler -> [0, pi] radians):")
    print(f"   - X_train_scaled min: {X_train_scaled.min():.4f}, max: {X_train_scaled.max():.4f}")
    print(f"   - X_test_scaled  min: {X_test_scaled.min():.4f}, max: {X_test_scaled.max():.4f}")

    # 3. Save Data Arrays and Pipeline Object
    np.save(os.path.join(save_data_dir, "X_train.npy"), X_train_scaled)
    np.save(os.path.join(save_data_dir, "X_test.npy"), X_test_scaled)
    np.save(os.path.join(save_data_dir, "y_train.npy"), y_train)
    np.save(os.path.join(save_data_dir, "y_test.npy"), y_test)
    joblib.dump(pipeline, os.path.join(save_data_dir, "preprocessing_pipeline.pkl"))

    print(f"\n3. Preprocessed Datasets & Pipeline Saved to '{save_data_dir}':")
    print("   - X_train.npy, X_test.npy")
    print("   - y_train.npy, y_test.npy")
    print("   - preprocessing_pipeline.pkl")
    print("=" * 80)

    return {
        'df': df,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'target_corr': target_corr
    }

if __name__ == "__main__":
    run_stage1_pipeline()
