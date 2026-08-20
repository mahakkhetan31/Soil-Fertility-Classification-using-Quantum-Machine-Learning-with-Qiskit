import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    
    # Check for dataset1.csv or data/soil_fertility.csv
    csv_path = os.path.join(base_dir, "dataset1.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(base_dir, "data", "soil_fertility.csv")

    output_plot_dir = os.path.join(base_dir, "data", "eda_plots")
    os.makedirs(output_plot_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found at {csv_path}.")

    print("=" * 65)
    print(f"STAGE 1: EXPLORATORY DATA ANALYSIS (EDA) - {os.path.basename(csv_path)}")
    print("=" * 65)

    df = pd.read_csv(csv_path)

    # 1. Dataset Overview
    print(f"\n1. DATASET DIMENSIONS:")
    print(f"   - Total Samples (Rows): {df.shape[0]}")
    print(f"   - Total Features (Columns): {df.shape[1] - 1} features + 1 target")
    print(f"   - Feature List: {list(df.columns[:-1])}")
    print(f"   - Target Column: '{df.columns[-1]}'")

    # 2. Missing & Duplicates
    print(f"\n2. DATA INTEGRITY & MISSING VALUES:")
    missing_count = df.isnull().sum().sum()
    dup_count = df.duplicated().sum()
    print(f"   - Total Missing Values: {missing_count}")
    print(f"   - Total Duplicate Rows: {dup_count}")

    # 3. Class Distribution
    target_col = df.columns[-1]
    class_counts = df[target_col].value_counts().sort_index()
    class_pct = (class_counts / len(df)) * 100
    print(f"\n3. TARGET CLASS DISTRIBUTION ('{target_col}'):")
    class_names = {0: "Less Fertile (Low)", 1: "Fertile (Medium)", 2: "Highly Fertile (High)"}
    for cls, count in class_counts.items():
        name = class_names.get(cls, f"Class {cls}")
        print(f"   - Class {cls} [{name}]: {count} samples ({class_pct[cls]:.2f}%)")

    # 4. Statistical Summary
    print(f"\n4. DESCRIPTIVE STATISTICS OF FEATURES:")
    stats_df = df.describe().T[['mean', 'std', 'min', '50%', 'max']]
    stats_df.columns = ['Mean', 'Std Dev', 'Min', 'Median (50%)', 'Max']
    print(stats_df.round(2).to_string())

    # 5. Correlation Analysis
    corr = df.corr()
    target_corr = corr[target_col].drop(target_col).sort_values(ascending=False)
    print(f"\n5. FEATURE CORRELATION WITH TARGET ('{target_col}'):")
    for feat, val in target_corr.items():
        print(f"   - {feat:5s}: {val:+.4f}")

    # Save Plots
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # Plot 1: Target Distribution
    plt.figure(figsize=(7, 5))
    palette = ["#e74c3c", "#f39c12", "#2ecc71"]
    ax = sns.barplot(x=class_counts.index, y=class_counts.values, palette=palette)
    plt.title("Soil Fertility Target Class Distribution", fontsize=13, fontweight='bold')
    plt.xlabel("Fertility Class (0: Low, 1: Medium, 2: High)", fontsize=11)
    plt.ylabel("Number of Samples", fontsize=11)
    plt.xticks(ticks=[0, 1, 2], labels=["Low (0)", "Medium (1)", "High (2)"])
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 6), textcoords='offset points', fontweight='bold')
    plt.tight_layout()
    plot1_path = os.path.join(output_plot_dir, "class_distribution.png")
    plt.savefig(plot1_path, dpi=300)
    plt.close()

    # Plot 2: Correlation Matrix Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5)
    plt.title("Soil Properties Correlation Matrix", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plot2_path = os.path.join(output_plot_dir, "correlation_heatmap.png")
    plt.savefig(plot2_path, dpi=300)
    plt.close()

    # Plot 3: Feature Histograms
    feature_cols = list(df.columns[:-1])
    plt.figure(figsize=(12, 10))
    for i, col in enumerate(feature_cols, 1):
        plt.subplot(4, 3, i)
        sns.histplot(df[col], kde=True, color='#3498db', bins=20)
        plt.title(f"{col} Distribution", fontsize=10, fontweight='bold')
        plt.xlabel("")
        plt.ylabel("")
    plt.tight_layout()
    plot3_path = os.path.join(output_plot_dir, "feature_distributions.png")
    plt.savefig(plot3_path, dpi=300)
    plt.close()

    print(f"\nGenerated Plots Saved:")
    print(f"   - {plot1_path}")
    print(f"   - {plot2_path}")
    print(f"   - {plot3_path}")
    print("=" * 65)

    return df

if __name__ == "__main__":
    run_eda()
