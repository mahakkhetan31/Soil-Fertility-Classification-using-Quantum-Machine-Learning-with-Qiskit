import os
import pandas as pd

def inspect_dataset():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    possible_paths = [
        os.path.join(base_dir, "dataset1.csv"),
        os.path.join(base_dir, "data", "soil_fertility.csv")
    ]

    csv_path = None
    for p in possible_paths:
        if os.path.exists(p):
            csv_path = p
            break

    if not csv_path:
        print("Error: Could not locate dataset file.")
        return

    df = pd.read_csv(csv_path)
    print("=" * 60)
    print(f"INSPECTING UPLOADED DATASET: {os.path.basename(csv_path)}")
    print("=" * 60)
    print(f"Dimensions: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst 5 Rows:")
    print(df.head())
    print("\nData Types & Missing Values:")
    print(df.info())
    print("\nMissing Values Count:")
    print(df.isnull().sum())

    target_col = [c for c in df.columns if c.lower() in ['output', 'target', 'fertility', 'class'] or c == df.columns[-1]][0]
    print(f"\nTarget Column Detected: '{target_col}'")
    print("Class Value Counts:")
    print(df[target_col].value_counts())
    print("=" * 60)

if __name__ == "__main__":
    inspect_dataset()
