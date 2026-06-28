import os
import pandas as pd
from ctgan import CTGAN

# -------------------------------------
# Create Output Folder
# -------------------------------------
os.makedirs("Data/synthetic", exist_ok=True)

# -------------------------------------
# Load Original Dataset
# -------------------------------------
real_data = pd.read_csv("Data/processed/adult_clean.csv")

print("=" * 60)
print("Original Dataset Loaded")
print("=" * 60)

print(f"Rows : {len(real_data)}")
print(f"Columns : {len(real_data.columns)}")

# -------------------------------------
# Load Trained CTGAN Model
# -------------------------------------
model = CTGAN.load("models/baseline_ctgan.pkl")

print("\nCTGAN Model Loaded Successfully")

# -------------------------------------
# Generate Synthetic Data
# -------------------------------------
num_rows = len(real_data)

synthetic_data = model.sample(num_rows)

print("\nSynthetic Data Generated Successfully")

# -------------------------------------
# Save Synthetic Dataset
# -------------------------------------
output_path = "Data/synthetic/baseline.csv"

synthetic_data.to_csv(output_path, index=False)

print("\nSynthetic Dataset Saved Successfully")
print(f"Location : {output_path}")

print("\nFirst Five Rows")
print(synthetic_data.head())