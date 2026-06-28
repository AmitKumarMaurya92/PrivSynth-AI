import pandas as pd
import os

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("data/raw/adult.csv")

print("Original Dataset Shape:", df.shape)

# -----------------------------
# Replace '?' with NaN
# -----------------------------
df.replace("?", pd.NA, inplace=True)

# -----------------------------
# Remove Missing Values
# -----------------------------
df.dropna(inplace=True)

print("After Removing Missing Values:", df.shape)

# -----------------------------
# Remove Duplicate Rows
# -----------------------------
duplicates = df.duplicated().sum()
print("Duplicate Rows:", duplicates)

df.drop_duplicates(inplace=True)

print("After Removing Duplicates:", df.shape)

# -----------------------------
# Reset Index
# -----------------------------
df.reset_index(drop=True, inplace=True)

# -----------------------------
# Create processed folder if it doesn't exist
# -----------------------------
os.makedirs("data/processed", exist_ok=True)

# -----------------------------
# Save Clean Dataset
# -----------------------------
df.to_csv("data/processed/adult_clean.csv", index=False)

print("\nClean dataset saved successfully!")
print("Location: data/processed/adult_clean.csv")