import pandas as pd

# -------------------------
# Load Clean Dataset
# -------------------------
df = pd.read_csv("data/processed/adult_clean.csv")

print("=" * 50)
print("Dataset Loaded Successfully")
print("=" * 50)

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

# -------------------------
# Define Categorical Columns
# -------------------------
categorical_columns = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "gender",
    "native-country",
    "income"
]

# -------------------------
# Find Numerical Columns
# -------------------------
numerical_columns = [
    col for col in df.columns
    if col not in categorical_columns
]

print("\nCategorical Columns")
print(categorical_columns)

print("\nNumerical Columns")
print(numerical_columns)

print("\nData Types")
print(df.dtypes)

print("\nDataset is ready for CTGAN training!")