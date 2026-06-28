import os
import pandas as pd
from ctgan import CTGAN

# -----------------------------------
# Load Clean Dataset
# -----------------------------------
df = pd.read_csv("data/processed/adult_clean.csv")

print("=" * 60)
print("Adult Census Dataset Loaded Successfully")
print("=" * 60)

print(f"Dataset Shape : {df.shape}")

# -----------------------------------
# Define Categorical Columns
# -----------------------------------
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

# -----------------------------------
# Create CTGAN Model
# -----------------------------------
model = CTGAN(
    epochs=200,
    batch_size=500,
    verbose=True
)

print("\nTraining Started...\n")

# -----------------------------------
# Train Model
# -----------------------------------
model.fit(
    df,
    discrete_columns=categorical_columns
)

print("\nTraining Completed!")

# -----------------------------------
# Create models folder
# -----------------------------------
os.makedirs("models", exist_ok=True)

# -----------------------------------
# Save Model
# -----------------------------------
model.save("models/baseline_ctgan.pkl")

print("\nModel Saved Successfully!")
print("Location : models/baseline_ctgan.pkl")