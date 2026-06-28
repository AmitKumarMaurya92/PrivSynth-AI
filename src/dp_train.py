import os
import pandas as pd
import pickle

try:
    from snsynth.pytorch.nn import DPCTGAN
    from snsynth.pytorch import PytorchDPSynthesizer
except ImportError:
    print("Please install snsynth first: pip install snsynth")
    exit()

# -----------------------------------
# Load Clean Dataset
# -----------------------------------
df = pd.read_csv("data/processed/adult_clean.csv")

print("=" * 60)
print("Dataset Loaded for DP Training")
print("=" * 60)

# Define Categorical Columns
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

# Define Continuous Columns
continuous_columns = [
    "age", 
    "fnlwgt", 
    "educational-num", 
    "capital-gain", 
    "capital-loss", 
    "hours-per-week"
]

# Privacy Budgets (Epsilons)
epsilons = [0.5, 1.0, 5.0]

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# -----------------------------------
# Train Models with Different Epsilons
# -----------------------------------
for eps in epsilons:
    print(f"\n--- Training DP-CTGAN with Epsilon = {eps} ---")
    
    # Initialize DP-CTGAN model using SmartNoise Synthesizers
    # This correctly integrates Opacus DP-SGD with CTGAN
    dp_gan = PytorchDPSynthesizer(
        epsilon=eps,
        gan=DPCTGAN(epochs=100) # Using 100 epochs instead of default to save time
    )
    
    # Train the model
    dp_gan.fit(
        df,
        categorical_columns=categorical_columns,
        continuous_columns=continuous_columns,
        preprocessor_eps=0.1
    )
    
    # Create filename (e.g., dp_ctgan_eps05.pkl)
    filename = f"models/dp_ctgan_eps{str(eps).replace('.', '')}.pkl"
    
    # Save the model
    with open(filename, "wb") as f:
        pickle.dump(dp_gan, f)
    
    print(f"Model saved successfully to {filename}")

print("\nAll DP Models trained and saved!")
