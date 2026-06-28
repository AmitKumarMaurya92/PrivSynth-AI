import os
import pandas as pd
import pickle

# Ensure output directory exists
os.makedirs("data/synthetic", exist_ok=True)

# Load original data to know how many rows to generate
try:
    original_df = pd.read_csv("data/processed/adult_clean.csv")
except FileNotFoundError:
    original_df = pd.read_csv("Data/processed/adult_clean.csv")

num_rows = len(original_df)
print("=" * 60)
print(f"Original dataset loaded. Number of rows to generate: {num_rows}")
print("=" * 60)

# Define the privacy budgets and corresponding file names
epsilons = {
    0.5: "epsilon_05.csv",
    1.0: "epsilon_10.csv",
    5.0: "epsilon_50.csv"
}

# Generate synthetic data for each epsilon
for eps, output_filename in epsilons.items():
    model_filename = f"models/dp_ctgan_eps{str(eps).replace('.', '')}.pkl"
    output_path = os.path.join("data", "synthetic", output_filename)
    
    print(f"\n--- Processing Epsilon = {eps} ---")
    
    # Load model
    try:
        with open(model_filename, "rb") as f:
            dp_gan = pickle.load(f)
        print(f"Successfully loaded model from {model_filename}")
        
        # Generate data
        print(f"Generating {num_rows} synthetic rows...")
        synthetic_data = dp_gan.sample(num_rows)
        
        # Save CSV
        synthetic_data.to_csv(output_path, index=False)
        print(f"Saved generated data to {output_path}")
        
    except FileNotFoundError:
        print(f"Error: Model file {model_filename} not found.")
    except Exception as e:
        print(f"Error generating data for epsilon {eps}: {e}")

print("\nData generation for all privacy budgets complete!")
