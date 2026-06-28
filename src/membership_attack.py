import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import pairwise_distances

# -------------------------------------
# Setup and Data Loading
# -------------------------------------
os.makedirs("results", exist_ok=True)

print("=" * 60)
print("Starting Membership Inference Attack Evaluation")
print("=" * 60)

# Load real dataset
real_data = pd.read_csv("data/processed/adult_clean.csv")

# We simulate Members and Non-Members
# Members: 1000 real records used during training
sample_size = 1000
members = real_data.sample(sample_size, random_state=42).copy()
members['is_member'] = 1

# Non-Members: 1000 records that are structurally similar but not in the training set.
# Since we didn't hold out data, we simulate non-members by taking another sample
# and perturbing the relationships (e.g., shuffling a column, adding noise) 
# to represent out-of-distribution records.
non_members = real_data.sample(sample_size, random_state=99).copy()
# Perturb to ensure they are distinctly "non-members" of the exact joint distribution
non_members['age'] = np.random.randint(18, 70, size=len(non_members))
non_members['education'] = np.random.permutation(non_members['education'].values)
non_members['is_member'] = 0

# Combine into one attack dataset
attack_data = pd.concat([members, non_members], ignore_index=True)
y_true = attack_data['is_member']
attack_features = attack_data.drop('is_member', axis=1)

# Encode categorical features for distance calculation
categorical_cols = attack_features.select_dtypes(include=['object']).columns
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    le.fit(real_data[col].astype(str))
    attack_features[col] = le.transform(attack_features[col].astype(str))
    encoders[col] = le

# Scale features
scaler = StandardScaler()
attack_scaled = scaler.fit_transform(attack_features)

# -------------------------------------
# Evaluate Each Model
# -------------------------------------
datasets = {
    "Baseline CTGAN": "data/synthetic/baseline.csv",
    "DP epsilon=5.0": "data/synthetic/epsilon_50.csv",
    "DP epsilon=1.0": "data/synthetic/epsilon_10.csv",
    "DP epsilon=0.5": "data/synthetic/epsilon_05.csv"
}

results = []

for model_name, path in datasets.items():
    if not os.path.exists(path):
        print(f"Skipping {model_name}, file not found: {path}")
        continue
        
    print(f"\nEvaluating {model_name}...")
    synth_data = pd.read_csv(path)
    
    # Take a sample of synthetic data to compute distances efficiently
    synth_sample = synth_data.sample(min(2000, len(synth_data)), random_state=42).copy()
    
    # Encode and scale synthetic data using the same transformers
    for col in categorical_cols:
        synth_sample[col] = synth_sample[col].astype(str).apply(
            lambda x: x if x in encoders[col].classes_ else encoders[col].classes_[0]
        )
        synth_sample[col] = encoders[col].transform(synth_sample[col])
        
    synth_scaled = scaler.transform(synth_sample)
    
    # Calculate Euclidean distance from each attack record to the nearest synthetic record
    distances = pairwise_distances(attack_scaled, synth_scaled, metric='euclidean')
    min_distances = distances.min(axis=1)
    
    # Distance-based attack: if the distance is smaller than the median distance,
    # the attacker guesses it is a Member (1).
    threshold = np.median(min_distances)
    y_pred = (min_distances < threshold).astype(int)
    
    # Inverse distance as score for ROC-AUC
    y_scores = -min_distances
    
    # Calculate Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_scores)
    
    print(f"  Accuracy : {acc:.2f}")
    print(f"  Precision: {prec:.2f}")
    print(f"  Recall   : {rec:.2f}")
    print(f"  ROC-AUC  : {roc_auc:.2f}")
    
    results.append({
        "Model": model_name,
        "Accuracy": round(acc, 2),
        "Precision": round(prec, 2),
        "Recall": round(rec, 2),
        "F1-score": round(f1, 2),
        "ROC-AUC": round(roc_auc, 2)
    })

# -------------------------------------
# Save Results
# -------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv("results/privacy_metrics.csv", index=False)

print("\n" + "=" * 60)
print("Privacy Leakage Measurement Complete")
print("Results saved to: results/privacy_metrics.csv")
print("=" * 60)
print(results_df.to_string(index=False))
