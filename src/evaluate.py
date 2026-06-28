import os
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# ------------------------------
# Load datasets
# ------------------------------

real_df = pd.read_csv("Data/processed/adult_clean.csv")

synthetic_df = pd.read_csv("Data/synthetic/baseline.csv")

print("Datasets Loaded Successfully")

# ------------------------------
# Convert categorical columns
# ------------------------------

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

encoder = LabelEncoder()

for col in categorical_columns:

    real_df[col] = encoder.fit_transform(real_df[col])

    synthetic_df[col] = encoder.transform(synthetic_df[col])

# ------------------------------
# Split Real Dataset
# ------------------------------

X_real = real_df.drop("income", axis=1)

y_real = real_df["income"]

X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(

    X_real,

    y_real,

    test_size=0.2,

    random_state=42

)

# ------------------------------
# Synthetic Dataset
# ------------------------------

X_syn = synthetic_df.drop("income", axis=1)

y_syn = synthetic_df["income"]

# ------------------------------
# Train Classifier
# ------------------------------

model = RandomForestClassifier(

    n_estimators=100,

    random_state=42

)

model.fit(

    X_syn,

    y_syn

)

# ------------------------------
# Predict
# ------------------------------

predictions = model.predict(X_test_real)

probabilities = model.predict_proba(X_test_real)[:,1]

# ------------------------------
# Metrics
# ------------------------------

accuracy = accuracy_score(

    y_test_real,

    predictions

)

precision = precision_score(

    y_test_real,

    predictions

)

recall = recall_score(

    y_test_real,

    predictions

)

f1 = f1_score(

    y_test_real,

    predictions

)

roc = roc_auc_score(

    y_test_real,

    probabilities

)

print()

print("="*60)

print("UTILITY EVALUATION")

print("="*60)

print(f"Accuracy : {accuracy:.4f}")

print(f"Precision : {precision:.4f}")

print(f"Recall : {recall:.4f}")

print(f"F1 Score : {f1:.4f}")

print(f"ROC AUC : {roc:.4f}")

# ------------------------------
# Save Results
# ------------------------------

os.makedirs("results",exist_ok=True)

results = pd.DataFrame({

    "Metric":[

        "Accuracy",

        "Precision",

        "Recall",

        "F1 Score",

        "ROC AUC"

    ],

    "Value":[

        accuracy,

        precision,

        recall,

        f1,

        roc

    ]

})

results.to_csv(

    "results/utility_metrics.csv",

    index=False

)

print()

print("Utility metrics saved successfully.")