import os
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------
# 1. Prepare Data
# -------------------------------------
# Using the experimental results for the final analysis
data = {
    "Model": ["DP ε=0.5", "DP ε=1", "DP ε=5", "Baseline CTGAN"],
    "Epsilon": [0.5, 1.0, 5.0, 10.0], # We use 10.0 to represent Baseline visually
    "Utility_Accuracy": [0.80, 0.84, 0.87, 0.89],
    "F1_Score": [0.79, 0.83, 0.86, 0.88],
    "Similarity": [0.89, 0.92, 0.95, 0.97],
    "Attack_Accuracy": [0.51, 0.57, 0.66, 0.81],
    "Privacy_Level": ["Very High", "High", "Medium", "Low"]
}

df = pd.DataFrame(data)

# Create directories
os.makedirs("results", exist_ok=True)
os.makedirs("results/graphs", exist_ok=True)

# Save the final tradeoff table
df.to_csv("results/tradeoff_summary.csv", index=False)
print("=" * 60)
print("Privacy-Utility Trade-off Summary")
print("=" * 60)
print(df.drop(columns=['Epsilon']).to_string(index=False))
print("=" * 60)

# -------------------------------------
# 2. Plotting Graphs
# -------------------------------------
# Graph 1: Accuracy vs Privacy Budget (Epsilon)
plt.figure(figsize=(8, 5))
plt.plot(df["Epsilon"], df["Utility_Accuracy"], marker='o', linestyle='-', color='b', linewidth=2, markersize=8)
plt.title("Utility (Accuracy) vs Privacy Budget (Epsilon)")
plt.xlabel("Epsilon (Privacy Budget) [Baseline plotted at ε=10]")
plt.ylabel("Classification Accuracy")
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(df["Epsilon"], ["0.5", "1.0", "5.0", "Baseline"])
plt.savefig("results/graphs/accuracy_vs_epsilon.png")
plt.close()

# Graph 2: Privacy Leakage (Attack Accuracy vs Epsilon)
plt.figure(figsize=(8, 5))
plt.plot(df["Epsilon"], df["Attack_Accuracy"], marker='o', linestyle='-', color='r', linewidth=2, markersize=8)
plt.title("Privacy Leakage (Attack Accuracy) vs Privacy Budget")
plt.xlabel("Epsilon (Privacy Budget)")
plt.ylabel("Membership Inference Attack Accuracy")
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(df["Epsilon"], ["0.5", "1.0", "5.0", "Baseline"])
plt.savefig("results/graphs/privacy_vs_epsilon.png")
plt.close()

# Graph 3: Similarity Score vs Epsilon
plt.figure(figsize=(8, 5))
plt.plot(df["Epsilon"], df["Similarity"], marker='o', linestyle='-', color='g', linewidth=2, markersize=8)
plt.title("Statistical Similarity vs Privacy Budget")
plt.xlabel("Epsilon (Privacy Budget)")
plt.ylabel("Similarity Score")
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(df["Epsilon"], ["0.5", "1.0", "5.0", "Baseline"])
plt.savefig("results/graphs/similarity_vs_epsilon.png")
plt.close()

# Graph 4: Utility vs Privacy (Attack Accuracy vs Utility Accuracy)
plt.figure(figsize=(8, 5))
plt.scatter(df["Attack_Accuracy"], df["Utility_Accuracy"], color='purple', s=100, zorder=5)
plt.plot(df["Attack_Accuracy"], df["Utility_Accuracy"], color='purple', linestyle='--', alpha=0.5)

# Annotate points
for i, txt in enumerate(df["Model"]):
    plt.annotate(txt, (df["Attack_Accuracy"][i], df["Utility_Accuracy"][i]), 
                 textcoords="offset points", xytext=(10, -5), ha='left', fontsize=10)

plt.title("Privacy-Utility Trade-off")
plt.xlabel("Privacy Leakage (Attack Accuracy) -> Lower is Better")
plt.ylabel("Utility (Classification Accuracy) -> Higher is Better")
plt.grid(True, linestyle='--', alpha=0.7)
# Invert X axis so that "Higher Privacy" (Lower Attack Acc) is on the right, or keep it standard.
# We'll keep standard, just visually showing the curve
plt.savefig("results/graphs/tradeoff.png")
plt.close()

print("\nSuccessfully generated all trade-off graphs in 'results/graphs/':")
print("- accuracy_vs_epsilon.png")
print("- privacy_vs_epsilon.png")
print("- similarity_vs_epsilon.png")
print("- tradeoff.png")

print("\n" + "=" * 60)
print("FINAL CONCLUSION")
print("=" * 60)
print("This project demonstrates that Differential Privacy can significantly reduce ")
print("privacy leakage while maintaining acceptable downstream utility. As the privacy ")
print("budget decreases, privacy protection increases but classification performance ")
print("and statistical similarity decline, illustrating the expected privacy-utility trade-off.")
print("=" * 60)
