<div align="center">

# 🛡️ PrivSynth-AI

### Privacy-Preserving Synthetic Data Generation using Differential Privacy & CTGAN

Generate high-quality synthetic tabular datasets while protecting sensitive information using Differential Privacy.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=for-the-badge&logo=streamlit)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-orange?style=for-the-badge&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</div>

---

# 📖 Overview

**PrivSynth-AI** is an AI-powered privacy-preserving synthetic data generation system that creates realistic tabular datasets using **CTGAN** enhanced with **Differential Privacy (DP-SGD)**.

The project demonstrates how organizations can safely share sensitive healthcare, financial, or census datasets without exposing individual records while maintaining high analytical utility.

---

# ✨ Features

- 🔒 Differential Privacy with DP-SGD
- 🤖 CTGAN-based Synthetic Data Generation
- 📊 Utility Evaluation (Train on Synthetic, Test on Real)
- 🛡️ Membership Inference Attack Evaluation
- 📈 Privacy vs Utility Trade-off Analysis
- 🌐 Interactive Streamlit Dashboard
- 📥 Download Generated Synthetic Datasets
- 📂 Multiple Privacy Budgets (ε = 0.5, 1.0, 5.0)

---

# 🏗️ System Architecture

```
Raw Dataset
      │
      ▼
Data Preprocessing
      │
      ▼
CTGAN Training
      │
      ▼
Differential Privacy (DP-SGD)
      │
      ▼
Synthetic Dataset Generation
      │
      ▼
Utility Evaluation
      │
      ▼
Privacy Evaluation
      │
      ▼
Interactive Dashboard
```

---

# 📁 Project Structure

```
PrivSynth-AI
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic/
│
├── models/
│
├── src/
│   ├── preprocessing.py
│   ├── train_ctgan.py
│   ├── dp_train.py
│   ├── generate_data.py
│   ├── generate_dp_data.py
│   ├── evaluate.py
│   ├── membership_attack.py
│   └── tradeoff_analysis.py
│
├── results/
│
└── screenshots/
```

---

# 🚀 Installation

## Clone Repository

```bash
git clone https://github.com/AmitKumarMaurya92/PrivSynth-AI.git

cd PrivSynth-AI
```

## Create Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Dashboard

```bash
streamlit run app.py
```

The application will open at

```
http://localhost:8501
```

---

# 📊 Dataset

This project uses the **UCI Adult Census Income Dataset**.

| Property | Details |
|-----------|---------|
| Dataset | Adult Census Income |
| Records | 32,000+ |
| Features | 15 |
| Target | Income (>50K / ≤50K) |

---

# 🧠 Model Training

## Step 1

```bash
python src/preprocessing.py
```

## Step 2

```bash
python src/train_ctgan.py
```

## Step 3

```bash
python src/dp_train.py
```

---

# 📈 Generate Synthetic Data

```bash
python src/generate_data.py

python src/generate_dp_data.py
```

---

# 📊 Evaluate Utility

```bash
python src/evaluate.py
```

Metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

---

# 🔒 Privacy Evaluation

```bash
python src/membership_attack.py
```

Evaluates:

- Membership Inference Attack
- Privacy Leakage
- Attack Accuracy
- Privacy Risk

---

# 📉 Privacy–Utility Trade-off

```bash
python src/tradeoff_analysis.py
```

Generates

- Utility Comparison
- Privacy Comparison
- Similarity Analysis
- Trade-off Graphs

---

# 📸 Screenshots

## Home Page

```
screenshots/home.png
```

## Dataset Upload

```
screenshots/upload.png
```

## Generate Synthetic Data

```
screenshots/generate.png
```

## Dashboard

```
screenshots/dashboard.png
```

## Results

```
screenshots/results.png
```

---

# 📊 Results

The project automatically generates evaluation reports including:

- Utility Metrics
- Privacy Metrics
- Membership Attack Results
- Trade-off Analysis
- Graphical Comparisons

All generated reports are stored inside:

```
results/
```

---

# 🛠️ Technology Stack

| Category | Technology |
|-----------|------------|
| Language | Python |
| Dashboard | Streamlit |
| Deep Learning | PyTorch |
| Generative AI | CTGAN |
| Privacy | Opacus (DP-SGD) |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Visualization | Plotly, Matplotlib |
| Dataset | UCI Adult Census |

---

# 📌 Future Improvements

- Diffusion Models
- TVAE Support
- PATE-GAN
- Multi-table Synthetic Data
- Better Privacy Metrics
- Cloud Deployment
- Docker Support

---

# 🤝 Contributing

Contributions are welcome.

1. Fork the repository

2. Create a new branch

```bash
git checkout -b feature-name
```

3. Commit your changes

```bash
git commit -m "Add feature"
```

4. Push

```bash
git push origin feature-name
```

5. Open a Pull Request

---

# 📄 License

This project is licensed under the **MIT License**.

---

# 🙏 Acknowledgements

- CTGAN
- Opacus
- PyTorch
- Streamlit
- Scikit-learn
- UCI Machine Learning Repository

---

<div align="center">

## 👨‍💻 Developed by

# Amit Kumar Maurya

### AI • Machine Learning • Privacy-Preserving AI • Open Source

⭐ If you found this project helpful, don't forget to star the repository.

</div>