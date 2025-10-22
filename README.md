# WATER-QUALITY-PREDICTION-USING-MACHINE-LEARNING-MODELS
A machine learning project that predicts water potability using various classification algorithms to ensure safe drinking water quality.
## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Machine Learning Models](#machine-learning-models)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [Authors](#authors)
- [License](#license)

## 🔍 Overview

Access to safe drinking water is a fundamental human right. This project uses machine learning to classify water samples as safe or unsafe for consumption based on key water quality indicators. We evaluated **6 different machine learning models** to predict water potability with varying accuracy rates.

**Key Highlights:**
- 🎯 Best Model: Support Vector Machine (SVM) with **66.08% accuracy**
- 📊 Dataset: 3,276 records with 10 water quality features
- 🔬 Models Tested: 6 classification algorithms
- 📈 Train/Test Split: 66% / 33%

## 📊 Dataset

The dataset contains **3,276 water samples** with the following features:

| Feature | Description |
|---------|-------------|
| **pH** | pH level of water |
| **Hardness** | Water hardness (mg/L) |
| **Solids** | Total dissolved solids (ppm) |
| **Chloramines** | Chloramines concentration (ppm) |
| **Sulfate** | Sulfate concentration (mg/L) |
| **Conductivity** | Electrical conductivity (μS/cm) |
| **Organic Carbon** | Organic carbon content (ppm) |
| **Trihalomethanes** | Trihalomethanes concentration (μg/L) |
| **Turbidity** | Water turbidity (NTU) |
| **Potability** | Target variable (0 = Not Potable, 1 = Potable) |

## 🤖 Machine Learning Models

We implemented and compared the following algorithms:

### 1. **Logistic Regression**
- Binary classification using sigmoid function
- Accuracy: **60.11%**

### 2. **Decision Tree**
- Tree-based hierarchical classification
- Accuracy: **62.62%**

### 3. **K-Nearest Neighbors (KNN)**
- Instance-based learning algorithm
- Accuracy: **57.56%** (Lowest)

### 4. **Gaussian Naïve Bayes**
- Probabilistic classifier based on Bayes' theorem
- Accuracy: **61.39%**

### 5. **Support Vector Machine (SVM)** ⭐
- Optimal hyperplane classification
- Accuracy: **66.08%** (Highest)

### 6. **Random Forest**
- Ensemble learning with multiple decision trees
- Accuracy: **62.84%**

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository
```bash
git clone https://github.com/yourusername/water-quality-analysis.git
cd water-quality-analysis
```

2. Create a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages
```bash
pip install -r requirements.txt
```

### Required Libraries

```
numpy
pandas
scikit-learn
matplotlib
seaborn
jupyter
```

## 💻 Usage

### Running the Analysis

1. **Jupyter Notebook**
```bash
jupyter notebook
```
Open `water_quality_analysis.ipynb` and run all cells.

2. **Python Script**
```bash
python train_models.py
```

### Basic Example

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv('water_quality.csv')

# Prepare features and target
X = data.drop('Potability', axis=1)
y = data['Potability']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVM model
svm_model = SVC(kernel='rbf')
svm_model.fit(X_train_scaled, y_train)

# Evaluate
accuracy = svm_model.score(X_test_scaled, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")
```

## 📈 Results

### Model Performance Comparison

| Model | Accuracy | Rank |
|-------|----------|------|
| **Support Vector Machine (SVM)** | 66.08% | 🥇 |
| **Random Forest** | 62.84% | 🥈 |
| **Decision Tree** | 62.62% | 🥉 |
| **Gaussian Naïve Bayes** | 61.39% | 4 |
| **Logistic Regression** | 60.11% | 5 |
| **K-Nearest Neighbors** | 57.56% | 6 |

### Key Findings

- ✅ **SVM performed best** with 66.08% accuracy, effectively separating potable and non-potable water samples
- 📊 Feature importance analysis revealed critical water quality indicators
- ⚠️ Class imbalance in the dataset affected model performance
- 🔄 Hyperparameter tuning and cross-validation improved model robustness

### Confusion Matrix Analysis

The models showed:
- High **True Positives** for the majority class
- Elevated **False Negatives** due to class imbalance
- Room for improvement in minority class detection

## 📁 Project Structure

```
water-quality-analysis/
│
├── data/
│   └── water_quality.csv
│
├── notebooks/
│   └── water_quality_analysis.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── train_models.py
│   └── evaluate_models.py
│
├── results/
│   ├── confusion_matrices/
│   ├── accuracy_comparison.png
│   └── feature_importance.png
│
├── requirements.txt
├── README.md
└── LICENSE
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 👥 Author

- **Ubaid Ullah**
 
**Course:** Machine Learning Lab (COMP-240L)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


⭐ If you found this project helpful, please consider giving it a star!

**Made with ❤️ for safe drinking water**
