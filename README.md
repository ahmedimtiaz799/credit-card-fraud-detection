# Credit Card Fraud Detection 🕵️‍♂️💳

This project focuses on detecting fraudulent credit card transactions using machine learning techniques. The dataset used is highly imbalanced, so techniques such as feature scaling, visualization, and threshold tuning are applied to improve fraud detection performance.

## 📂 Dataset

- **Source**: [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Description**: Contains transactions made by European cardholders in September 2013. Out of 284,807 transactions, 492 are fraudulent (Class 1), making it a highly imbalanced dataset.

---

## ⚙️ Technologies Used

- Python
- Pandas, NumPy
- Seaborn, Matplotlib
- Scikit-learn
- Jupyter Notebook

---

## 🔍 Project Workflow

### 1. **Data Loading & Initial Exploration**
- Loaded dataset using `pandas.read_csv()`
- Viewed basic info using `.shape`, `.info()`, `.head()`, `.columns`

### 2. **Data Cleaning**
- Removed duplicates
- Checked and confirmed no null values
- Analyzed class distribution

### 3. **Exploratory Data Analysis (EDA)**
- Plotted class distribution with `seaborn.countplot()`
- Analyzed distributions of `Amount` and `Time` features
- Created correlation heatmap
- Visualized boxplots to identify differences in fraud vs. non-fraud amounts

### 4. **Feature Engineering**
- Extracted `Hour` from `Time` feature
- Dropped `Time`
- Scaled the `Amount` feature using `RobustScaler`

### 5. **Model Training**
- **Features (`X`)**: All columns except `Class`
- **Target (`y`)**: `Class`
- Split data using `train_test_split` with stratification

#### Logistic Regression:
- Trained with `class_weight='balanced'` due to imbalance
- Default threshold (0.5) resulted in low F1-score: **0.10**
- Threshold tuning (0.05, 0.3) showed improvement in F1-score

#### Random Forest Classifier:
- Trained with `class_weight='balanced'`
- Delivered significantly better results:
  - Precision (fraud): 0.97
  - Recall (fraud): 0.71
  - F1-score (fraud): 0.82

---

## 📈 Evaluation Metrics

- **F1 Score**
- **Confusion Matrix**
- **Classification Report (Precision, Recall, F1)**

> Random Forest performed best overall and handled the class imbalance more effectively than Logistic Regression.

---

## 📊 Visualizations

- Class distribution (`sns.countplot`)
- Amount distribution (`sns.histplot`)
- Time distribution (`sns.histplot`)
- Boxplot of Amount by Class
- Correlation heatmap

---

## ✅ Results Summary

| Model              | F1-Score (Fraud) | Precision (Fraud) | Recall (Fraud) |
|-------------------|------------------|-------------------|----------------|
| Logistic Regression | 0.10             | Low               | Low            |
| Random Forest       | 0.82             | 0.97              | 0.71           |


