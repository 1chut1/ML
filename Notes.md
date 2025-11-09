# 🧠 Machine Learning — Comprehensive Notes  
**Course:** B.E. Information Technology (SPPU – 2019 Pattern)**  
**Subject:** Machine Learning  
**Type:** Theoretical Notes (with Practical Context)  
**Format:** GitHub-compatible Markdown  

---

## 📘 1. Introduction to Machine Learning

Machine Learning (ML) is a subfield of Artificial Intelligence that enables computers to learn patterns and make decisions without explicit programming.

A machine "learns" when it improves its performance **P** at a certain task **T** through experience **E**.

Example:  
- **Task (T):** Classify emails as spam or not spam.  
- **Experience (E):** Thousands of labeled emails.  
- **Performance (P):** Accuracy of spam predictions.

---

### 🧩 1.1 Types of Machine Learning

| Type | Description | Example Algorithms |
|------|--------------|--------------------|
| **Supervised Learning** | Learns from labeled data (input + output). | Linear Regression, Logistic Regression, Decision Tree |
| **Unsupervised Learning** | Learns structure in unlabeled data. | K-Means, PCA, Hierarchical Clustering |
| **Semi-Supervised Learning** | Uses both labeled and unlabeled data. | Self-training, Label Propagation |
| **Reinforcement Learning** | Learns optimal actions via rewards and penalties. | Q-Learning, SARSA, Deep Q Networks |

---

### 🧠 1.2 Real-World Applications

- Predictive Analytics — sales, stock trends  
- Medical Diagnosis — disease prediction  
- NLP — sentiment analysis, chatbots  
- Computer Vision — face and object detection  
- Recommender Systems — Netflix, YouTube, Amazon  

---

## ⚙️ 2. ML Workflow

1. **Define the Problem** → Classification? Regression? Clustering?  
2. **Data Collection** → Gather data from sensors, APIs, databases.  
3. **Data Preprocessing** → Clean, normalize, encode, and split.  
4. **Model Selection** → Choose algorithm based on problem.  
5. **Training** → Fit model on training data.  
6. **Evaluation** → Measure model performance.  
7. **Optimization** → Tune hyperparameters to improve metrics.  
8. **Deployment** → Integrate trained model into applications.

---

## 📊 3. Data Preprocessing Essentials

### 3.1 Handling Missing Data
- Replace missing values using:
  - **Mean/Median/Mode imputation**
  - **Forward Fill/Backward Fill**
  - **Drop missing rows**

### 3.2 Encoding Categorical Data
- **Label Encoding** — assign numeric labels.  
- **One-Hot Encoding** — create binary indicator columns.

### 3.3 Feature Scaling
- **Normalization (Min–Max):**  
  x' = (x − min(x)) / (max(x) − min(x))
- **Standardization (Z-score):**  
  x' = (x − mean) / standard deviation

### 3.4 Outlier Detection
- Identify anomalies using boxplots, IQR, or z-scores.

---

## 🔢 4. Confusion Matrix Explained

The **Confusion Matrix** is a performance table for classification problems.  
It compares predicted labels with actual labels.

| Actual / Predicted | Positive | Negative |
|--------------------|-----------|-----------|
| Positive | **TP** (True Positive) | **FN** (False Negative) |
| Negative | **FP** (False Positive) | **TN** (True Negative) |

### 4.1 Metrics Derived from Confusion Matrix
- **Accuracy** = (TP + TN) / (TP + TN + FP + FN)  
  → Overall correctness.  

- **Precision** = TP / (TP + FP)  
  → How many predicted positives are correct.  

- **Recall (Sensitivity)** = TP / (TP + FN)  
  → How many actual positives were captured.  

- **Specificity** = TN / (TN + FP)  
  → How many actual negatives were correctly predicted.  

- **F1-Score** = 2 * (Precision * Recall) / (Precision + Recall)  
  → Balanced measure of precision and recall.

**Example Interpretation:**  
A disease detection model with:
- High precision = few false alarms.  
- High recall = fewer missed diseases.  

---

## 📈 5. Regression Metrics

| Metric | Formula | Meaning |
|--------|----------|---------|
| **MAE (Mean Absolute Error)** | (1/n) * Σ |y − ŷ| | Average absolute deviation |
| **MSE (Mean Squared Error)** | (1/n) * Σ (y − ŷ)² | Penalizes large errors |
| **RMSE (Root Mean Squared Error)** | sqrt(MSE) | More interpretable scale |
| **R² (Coefficient of Determination)** | 1 − (SS_res / SS_tot) | Fit quality (close to 1 is better) |

---

## 🌳 6. Decision Tree Algorithm

### 6.1 Concept
A **Decision Tree** recursively splits data based on feature values that best separate the classes.

Each internal node → feature test  
Each branch → outcome of test  
Each leaf node → class label or value

---

### 6.2 Entropy and Information Gain

**Entropy (measure of impurity):**  
Entropy(S) = − p₁ * log₂(p₁) − p₂ * log₂(p₂)

- 0 → perfectly pure node  
- 1 → completely mixed node

**Information Gain:**  
IG(S, A) = Entropy(S) − Σ ((|Sᵢ| / |S|) * Entropy(Sᵢ))

The feature with **highest IG** is chosen for splitting.

---

### 6.3 Gini Index
Alternative impurity measure:  
Gini = 1 − Σ (pᵢ²)

Lower Gini = purer node.

---

## 🔺 7. Linear & Logistic Regression

### 7.1 Linear Regression
Predicts continuous outcomes using linear relation:
y = m*x + c

Goal: minimize Mean Squared Error between predicted (ŷ) and actual (y).

### 7.2 Logistic Regression
Used for classification (binary or multiclass).  
Maps output to probability using **sigmoid function**:
P(y=1) = 1 / (1 + e^−z)

If P > 0.5 → class = 1  
Else → class = 0

---

## 🧮 8. Gradient Descent (Optimization)

**Purpose:** Find model parameters that minimize the loss function.

**Process:**
1. Initialize weights randomly.  
2. Compute loss (error).  
3. Update weights:  
   w = w − α * (∂J/∂w)  
   where α = learning rate.  
4. Repeat until convergence (loss stops decreasing).

### Variants
- **Batch Gradient Descent:** Uses all data at once.  
- **Stochastic Gradient Descent (SGD):** Updates after each sample.  
- **Mini-Batch GD:** Uses small batches (balance between speed and stability).

---

## 📉 9. Bias–Variance Tradeoff

- **Bias:** Error due to assumptions; model too simple.  
- **Variance:** Error due to sensitivity to training data.  

| Model Behavior | Bias | Variance | Performance |
|----------------|------|-----------|--------------|
| Underfitting | High | Low | Poor |
| Overfitting | Low | High | Poor |
| Balanced | Moderate | Moderate | Optimal |

---

## 🔍 10. Regularization

Regularization discourages overly complex models by penalizing large weights.

| Type | Formula | Effect |
|------|----------|--------|
| **L1 (Lasso)** | Cost = Loss + λ * Σ|w| | Forces some weights to zero (feature selection) |
| **L2 (Ridge)** | Cost = Loss + λ * Σ(w²) | Shrinks weights smoothly |
| **Elastic Net** | Combination of L1 and L2 | Balances both penalties |

Higher λ → stronger penalty → simpler model.

---

## 🧠 11. K-Means Clustering

**Goal:** Partition data into K clusters where each point belongs to the nearest centroid.

**Steps:**
1. Choose number of clusters (K).  
2. Randomly assign cluster centroids.  
3. Assign each point to the nearest centroid.  
4. Recalculate centroids.  
5. Repeat until centroids stabilize.

**Evaluation:**
- **Elbow Method:** Plot SSE vs. K to find optimal clusters.  
- **Silhouette Score:** Measures cluster separation. (1 = ideal)

---

## 🧬 12. Hierarchical Clustering

Builds nested clusters using a tree (dendrogram).  
- **Agglomerative:** Start with individual points → merge.  
- **Divisive:** Start with all points → split.  

**Linkage Methods:**
- Single (minimum distance)  
- Complete (maximum distance)  
- Average (mean distance)  
- Ward (minimizes variance)

---

## 🎯 13. Principal Component Analysis (PCA)

**Purpose:** Reduce dimensionality while retaining variance.

**Steps:**
1. Standardize data.  
2. Compute covariance matrix.  
3. Calculate eigenvectors & eigenvalues.  
4. Select top k eigenvectors (principal components).  
5. Transform data into new feature space.

Result: Fewer features → faster training → less noise.

---

## 🧩 14. Overfitting and Model Validation

### 14.1 Overfitting
Model learns training data too well (memorizes noise).  
- Symptoms: High training accuracy, low test accuracy.

### 14.2 Cross-Validation
- **K-Fold CV:** Split data into K parts; train on (K−1) parts, test on the rest.  
- Average all test accuracies for reliable estimate.

---

## 🔬 15. Evaluation and Visualization Tools

- **ConfusionMatrixDisplay (sklearn):** Visualize confusion matrix.  
- **Classification Report:** Shows precision, recall, F1-score.  
- **Learning Curves:** Plot training vs. validation accuracy.  
- **ROC Curve (Receiver Operating Characteristic):** Plots TPR vs. FPR.  
- **AUC (Area Under Curve):** Measures classifier quality (closer to 1 = better).

---

## 🧩 16. Advanced ML Topics Overview

| Topic | Description |
|--------|-------------|
| **Ensemble Learning** | Combines multiple models to improve accuracy (Bagging, Boosting, Stacking). |
| **Bagging** | Trains many models on random data subsets (e.g., Random Forest). |
| **Boosting** | Sequentially corrects errors (e.g., AdaBoost, XGBoost, Gradient Boosting). |
| **Neural Networks** | Multi-layer models inspired by human brain structure. |
| **Deep Learning** | Neural networks with many hidden layers (CNNs, RNNs, LSTMs). |
| **Transfer Learning** | Uses pre-trained models for new tasks. |
| **Explainable AI (XAI)** | Makes ML decisions interpretable to humans. |

---

## 🧾 17. Summary

- Machine Learning enables **data-driven decisions**.  
- Preprocessing and feature selection are **key to performance**.  
- Model choice depends on **problem type and data nature**.  
- Evaluation metrics define real success — not accuracy alone.  
- Balance **bias–variance** to achieve robust generalization.  
- Modern trends move toward **explainability and automation**.

---

**End of Comprehensive Notes**
