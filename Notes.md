# 🧠 Machine Learning — Comprehensive Notes  
**Course:** B.E. Information Technology (SPPU – 2019 Pattern)  
**Subject:** Machine Learning  
**Purpose:** Theoretical Reference + Practical Integration  

---

## 📘 1. Introduction to Machine Learning

### What is Machine Learning?

Machine Learning (ML) is a subset of Artificial Intelligence (AI) that focuses on building systems capable of learning from experience (data).  
Instead of being explicitly programmed, an ML model discovers patterns and relationships in data and uses them to make predictions or decisions.

**Formal Definition (Tom Mitchell, 1997):**  
> “A computer program is said to learn from experience *E* with respect to some class of tasks *T* and performance measure *P* if its performance at tasks in *T*, as measured by *P*, improves with experience *E*.”

---

### Key Features of ML
- Learns automatically from data  
- Improves over time with more examples  
- Handles noisy and complex real-world data  
- Reduces human intervention in decision-making  

---

### Types of Machine Learning

#### 1. Supervised Learning
- Works on labeled data (input + expected output)
- Learns mapping function f: X → Y
- **Examples:**
  - Classification → Predict categories (spam/ham, disease/no disease)
  - Regression → Predict continuous values (house price, temperature)
- **Algorithms:** Linear Regression, Logistic Regression, Decision Tree, SVM, KNN, Naive Bayes

#### 2. Unsupervised Learning
- Works on unlabeled data; finds hidden patterns
- **Examples:** K-Means, Hierarchical Clustering, PCA
- **Goal:** Discover structure in data

#### 3. Semi-Supervised Learning
- Uses both labeled and unlabeled data  
- **Example:** Label propagation, graph-based models

#### 4. Reinforcement Learning
- Agent learns by interacting with the environment  
- Objective: Maximize cumulative reward  
- **Examples:** Q-Learning, Deep Q-Networks, AlphaGo

---

## 🧩 2. Machine Learning System Workflow

1. **Problem Definition:** Identify the problem type (classification, regression, clustering)  
2. **Data Collection:** Gather data from sensors, APIs, or databases  
3. **Data Preprocessing:** Handle missing values, encode categorical variables, scale features  
4. **Splitting Dataset:** Train (70–80%) and Test (20–30%)  
5. **Model Training:** Train using chosen algorithm and optimize parameters  
6. **Model Evaluation:** Measure accuracy, precision, recall, R², etc.  
7. **Model Optimization:** Tune hyperparameters (Grid Search, Random Search)  
8. **Model Deployment:** Use trained model in real-world applications  

---

## ⚙️ 3. Important Concepts

### Features and Labels
- **Feature (X):** Input variable (e.g., Age, Income)  
- **Label (Y):** Output variable (e.g., Disease = Yes/No)

### Overfitting vs Underfitting
- **Overfitting:** Model performs well on training but poorly on test data  
  - Solution: Regularization, dropout, pruning  
- **Underfitting:** Model too simple to capture patterns  
  - Solution: Add features or use a complex model

### Bias–Variance Tradeoff
| Bias | Variance | Result |
|------|-----------|--------|
| High | Low | Underfitting |
| Low | High | Overfitting |
| Balanced | Balanced | Good generalization |

---

## 🧮 4. Mathematics Behind ML

### Linear Algebra
- Vectors and matrices represent datasets and model parameters
- Dot Product measures vector similarity
- Matrix Multiplication used in neural networks

### Probability & Statistics
- **Bayes’ Theorem:**  
  P(A|B) = (P(B|A) * P(A)) / P(B)
- **Expectation:** Mean of random variable  
- **Variance:** Spread of data from mean

### Calculus
- Used for optimization (minimizing loss)
- **Gradient Descent:**  
  theta = theta - alpha * (dJ(theta) / dtheta)  
  where alpha = learning rate

### Distance Metrics
- **Euclidean Distance:** sqrt((x1 - y1)^2 + (x2 - y2)^2)  
- **Manhattan Distance:** |x1 - y1| + |x2 - y2|

---

## 📊 5. Model Evaluation Metrics

### Classification Metrics

| Metric | Formula | Meaning |
|--------|----------|---------|
| **Accuracy** | (TP + TN) / (TP + TN + FP + FN) | Proportion of correct predictions |
| **Precision** | TP / (TP + FP) | How many predicted positives are true |
| **Recall (Sensitivity)** | TP / (TP + FN) | How many actual positives are captured |
| **F1-Score** | 2 * (Precision * Recall) / (Precision + Recall) | Balance between Precision & Recall |

- **Confusion Matrix Example:**

| Actual / Predicted | Positive | Negative |
|--------------------|-----------|-----------|
| Positive | TP | FN |
| Negative | FP | TN |

---

### Regression Metrics

| Metric | Formula | Description |
|---------|----------|-------------|
| **Mean Absolute Error (MAE)** | (1/n) * Σ|y - ŷ| | Average absolute difference |
| **Mean Squared Error (MSE)** | (1/n) * Σ(y - ŷ)² | Penalizes larger errors |
| **R² Score** | 1 - (SS_res / SS_tot) | Goodness of fit |

---

### Clustering Metrics

| Metric | Formula | Description |
|---------|----------|-------------|
| **Silhouette Score** | (b - a) / max(a, b) | Cluster separation and compactness |
| **SSE (Inertia)** | Σ (distance of points from centroid)² | Compactness measure |

---

## 🔍 6. Major Algorithm Families

### Linear Models
- **Linear Regression:** Predicts continuous outcomes  
- **Logistic Regression:** Classification using sigmoid function

### Tree-Based Models
- **Decision Tree:** Uses entropy or Gini impurity to split data  
- **Random Forest:** Ensemble of trees reducing overfitting  
- **Gradient Boosting:** Sequential trees correcting previous errors

### Instance-Based Models
- **KNN (K-Nearest Neighbors):** Classifies by majority vote among nearest data points

### Probabilistic Models
- **Naive Bayes:** Based on Bayes’ theorem assuming feature independence

### Unsupervised Models
- **K-Means Clustering:** Groups data into K clusters by minimizing intra-cluster variance  
- **Hierarchical Clustering:** Builds nested clusters using linkage distance  
- **PCA (Principal Component Analysis):** Reduces dimensionality using variance maximization

---

## 📚 7. Regularization Techniques

Regularization prevents overfitting by penalizing large coefficients.

| Technique | Penalty Term | Description |
|------------|---------------|--------------|
| **L1 (Lasso)** | lambda * Σ|w| | Shrinks some weights to zero |
| **L2 (Ridge)** | lambda * Σw² | Distributes penalty across weights |
| **Elastic Net** | Combination of L1 & L2 | Balances sparsity and stability |

---

## 🧠 8. Feature Engineering

### Key Steps
1. **Feature Extraction:** Derive new features from raw data  
2. **Feature Selection:** Remove redundant or irrelevant attributes  
3. **Dimensionality Reduction:** Use PCA or autoencoders  
4. **Encoding Categorical Data:** Label or One-Hot Encoding  
5. **Feature Scaling:**  
   - Normalization: (x - min) / (max - min)  
   - Standardization: (x - mean) / standard deviation  

---

## 💡 9. Modern ML Trends

| Field | Examples |
|--------|-----------|
| **Deep Learning** | Neural Networks, CNNs, RNNs |
| **Ensemble Learning** | Bagging, Boosting, Stacking |
| **Transfer Learning** | Fine-tuning pre-trained models |
| **AutoML** | Automated hyperparameter tuning |
| **Explainable AI (XAI)** | Interpretable ML models |
| **MLOps** | Managing end-to-end ML lifecycle |

---

## 🧾 10. Key Insights and Summary

- ML systems **learn patterns from data**, not rules  
- **Data preprocessing** is crucial for performance  
- **Evaluation metrics** define model success  
- **Bias–Variance Tradeoff** governs model generalization  
- Ensemble and hybrid methods often outperform single models  

---

**End of General Notes**
