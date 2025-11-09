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

### Key Features of ML:
- Learns automatically from data.
- Improves over time with more examples.
- Handles noisy and complex real-world data.
- Reduces human intervention in decision-making.

---

### Types of Machine Learning:

#### **1. Supervised Learning**
- Trains on *labeled* data (input + expected output).  
- Objective: Learn mapping \( f: X \rightarrow Y \).
- **Examples:**
  - Classification → Predict category (spam/ham, disease/no disease).  
  - Regression → Predict continuous value (house price, temperature).  
- **Algorithms:** Linear Regression, Logistic Regression, Decision Trees, SVM, KNN, Naive Bayes.

#### **2. Unsupervised Learning**
- Works on *unlabeled* data; finds hidden patterns or groupings.  
- **Examples:**
  - Clustering → K-Means, DBSCAN, Hierarchical Clustering.  
  - Dimensionality Reduction → PCA, t-SNE.
- **Goal:** Discover structure or distribution within data.

#### **3. Semi-Supervised Learning**
- Uses both labeled and unlabeled data (useful when labeling is expensive).
- **Example:** Label propagation algorithms, graph-based methods.

#### **4. Reinforcement Learning (RL)**
- Agent learns by interacting with an environment and receiving rewards/punishments.
- **Goal:** Maximize cumulative reward.
- **Examples:** Q-learning, Deep Q Networks (DQN), AlphaGo.

---

## 🧩 2. Machine Learning System Workflow

### **1. Problem Definition**
Identify the problem type — classification, regression, or clustering.

### **2. Data Collection**
Gather raw data from databases, sensors, APIs, or logs.

### **3. Data Preprocessing**
Data preprocessing ensures data is clean and usable.

- **Handling Missing Values:** Use mean/median imputation or delete rows.
- **Feature Encoding:** Convert categorical variables into numeric (Label Encoding, One-Hot Encoding).
- **Feature Scaling:** Normalize or standardize numerical features.
- **Outlier Removal:** Identify and handle anomalies.
- **Feature Selection:** Keep only relevant features to avoid overfitting.

### **4. Splitting Dataset**
- **Training Set:** Used to train the model (usually 70–80%).
- **Test Set:** Used to evaluate performance (20–30%).
- Optionally, a **Validation Set** can be used for hyperparameter tuning.

### **5. Model Training**
Model learns patterns from data by optimizing parameters using algorithms like Gradient Descent.

### **6. Model Evaluation**
- **Supervised:** Use Accuracy, Precision, Recall, R² Score.
- **Unsupervised:** Use Silhouette Score, Davies–Bouldin Index.
- **Cross-Validation:** Ensure generalization.

### **7. Model Optimization**
Use hyperparameter tuning methods:
- Grid Search
- Random Search
- Bayesian Optimization

### **8. Model Deployment**
Export the trained model and integrate into real-world applications via APIs or microservices.

---

## ⚙️ 3. Important Concepts

### **a. Features and Labels**
- **Feature (X):** Independent variable or input (e.g., Age, Income).
- **Label (Y):** Dependent variable or output (e.g., Disease = Yes/No).

### **b. Overfitting vs Underfitting**
- **Overfitting:** Model learns noise, performs poorly on unseen data.  
  *Solution:* Regularization, dropout, pruning.
- **Underfitting:** Model too simple; fails to capture relationships.  
  *Solution:* Use more complex models or add relevant features.

### **c. Bias–Variance Tradeoff**
| Bias | Variance | Behavior |
|------|-----------|-----------|
| High Bias | Low Variance | Underfitting |
| Low Bias | High Variance | Overfitting |
| Optimal | Balanced | Good generalization |

---

## 🧮 4. Mathematics Behind ML

### **A. Linear Algebra**
- **Vectors & Matrices:** Represent datasets and model parameters.
- **Dot Product:** Measures similarity between vectors.
- **Matrix Multiplication:** Used in weight updates (especially in neural networks).

### **B. Probability & Statistics**
- **Bayes’ Theorem:**  
  \[
  P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
  \]

- **Expectation:**  
  \[
  E[X] = \sum_i x_i P(x_i)
  \]

- **Variance:**  
  \[
  Var(X) = E[(X - \mu)^2]
  \]

### **C. Calculus**
Used for model optimization (finding minima in loss function).

- **Gradient Descent:**  
  \[
  \theta := \theta - \alpha \frac{\partial J(\theta)}{\partial \theta}
  \]
  where \( J(\theta) \) = loss, and \( \alpha \) = learning rate.

### **D. Distance Metrics**
- **Euclidean Distance:**  
  \[
  d(x, y) = \sqrt{\sum_{i=1}^n (x_i - y_i)^2}
  \]
- **Manhattan Distance:**  
  \[
  d(x, y) = \sum_{i=1}^n |x_i - y_i|
  \]

---

## 📊 5. Model Evaluation Metrics (In Depth)

### **1. Classification Metrics**

#### Confusion Matrix:
| Actual / Predicted | Positive | Negative |
|--------------------|-----------|-----------|
| Positive | TP | FN |
| Negative | FP | TN |

- **Accuracy:**  
  \[
  Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
  \]

- **Precision:**  
  \[
  Precision = \frac{TP}{TP + FP}
  \]

- **Recall (Sensitivity):**  
  \[
  Recall = \frac{TP}{TP + FN}
  \]

- **F1-Score:**  
  \[
  F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
  \]

---

### **2. Regression Metrics**

- **Mean Absolute Error (MAE):**  
  \[
  MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y_i}|
  \]

- **Mean Squared Error (MSE):**  
  \[
  MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y_i})^2
  \]

- **R² Score:**  
  \[
  R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum (y_i - \hat{y_i})^2}{\sum (y_i - \bar{y})^2}
  \]

---

### **3. Clustering Metrics**

- **Silhouette Score:**  
  \[
  s = \frac{b - a}{\max(a, b)}
  \]
  where  
  \( a \) = average intra-cluster distance,  
  \( b \) = average nearest-cluster distance.

- **SSE (Sum of Squared Errors):**  
  \[
  SSE = \sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2
  \]

---

## 🔍 6. Major Algorithmic Families

### **A. Linear Models**
- **Linear Regression:** Predicts continuous values.  
  \[
  y = w_0 + w_1x_1 + w_2x_2 + ... + w_nx_n
  \]
- **Logistic Regression:** Classification using sigmoid activation.  
  \[
  P(y=1|x) = \frac{1}{1 + e^{-(w^Tx + b)}}
  \]

---

### **B. Tree-Based Models**

- **Decision Tree Entropy:**  
  \[
  E(S) = -\sum_{i=1}^c p_i \log_2(p_i)
  \]
- **Information Gain:**  
  \[
  IG(S, A) = E(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} E(S_v)
  \]
- **Gini Index:**  
  \[
  Gini = 1 - \sum_{i=1}^c (p_i)^2
  \]

---

### **C. Instance-Based Models**
- **K-Nearest Neighbors (KNN):**  
  Classifies a point based on the majority class of its k-nearest neighbors using distance metrics.

---

### **D. Probabilistic Models**
- **Naive Bayes Classifier:**  
  \[
  P(C|X) = \frac{P(X|C) \cdot P(C)}{P(X)}
  \]
  assuming conditional independence among features.

---

### **E. Unsupervised Models**
- **K-Means Clustering:**  
  Minimizes intra-cluster variance.  
  Objective:  
  \[
  J = \sum_{i=1}^k \sum_{x_j \in C_i} ||x_j - \mu_i||^2
  \]
- **Principal Component Analysis (PCA):**  
  Projects data into new axes of maximum variance using eigenvectors of the covariance matrix.

---

## 📚 7. Regularization Techniques

Regularization prevents overfitting by penalizing large coefficients.

| Technique | Penalty Term | Formula |
|------------|--------------|----------|
| **L1 (Lasso)** | \( \lambda \sum |w_i| \) | Adds absolute penalty |
| **L2 (Ridge)** | \( \lambda \sum w_i^2 \) | Adds squared penalty |
| **Elastic Net** | Combination of L1 & L2 | \( \lambda_1 \sum |w_i| + \lambda_2 \sum w_i^2 \) |

Effect: Shrinks unimportant weights → simplifies the model.

---

## 🧠 8. Feature Engineering

### Steps:
1. **Feature Extraction:** Create new variables from existing data.
2. **Feature Selection:** Choose most informative features.
3. **Dimensionality Reduction:** Use PCA or Autoencoders.
4. **Encoding Categorical Data:** Label or One-Hot Encoding.
5. **Scaling:**  
   - **Normalization (Min–Max):**  
     \[
     X' = \frac{X - X_{min}}{X_{max} - X_{min}}
     \]
   - **Standardization (Z-score):**  
     \[
     X' = \frac{X - \mu}{\sigma}
     \]

---

## 💡 9. Modern Machine Learning Trends

| Field | Techniques |
|--------|-------------|
| **Deep Learning** | Neural Networks, CNNs, RNNs |
| **Ensemble Methods** | Bagging, Boosting, Stacking |
| **Transfer Learning** | Reusing pre-trained models |
| **AutoML** | Automated hyperparameter tuning |
| **Explainable AI (XAI)** | Interpretable ML models |
| **MLOps** | Lifecycle management for ML models |

---

## 🧾 10. Summary of Key Insights

- ML systems **learn patterns from data** instead of following fixed rules.  
- The **data preprocessing** stage is as important as model selection.  
- **Evaluation metrics** define real success, not raw accuracy alone.  
- The **bias–variance tradeoff** guides model generalization.  
- Combining algorithms and ensemble methods often yields higher robustness.  

---

**End of General Notes**
