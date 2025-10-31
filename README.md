# **DA5401 Assignment 7: Multi-Class Model Selection using ROC and Precision–Recall Curves**
- **AKSHAY KUMAR G**  
- **ME22B044**

---

## **Objective**
The goal of this assignment is to apply **multi-class model evaluation techniques** using **ROC (Receiver Operating Characteristic)** and **Precision–Recall Curves (PRC)**.  
The task involves training multiple classifiers on the **UCI Landsat Satellite dataset** and performing **model selection** based on the interpretability of ROC–AUC and PRC–AP metrics rather than just accuracy.

---

## **Problem Statement**
The Landsat Satellite dataset is a **6-class image classification problem** involving high-dimensional spectral data.  
The main objective is to:

- Compare diverse classifiers (both strong and weak performers).  
- Apply **One-vs-Rest (OvR)** averaging to compute **multi-class ROC and PRC curves**.  
- Identify the best and worst models by analyzing both curve behaviors and numeric performance metrics.  
- Provide a reasoned **recommendation** considering trade-offs between accuracy, ROC–AUC, and PRC–AP.

**Dataset Details:**
- **Features (X):** 36 spectral band values per pixel.  
- **Target (Y):** Land-cover class labels (6 classes: 1, 2, 3, 4, 5, 7).  
- **Train/Test Split:** 70/30 with stratification.  
- **Preprocessing:** Standardized features using `StandardScaler`.  

---

## **Part A: Data Preparation and Baseline [5 points]**

### **Models Trained**
| Model | Library | Expected Performance |
|--------|----------|---------------------|
| K-Nearest Neighbors (KNN) | `sklearn.neighbors.KNeighborsClassifier` | Moderate to Good |
| Decision Tree | `sklearn.tree.DecisionTreeClassifier` | Moderate |
| Dummy Classifier (Prior) | `sklearn.dummy.DummyClassifier` | Baseline |
| Logistic Regression | `sklearn.linear_model.LogisticRegression` | Good (Linear baseline) |
| Naive Bayes (Gaussian) | `sklearn.naive_bayes.GaussianNB` | Variable |
| Support Vector Machine (SVM) | `sklearn.svm.SVC` | Good (Requires probability=True) |

### **Baseline Evaluation**
| Model | Accuracy | Weighted F1 |
|--------|-----------:|------------:|
| **KNN** | 0.9109 | 0.9100 |
| **SVM** | 0.8975 | 0.8960 |
| Logistic Regression | 0.8384 | 0.8119 |
| Decision Tree | 0.8358 | 0.8366 |
| Naive Bayes | 0.7934 | 0.8001 |
| Dummy | 0.2382 | 0.0917 |

**Observation:**  
- **KNN** achieves the highest overall accuracy and F1-score, followed closely by **SVM**.  
- **Dummy Classifier** performs near random, serving as the baseline reference.

---

## **Part B: ROC Analysis for Model Selection [20 points]**

### **1️⃣ Multi-Class ROC (Concept)**  
- ROC measures the trade-off between **True Positive Rate (TPR)** and **False Positive Rate (FPR)**.  
- For multi-class data, the **One-vs-Rest (OvR)** method is applied: each class is treated as “positive” while others are “negative.”  
- **Macro-averaged AUC** summarizes overall separability across all classes:

$$
\text{Macro-AUC} = \frac{1}{K}\sum_{i=1}^{K} AUC_i
$$

### **2️⃣ ROC–AUC Results**
| Model | Macro AUC |
|--------|-----------:|
| **SVM** | 0.9913 |
| **KNN** | 0.9864 |
| Logistic Regression | 0.9775 |
| Naive Bayes | 0.9606 |
| Decision Tree | 0.9015 |
| Dummy | 0.6181 |

**Interpretation:**  
- **SVM** achieves the highest macro-AUC (≈ 0.99), showing excellent ability to distinguish between all six land-cover classes.  
- **Dummy Classifier** has AUC close to random guessing.  
- AUC < 0.5 (not seen here) would indicate inverted or misleading probability estimates.

---

## **Part C: Precision–Recall Curve (PRC) Analysis [20 points]**

### **Why PRC?**  
- ROC can appear optimistic under **class imbalance** since it considers true negatives.  
- PRC focuses solely on **positive detection performance**:  

$$
\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}
$$

- It better reveals trade-offs between finding positives and avoiding false alarms.

### **PRC–AP Results**
| Model | Average Precision (AP) |
|--------|----------------------:|
| **KNN** | 0.9225 |
| **SVM** | 0.9095 |
| Logistic Regression | 0.8228 |
| Naive Bayes | 0.8022 |
| Decision Tree | 0.7080 |
| Dummy | 0.1667 |

**Interpretation:**  
- **KNN** performs best in terms of precision–recall balance, indicating consistent confidence in positive predictions.  
- Poor models (like Dummy) exhibit flat PRC curves, where precision collapses quickly as recall rises.

---

## **Part D: Final Recommendation [5 points]**

### **Model Comparison Summary**
| Metric | Top Model | Observation |
|---------|-----------|-------------|
| Accuracy / F1 | **KNN** | Strong baseline performance |
| ROC–AUC | **SVM** | Best class discrimination |
| PRC–AP | **KNN** | Best precision–recall trade-off |

### **Recommendation**
- **Best Overall:** ✅ **KNN Classifier**  
  - Delivers top accuracy and PRC performance.  
  - Balanced across metrics with excellent generalization.  
- **Runner-up:** **SVM**, slightly better in ROC–AUC but marginally lower precision.  
- **Worst:** **Dummy Classifier**, confirming the lower performance bound.  

**Conclusion:**  
KNN and SVM are both high-performing classifiers.  
However, **KNN** provides the most stable balance between **precision, recall, and class separability**, making it the **recommended model** for multi-class land-cover classification.

---

## **Brownie Points: Ensemble and Inverted Models [+5 points]**

### **Extended Models**
| Model | Accuracy | Weighted F1 | Macro AUC | Average Precision |
|--------|-----------:|-------------:|-----------:|------------------:|
| **XGBoost** | 0.9171 | 0.9156 | 0.9944 | 0.9509 |
| **Random Forest** | 0.9171 | 0.9142 | 0.9937 | 0.9421 |
| **Inverteddummy** |	0.170378 |	0.175786 |	0.500000	| 0.166667 |
| **Flippedlogreg** |	0.858622 |	0.851291 |	0.014092 |	0.090171|


**Analysis:**
- **XGBoost** achieved the best overall performance (AUC ≈ 0.994, AP ≈ 0.951), outperforming all baseline models.  
- **Random Forest** also performed extremely well, highlighting the advantage of ensemble averaging.  
- The **Inverted Dummy** intentionally yields AUC ≈ 0.5, verifying that the evaluation framework correctly detects random or inverted performance.

**Final Takeaway:**  
- **XGBoost** is the most powerful model in this experiment.  
- Among simpler algorithms, **KNN** offers the best precision–recall balance, and **SVM** provides the strongest class separation.  

---

✅ **Final Summary:**  
This assignment demonstrates that **model evaluation using ROC and PRC curves** gives a much deeper understanding than relying on accuracy alone.  
Through careful multi-metric comparison, **KNN** and **SVM** emerge as robust classifiers, while **XGBoost** provides the best overall performance under ensemble learning.

---
