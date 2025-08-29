# Credit Card Fraud Detection Classifier

This project focuses on building a machine learning classifier to detect fraudulent credit card transactions. The dataset used was obtained from Kaggle’s Credit Card Fraud Detection dataset
. The dataset is highly imbalanced, with fraudulent cases being significantly fewer than legitimate ones.

# 📂 Project Overview

Preprocessed and split dataset using train-test split with stratification to preserve class distribution.

Trained and compared multiple models:

Logistic Regression

Support Vector Machine (SVM)

Random Forest

XGBoost

Performed hyperparameter tuning on XGBoost using RandomizedSearchCV.

# 📊 Results

Best performing model: XGBoost

Evaluation metrics on test data:

Recall: 82%

F1 Score: 0.88

These results highlight the importance of recall in fraud detection, where minimizing false negatives is crucial to prevent fraudulent activity from going undetected.

# 🛠️ Tools & Libraries

Python

Pandas, NumPy (data manipulation)

Scikit-learn (model training & evaluation)

XGBoost (optimized gradient boosting)

Matplotlib, Seaborn (visualization)

# 💡 Key Learnings

Learned strategies for handling imbalanced datasets.

Compared model trade-offs between interpretability and predictive power.

Understood the value of tuning hyperparameters for boosting models.

Emphasized recall over accuracy for high-stakes classification tasks.

# 🔮 Future Considerations

To further improve performance and robustness:

Deep Learning Approaches: Experiment with neural networks (e.g., simple MLP or autoencoders for anomaly detection).

Sampling Methods: Apply SMOTE, undersampling, or hybrid methods to balance the dataset.

Evaluation Metrics: Include ROC-AUC and Precision-Recall AUC, which are better suited for imbalanced datasets.

Feature Engineering: Explore derived features or time-series aspects of transactions.

Deployment: Build an API or stream-based detection system for real-time fraud detection.
