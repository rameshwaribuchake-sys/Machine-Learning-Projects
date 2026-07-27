# 📊 Customer Churn Prediction Using Machine Learning

## 📌 Project Overview

Customer churn is a major challenge for banks and financial institutions. Losing customers can negatively impact business growth and revenue.

This project uses **Machine Learning** to predict whether a bank customer is likely to leave the bank. A **Random Forest Classifier** is trained on historical customer data to identify customers who are at risk of churning.

The model can help banks take proactive actions such as personalized offers, improved customer service, and customer retention campaigns.

---

## 🎯 Problem Statement

The bank aims to minimize customer attrition by identifying customers who are likely to discontinue their relationship with the bank.

Using historical customer data, this project develops a predictive Machine Learning model that enables proactive customer engagement and personalized retention strategies.

---

## 🚀 Objectives

* Predict whether a customer will leave the bank.
* Analyze customer characteristics related to churn.
* Preprocess categorical and numerical data.
* Train a Random Forest Classification model.
* Evaluate the model using performance metrics.
* Identify the most important features influencing customer churn.
* Save the trained model for future predictions.

---

## 📂 Dataset

The project uses the **Churn Modelling Dataset**.

### Important Features

* `CreditScore` – Customer's credit score
* `Geography` – Customer's country
* `Gender` – Customer's gender
* `Age` – Customer's age
* `Tenure` – Number of years with the bank
* `Balance` – Customer's bank balance
* `NumOfProducts` – Number of bank products used
* `HasCrCard` – Whether the customer has a credit card
* `IsActiveMember` – Whether the customer is an active member
* `EstimatedSalary` – Estimated customer salary

### Target Variable

* `Exited`

  * `0` → Customer stayed with the bank
  * `1` → Customer left the bank

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Jupyter Notebook
* Pickle

---

## 🔄 Project Workflow

1. **Data Collection**
2. **Data Loading**
3. **Data Exploration**
4. **Missing Value Checking**
5. **Categorical Data Encoding**
6. **Feature Selection**
7. **Train-Test Split**
8. **Feature Scaling**
9. **Random Forest Model Training**
10. **Prediction**
11. **Model Evaluation**
12. **Feature Importance Analysis**
13. **Model Saving and Reuse**

---

## 🤖 Machine Learning Algorithm

### Random Forest Classifier

Random Forest is an ensemble Machine Learning algorithm that combines multiple decision trees to make accurate predictions.

In this project, the Random Forest model is used to classify customers into two categories:

* **0 – Customer will stay**
* **1 – Customer will churn**

### Model Configuration

```python
RandomForestClassifier(
    n_estimators=100,
    criterion='gini',
    random_state=42
)
```

---

## 📊 Model Evaluation

The model is evaluated using:

### Accuracy

Measures the overall percentage of correct predictions.

### Confusion Matrix

Shows:

* True Positives
* True Negatives
* False Positives
* False Negatives

### Classification Report

Provides:

* Precision
* Recall
* F1-Score
* Support

---

## 📈 Feature Importance

The project analyzes the importance of different features using the Random Forest model.

Feature importance helps identify which customer attributes have the greatest influence on the prediction of customer churn.

A feature importance visualization is also generated using Matplotlib.

---

## 💾 Saved Model Files

The trained Machine Learning model and preprocessing object are saved using Pickle.

```text
random_forest_churn_model.pkl
scaler.pkl
```

These files can be loaded later to make predictions on new customer data without retraining the model.

---

## ▶️ How to Run the Project

### 1. Clone the Repository

```bash
git clone <your-repository-link>
```

### 2. Install Required Libraries

```bash
pip install pandas numpy scikit-learn matplotlib jupyter
```

### 3. Open the Jupyter Notebook

```bash
jupyter notebook
```

### 4. Run the Notebook

Open:

```text
Python_Implementation_for_churn_prediction.ipynb
```

Make sure the dataset file is present in the same folder:

```text
Churn_Modelling.csv
```

---

## 📁 Project Structure

```text
Customer-Churn-Prediction/
│
├── Churn_Modelling.csv
├── Python_Implementation_for_churn_prediction.ipynb
├── random_forest_churn_model.pkl
├── scaler.pkl
└── README.md
```

---

## 🌍 Real-World Applications

This project can be used by banks and financial institutions to:

* Identify high-risk customers.
* Improve customer retention.
* Provide personalized offers.
* Reduce customer loss.
* Improve customer relationship management.
* Support data-driven business decisions.

---

## 🔮 Future Scope

* Develop a web-based dashboard using Streamlit.
* Add real-time customer churn prediction.
* Compare multiple Machine Learning algorithms.
* Improve model performance using hyperparameter tuning.
* Add explainable AI techniques.
* Deploy the model as a web application or API.

---

## 👩‍💻 Author

**Rameshwari Buchake**

Artificial Intelligence and Data Science Engineering Student

---

## ⭐ Conclusion

The Customer Churn Prediction project demonstrates how Machine Learning can be used to identify customers who are likely to leave a bank.

Using data preprocessing, feature scaling, and a Random Forest Classifier, the project provides a data-driven approach to customer retention and business decision-making.
