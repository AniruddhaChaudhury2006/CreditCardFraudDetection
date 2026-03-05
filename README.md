# Credit Card Fraud Detection

This project aims to detect fraudulent credit card transactions using machine learning techniques. Due to the highly imbalanced nature of credit card transaction datasets (where fraudulent transactions are very rare compared to legitimate ones), special attention is paid to data balancing during preprocessing.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [How to Run](#how-to-run)
- [Technologies Used](#technologies-used)

## Introduction
Credit card fraud is a significant problem that causes substantial financial losses for individuals and institutions. This project develops a machine learning model to identify fraudulent transactions, thereby helping to mitigate these risks. We focus on building a classification model that can accurately distinguish between legitimate and fraudulent transactions.

## Dataset
The dataset used for this project is `creditcard.csv`, which contains anonymized credit card transaction data. It includes 30 features (V1-V28, Time, Amount) and a target variable `Class`, where `0` represents a legitimate transaction and `1` represents a fraudulent one.

**Key characteristics of the dataset:**
- **Highly Imbalanced:** The dataset is heavily skewed, with a very small percentage of transactions being fraudulent.
  - Legitimate transactions: 225419
  - Fraudulent transactions: 415
- **Features `V1` to `V28`:** These are the principal components obtained from PCA transformation, protecting user identities.
- **`Time`:** The seconds elapsed between each transaction and the first transaction in the dataset.
- **`Amount`:** The transaction amount.

## Methodology

### 1. Data Preprocessing
- **Handling Missing Values:** Checked for and addressed any missing values in the dataset. (One row was found with NaNs in `V23` through `Class`).
- **Data Imbalance Handling (Undersampling):** To address the severe class imbalance, undersampling was applied to the majority class (legitimate transactions). A sample of legitimate transactions equal to the number of fraudulent transactions (415) was randomly selected.
- **Concatenation:** The undersampled legitimate transactions and all fraudulent transactions were combined to create a new, balanced dataset.

### 2. Model Training
- **Splitting Data:** The balanced dataset was split into features (X) and target (Y). This data was then further divided into training and testing sets using `train_test_split` with a 80/20 ratio, ensuring stratification to maintain the class distribution.
- **Model Selection:** Logistic Regression was chosen as the classification model due to its interpretability and effectiveness in binary classification tasks.
- **Training:** The Logistic Regression model was trained on the preprocessed training data.

### 3. Model Evaluation
- The model's performance was evaluated using accuracy score on both the training and testing datasets.

## Results

- **Training Data Accuracy:** 94.58%
- **Test Data Accuracy:** 92.17%

The model demonstrates good accuracy on both the training and test sets, indicating its ability to generalize to unseen data despite the initial imbalance challenge. A predictive system was successfully built to classify new transactions as normal or fraudulent.

## How to Run
1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```
2.  **Install dependencies:**
    ```bash
    pip install pandas numpy scikit-learn
    ```
3.  **Place the `creditcard.csv` file** in the appropriate directory (e.g., `/content/creditcard.csv` as used in the notebook).
4.  **Run the Jupyter Notebook/Colab file.**

## Technologies Used
-   Python
-   Pandas (for data manipulation)
-   NumPy (for numerical operations)
-   Scikit-learn (for machine learning models and utilities)
