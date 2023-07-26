# Telecomunication_user_churn_classifier

This repository contains code for a telecommunication user churn classifier. The goal of this project is to predict whether a telecommunication user will churn (leave the service) or stay with the company.

## 1. Data Mining

The initial section of the code focuses on data mining and data loading. It imports necessary libraries such as pandas, numpy, matplotlib, and seaborn. The code reads the data from a CSV file named "telecom_customer_churn.csv" and stores it in a DataFrame called 'df'.

## 2. Data Cleaning

In this section, data cleaning operations are performed. The code separates the numerical and categorical columns using a custom function from the 'preprolib' library. Certain columns are also added to an 'ignore_list', and the function 'cat_or_num' helps identify which columns are categorical and which are numerical. The code then proceeds to plot count plots for each categorical variable with respect to the 'Customer Status' to analyze churn behavior. It also displays histograms and a heatmap to analyze the correlation between numerical variables.

## 2.1 EDA
Here we explore the features and how they affect each other and the target variable. They are separated into Numerical, ordinal and nominal variables.

## 2.2 Data Preprocessing

Data preprocessing is a crucial step in machine learning. The code performs data preprocessing tasks using pipelines from scikit-learn. It does the following:

1. Separates the features and the target label from the DataFrame.
2. Converts the target label ('Customer Status') to ordinal categories for model training purposes.
3. Splits the dataset into training and testing sets.
4. Creates separate pipelines for numeric and categorical columns to handle missing values and perform appropriate transformations.
5. Uses a ColumnTransformer to apply the pipelines to the respective columns and transforms the training and test data.

## 3. Model Selection

In this section, various machine learning models are trained and evaluated. The following models are used:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Decision Tree
4. Random Forest
5. Gaussian Naive Bayes
6. AdaBoost

Each model is trained on the preprocessed training data and then evaluated using metrics such as accuracy, precision, recall, and F1-score on the test data. The results are stored in a DataFrame and sorted by accuracy in descending order.

Work in progress
