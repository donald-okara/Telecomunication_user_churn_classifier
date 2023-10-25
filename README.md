# Telecomunication_user_churn_classifier

This repository contains code for a telecommunication user churn classifier. The goal of this project is to predict whether a telecommunication user will churn (leave the service) or stay with the company.

## 1. Data Mining

The initial section of the code focuses on data mining and data loading. It imports necessary libraries such as pandas, numpy, matplotlib, and seaborn. The code reads the data from a CSV file named "telecom_customer_churn.csv" and stores it in a DataFrame called 'df'.

## 2. Data Cleaning

In this section, data cleaning operations are performed. The code separates the numerical and categorical columns using a custom function from the 'preprolib' library. Certain columns are also added to an 'ignore_list', and the function 'cat_or_num' helps identify which columns are categorical and which are numerical. The code then proceeds to plot count plots for each categorical variable with respect to the 'Customer Status' to analyze churn behavior. It also displays histograms and a heatmap to analyze the correlation between numerical variables.

## 2.1 EDA
Here I explore the features and how they affect each other and the target variable. They are separated into Numerical, ordinal and nominal variables.
#### 2.1.1 Numeric Data
Here I explore the numeric features and how they affect each other and the target variable.

### 2.1.2 Categorical Data
Here I explore the categorical features and how they affect each other and the target variable.


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


# Telecom User Churn Dashboard
## Overview

This is a comprehensive guide to the Telecom User Churn Dashboard, which is designed for data analysis and visualization of customer churn data. This dashboard is built using the Dash web application framework, Plotly for interactive visualizations, and scikit-learn for machine learning. The dashboard allows users to explore the distribution of customer churn, analyze features affecting churn, and view model evaluation metrics.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Data Preparation](#data-preparation)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis)
4. [Machine Learning Model](#machine-learning-model)
5. [Dashboard Features](#dashboard-features)
6. [Running the Dashboard](#running-the-dashboard)

## Getting Started <a name="getting-started"></a>

### Prerequisites

Before running the dashboard, ensure you have the following installed:

- Python (3.7 or later)
- Dash (Dash is a Python web application framework for creating interactive, web-based data visualizations)
- Plotly (for creating interactive visualizations)
- pandas (for data manipulation)
- scikit-learn (for machine learning)
- Jupyter Notebook (for running Python scripts)

You can install the required Python libraries using pip:

```bash
pip install dash plotly pandas scikit-learn jupyter
```

### Installation

1. Clone the repository or download the code.
2. Ensure you have a CSV file with telecom customer churn data. Update the file path in the code as needed.

## Data Preparation <a name="data-preparation"></a>

### Data Cleaning and Transformation

- The data is read from the CSV file, and the "Customer Status" column is transformed to binary, marking customers as either "Churned" or "Not Churned."

- Numerical and categorical columns are separated. You can specify columns to ignore in the `ignore_list`, as they might not be relevant to the analysis.

- Specific columns are removed from the numerical and categorical column lists as per your requirements.

### Feature Engineering

- Features and the label (target) are defined. Features include both categorical and numerical columns.

- The label column is converted to ordinal categories, allowing it to be used for classification.

### Data Splitting

- The dataset is split into training and testing sets. The train-test split ratio is 70-30.

### Data Preprocessing

- Pipelines are defined for preprocessing numerical and categorical columns. Numerical data is imputed with mean values and standardized, while categorical data is imputed with the most frequent value and encoded.

- A ColumnTransformer is created to apply the preprocessing to both numerical and categorical columns.

- The preprocessing is applied to both the training and test data.

## Exploratory Data Analysis (EDA) <a name="exploratory-data-analysis"></a>

### Visualizing Data

- The dashboard provides various interactive visualizations for EDA. These include:

    - Distribution of the target variable ("Customer Status") - a bar chart showing the number of churned and non-churned customers.
    
    - Distribution of other columns (histograms) - histograms for selected numerical and categorical columns.
    
    - Comparison plots of features versus the target variable - box plots comparing numerical features against customer status.

    - Categorical feature comparison (pie charts) - pie charts comparing categorical features against customer status.

### Model Evaluation Metrics

- Model evaluation metrics, including accuracy, precision, recall, and F1-score, are displayed in the dashboard.

## Machine Learning Model <a name="machine-learning-model"></a>

- A RandomForestClassifier is trained using the preprocessed training data.

- The model is evaluated using standard machine learning metrics, and the confusion matrix is displayed.

- Feature importance is calculated, and the top features affecting customer churn are visualized in a bar chart.

## Dashboard Features <a name="dashboard-features"></a>

- The dashboard includes a user-friendly interface with the following components:

    - Dropdowns for selecting features and customer status for analysis.
    
    - A bar chart displaying the distribution of customer status.
    
    - Histograms for selected columns.
    
    - Box plots for numeric features versus customer status.
    
    - Pie charts comparing categorical features against customer status.
    
    - A confusion matrix plot for model evaluation.
    
    - A bar chart showing feature importance.
    
    - Display of cross-validation scores and model evaluation metrics.

## Running the Dashboard <a name="running-the-dashboard"></a>

To run the dashboard, execute the script in your Python environment:

```bash
python [your_script_name].py
```

The dashboard will run locally, and you can access it via your web browser.

## Conclusion

The Telecom User Churn Dashboard is a powerful tool for analyzing customer churn data, exploring feature importance, and evaluating machine learning models. It provides an interactive and user-friendly interface to gain insights into customer behavior and make data-driven decisions.
