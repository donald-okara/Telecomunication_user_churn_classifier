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

#### Inferences
##### Age
Most users from age 30 to 60 stayed, 35 to 65 churned and 30 to 50 just joined. For the most part, age is too close to call as a factor for churn.

##### Number of Dependents
The relationship between number of dependents and Customer status is linear as the more dependents there are the less likely to churn the user is. *To confirm after scaling.*

##### Number of Referrals
A relationship between number of referances and Customer status is also linear as the more a user refers other customers, the higher the likelyhood they stay. *To confirm this fact after scaling*

##### Tenure in Months
The relationship between Tenure in Months and Customer status is also linear as the longer the customer tenure the more likely they stay. *To confirm after smoothing and scaling*


##### Avg Monthly Long distance Charges
The facts provided by the graph is inconclusive. The uniformity of it is quite compelling but the graphs for avg monthly long distance charges against each respective status are identical. Might lead to overfitting.

##### Avg Monthly GB Download
Same as above. Plus tons of outliers.

##### Monthly Charge
The lower the monthly charge the more likely the user is to stay. Hence we must take our attention to the data plans columns to check which influences churn the most and also their details. (Should check if we are to discard this.)

##### Total Charges.
I am puzzled here. Monthly charge is a customers total charge per month and total charge calculated at the end of each quarter. Here, ths lower the charge the more chances of churn. *Further investigation to be done on the relationship and of course, smoothing*

##### Total Refunds
Interesting graph. Should smooth to get a better read.

##### Total Extra Data Charges
Interesting graph. Should smooth to get a better read.

##### Total Long Distance Charges
The higher the number of long distance charges the more likely they are to stay. Speaks to quality of long distance service.

##### Total Revenue 
The higher the revenue from a user the more likely they are to stay. Speaks to loyalty. 
#### 2.1.2 Categorical Data
Here I explore the categorical features and how they affect each other and the target variable.

#### Inferences

##### 'Gender',

Identical graphs. This column is inqonsequential

  

##### 'Married',

Identical graphs. This column is inqonsequential

  

##### 'Offer',

A large percentage of users in offer E leave. Followed by users subscrided to no offers. This column is to be coded ordinally based on percentage of churn per category

  

##### 'Phone Service',

People who pay for this service are more likely to stay. This column is to be coded ordinally based on percentage of churn per category.

##### 'Multiple Lines',

People who pay for this service are more likely to stay. This column is to be coded ordinally based on percentage of churn per category.

  

##### 'Internet Service',

People who pay for this service are more likely to stay. This column is to be coded ordinally based on percentage of churn per category.

  

##### 'Internet Type',

Further investigation to be done on this column

  

##### 'Online Security',

People who pay for this service are more likely to stay. This column is to be coded ordinally based on percentage of churn per category.

  

##### 'Online Backup',

People who pay for this service are more likely to stay. This column is to be coded ordinally based on percentage of churn per category.

  

##### 'Device Protection Plan',

People who pay for this service are more likely to stay. This column is to be coded ordinally based on percentage of churn per category.

  

##### 'Premium Tech Support',

People who pay for this service are more likely to stay. This column is to be coded ordinally based on percentage of churn per category.

  

##### 'Streaming TV',

People who pay for this service are more likely to stay. This column is to be coded ordinally based on percentage of churn per category.

  

##### 'Streaming Movies',

People who pay for this service are more likely to stay. This column is to be coded ordinally based on percentage of churn per category.

  

##### 'Streaming Music',

People who pay for this service are more likely to stay. This column is to be coded ordinally based on percentage of churn per category.

  

##### 'Unlimited Data',

People who pay for this service are more likely to stay. This column is to be coded ordinally based on percentage of churn per category.

  

##### 'Contract',

The longer the contract the more likely the user will stay. This column is to be coded ordinally based on percentage of churn per category.

  

##### 'Paperless Billing',

Further investigation is required

  

##### 'Payment Method'

Users who pay via mail are less common and are more likely to churn. This column is to be coded ordinally based on percentage of churn per category.


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
