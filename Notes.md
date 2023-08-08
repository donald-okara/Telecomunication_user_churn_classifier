# Notes

These are my notes for the telecommunication user churn project. Here I will note down my thoughts on EDA, deployment, model selection, and whatever else I think is relevant.

As of now, I am creating a model for Customer Status and Churn Category. The EDA approach so far has been using the relationship between the features and the target. I will now explore the relationship between the features themselves i.e. using PCA to reduce dimensionality.

## EDA inferences
### Numerical data
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
### Categorical Data
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

##### Before pruning categorical data
Model Evaluation: 
Accuracy: 0.8248816768086545 
Precision: 0.8181715735438497 
Recall: 0.8248816768086545 
F1-score: 0.8174064250075703

##### After
Model Evaluation: 
Accuracy: 0.8323191345503719 
Precision: 0.8273756347717985 
Recall: 0.8323191345503719 
F1-score: 0.8263798630584416