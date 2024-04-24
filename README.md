# NTU-SC1015-Mini-Project
SC1015 mini project (diabetes)

### About
This is our mini project for SC1015 (FCS4 Team 11) - Introduction to Data Science and Artificial Intelligence which focuses on Dataset taken from Kaggle regarding the body matrix of patients with and without diabetes. It aims to help predict the outcome of whether one has diabetes and provides early diagnosis to prevent serious consequences of delayed treatment.

| Name | Parts Done | Github ID |
| --- | --- | --- |
| Benjamin Yeoh Jun Jie | Exploratory Data Analysis, Data Pre-processing , Classificaiton Decision Tree | @byeoh001|
| Baskar Athithiya Kayalvizhi| Exploratory Data Analysis, Data Pre-processing , Logistic Regression Model| @Athithiya02 |

### Problem Statement
Our problem statment is to determine which is more important in predicting diabetes, family history or current health conditions.

### Predictive Models Used
***
1. Classificiation Decision Tree <br>
- This is a supervised machine learning algorithm used for classification tasks, where it recursively splits the feature space into subsets, making decisions based on feature values to classify instances into different classes.
- Decision Making: Algorithm recursively splits the dataset into subsets based on feature values to maximize homogeneity of classes.
- Building Process: Continues splitting until a stopping criterion is met, like pure subsets or no significant improvement in classification.
- Data Handling: Can handle both numerical and categorical data, capturing non-linear relationships.
- Purpose: Used for classification tasks to predict the class of a given input based on its features. <br>

2. Logistic Regression Model <br>
- This is a supervised machine learning algorithm used for classification tasks, where it models the probability of a certain class or event occurring based on input features.
- Function: Models the probability of an input belonging to a certain class using a logistic function.
- Training: Learns the best coefficients by minimizing a loss function that measures the difference between predicted probabilities and actual labels.
- Output: Predicts probabilities between 0 and 1, indicating the likelihood of the input belonging to the positive class.
- Relationship: Assumes a linear relationship between features and the log-odds of the outcome. <br>

### Conclusion
***
- Current health is a more important factor than family history as the classification accuracy and metrics such as False Negative Rate (FNR) did not improve with the inclusion of the diabetes pedigree function.
- Among the current health conditions, skin thickness is the leading factor in predicting diabetes as it is the root node and it appears freqently in the top portion of the classification decision tree.

### What did we learn from this project?
***
- Filtering the raw datasets by removing the outliers and computing the mean.
- Imputation of the NULL values by replacing them with the previosuly calculated mean to maintain the integrity of the dataset.
- Classification Decision Tree from sklearn
- Another form of categorical output model which is Logistic Regression from sklearn
- Concepts about the confusion metrics such as True Positive Rate, True Negative Rate, False Positive Rate, False Negative Rate.
- 
### References
***
1.  https://www.diabetesatlas.org/data/en/country/179/sg.html
2.  https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
3.  https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
