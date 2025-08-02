# Titanic-Disaster-Dataset-Solution
Machine Learning Model using Decision Tree Classifier. 
# 1. Importing core data science and machine learning libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 2. Loading the CSV Data
test=pd.read_csv('test.csv')
train =pd.read_csv('train.csv')

# 3. Filling missing values
# Exploring and Preprocessing the Data

# Handling missing data: for example, Age and Fare have been filled with medians ; Cabin has been dropped due to too many missing values.

# categorical variables (like Sex, Embarked) have been Converted to numeric, using label encoding

# Irrelevant columns: such as Name, Ticket have been dropped ( PassengerId in test data is kept for the final submission).
train['Age'].fillna(train['Age'].median(), inplace=True)
train['Fare'].fillna(train['Fare'].median(), inplace=True)
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)
train.drop(['Name', 'Ticket'], axis=1, inplace=True)
train.head()

# 4. Selection of Features and Target
# Decision about which columns are features (X) and which column is target (y):
X = train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = train['Survived']

# 5. Spliting training data for internal validation:
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=54)

# 6. Initializing and Training the Model
# Creating a Decision Tree classifier and fit it:
clf = DecisionTreeClassifier(max_depth=3, random_state=54)
clf.fit(X_train, y_train)

# 7. Prediction and Evaluation on Validation Set
# Checking accuracy to evaluate model performance:

y_pred = clf.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_pred))

# 8. Preprocessing test data similar to train data
# Performing the same preprocessing on the test set, using the model for prediction, and preparing submission as required by Kaggle:
test['Age'].fillna(train['Age'].median(), inplace=True)
test['Fare'].fillna(train['Fare'].median(), inplace=True)
test['Sex'] = test['Sex'].map({'male': 0, 'female': 1})
test['Embarked'] = test['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

X_test = test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
test_predictions = clf.predict(X_test)

submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': test_predictions
})
submission.to_csv('submission_titanic.csv', index=False)
# Categorical variables have been converted
train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
train['Embarked'] = train['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
