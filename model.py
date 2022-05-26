
# -*- coding: utf-8 -*-
"""
Created on Fri May 13 13:22:36 2022

@author:mayuri
"""
# importing the libraries
import pandas as pd               # for Data Manipulation
import numpy as np                #for Mathematical calculations
# Importing sample dataset using pandas
sample = pd.read_excel(r"D:\PROJECT DS\sampla_data_08_05_2022(final).xlsx")
sample.head() # It shows the last five rows of the dataset.

sample.tail()    # It shows the last five rows of the dataset.

sample.columns  # it shows the column names of the dataset


sample.info()

sample.describe()

# Missing values
sample.isnull().sum()


# using label encoder to convert the categorical(Class Variable) column to numerical
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
sample["Reached_On_Time"] = lb.fit_transform(sample["Reached_On_Time"])
sample['Patient_Gender']=lb.fit_transform(sample['Patient_Gender'])
sample['Test_Name']=lb.fit_transform(sample['Test_Name'])
sample['Sample']=lb.fit_transform(sample['Sample'])
sample['Way_Of_Storage_Of_Sample']=lb.fit_transform(sample['Way_Of_Storage_Of_Sample'])
sample['Cut_off_Schedule']=lb.fit_transform(sample['Cut_off_Schedule'])
sample['Traffic_Conditions']=lb.fit_transform(sample['Traffic_Conditions'])
sample['Mode_Of_Transport']=lb.fit_transform(sample['Mode_Of_Transport'])

sample1 = sample.drop(['Patient_ID','Patient_Age','Test_Booking_Date','Sample_Collection_Date','Agent_ID','Mode_Of_Transport'],axis=1)
sample1.columns

import matplotlib.pyplot as plt

plt.scatter(x = sample1 ['Time_Taken_To_Reach_Lab_MM'],y=sample1['Reached_On_Time'], color = 'blue') 
plt.scatter(x=sample ['Time_Taken_To_Reach_Lab_MM'],y=sample1['Reached_On_Time'], color = 'blue')

import seaborn as sns



sns.pairplot(sample1.iloc[:, :])

sns.jointplot(x=sample1['Time_Taken_To_Reach_Lab_MM'],y=sample1['Reached_On_Time'])
sns.jointplot(x=sample1['Cut-off time_HH_MM'], y=sample1['Reached_On_Time'])
sns.jointplot(x=sample1['Scheduled_Sample_Collection_Time_HH_MM'],y=sample1['Reached_On_Time'])
sns.jointplot(x=sample1['Test_Booking_Time_HH_MM'], y=sample1['Reached_On_Time'])


# Input and Output Split
predictors = sample1.iloc[:, :-1]
target = sample1.iloc[:,-1]

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.25, random_state=0)

from sklearn.tree import DecisionTreeClassifier as DT
model = DT(criterion = 'entropy')
model.fit(x_train, y_train)

# Prediction on Test Data
preds = model.predict(x_test)
pd.crosstab(y_test, preds, rownames=['Actual'], colnames=['Predictions'])

# Test Data Accuracy 
from sklearn.metrics import accuracy_score, confusion_matrix
accuracy_score(y_test, model.predict(x_test))

# Prediction on Train Data
preds = model.predict(x_train)
pd.crosstab(y_train, preds, rownames = ['Actual'], colnames = ['Predictions'])

# Train Data Accuracy 
from sklearn.metrics import accuracy_score, confusion_matrix
accuracy_score(y_train, model.predict(x_train))

#random forest classifier
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=42)

rf_clf.fit(x_train, y_train)

# Test Data Accuracy 
from sklearn.metrics import accuracy_score, confusion_matrix
print(confusion_matrix(y_test, rf_clf.predict(x_test)))
accuracy_score(y_test, rf_clf.predict(x_test))

# Train Data Accuracy
accuracy_score(y_train, rf_clf.predict(x_train))

# model is overfitting
# Creating new model testing with new parameters
forest_new = RandomForestClassifier(n_estimators=100,max_depth=10,min_samples_split=20,criterion='gini')  # n_estimators is the number of decision trees
forest_new.fit(x_train, y_train)

print('Train accuracy: {}'.format(forest_new.score(x_train, y_train)))

print('Test accuracy: {}'.format(forest_new.score(x_test, y_test)))

# saving the model
# importing pickle
import pickle
pickle.dump(model, open('model.pkl', 'wb'))

# load the model from disk
model = pickle.load(open('model.pkl', 'rb'))

# checking for the results
list_value = pd.DataFrame(sample1.iloc[0:1,:14])
list_value

print(model.predict(list_value))
