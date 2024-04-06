import pandas as pd
import os
import matplotlib as plt
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error


def get_mae(max_leaf_nodes, train_x, val_x, train_y, val_y):
    total = 0
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_x, train_y)
    model_Predictions = model.predict(val_x)
    for item in model_Predictions:
        total += item
    result = total/len(model_Predictions)
    MAE = [mean_absolute_error(val_y, model_Predictions), result]
    return (MAE)


files = pd.ExcelFile("data_UPDATED.csv.xlsx")
trainData = pd.read_excel(files, sheet_name='train', nrows = 1000)
trainData = trainData.drop(['creditCardNum', 'business', 'firstName',  'lastName',  'transNum', 'street', 'category', 'transDate', 'job', 'merchLongitude', 'city', 'dateOfBirth', 'state'], axis = 1)
gender = {"M": 0, "F": 1}

trainData.gender = [gender[item] for item in trainData.gender]
y = trainData.isFraud
x= trainData
train_x,val_x,train_y,val_y = train_test_split(x,y,random_state=0)


my_mae = get_mae(1000,train_x,val_x,train_y,val_y)
print("Max leaf nodes: %d \t\t Mean absolute error: %f \t\t Avg Prediction %f" %(1000, my_mae[0], my_mae[1]))
print("Done")



