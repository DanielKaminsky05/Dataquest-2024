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


print("Beginning the process")
files = pd.ExcelFile("data_UPDATED.csv.xlsx")
train_data = pd.read_excel(files, "train")
train_data.dropna()
#train_data = train_data.drop(['creditCardNum', 'business', 'firstName',  'lastName',  'transNum', 'category', 'transDate', 'merchLongitude'], axis = 1)

y = train_data['isFraud']
gender = {"M": 0, "F": 1}
train_data.gender = [gender[item] for item in train_data.gender]


train_data_features = ["business", ]
x= train_data[train_data_features]
train_x,val_x,train_y,val_y = train_test_split(x,y,random_state=0)
print("Describe: ", x.describe())
print("Head: ", x.head())

for max_leaf_nodes in [5, 50, 500, 5000, 50000]:
    my_mae = get_mae(max_leaf_nodes, train_x, val_x, train_y, val_y)
    print("Max leaf nodes: %d \t\t Mean absolute error: %f \t\t Avg Prediction %f" %(1000, my_mae[0], my_mae[1]))

print("Done")
