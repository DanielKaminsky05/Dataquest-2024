import pandas as pd
import os
import matplotlib as plt
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


files = pd.ExcelFile("data_UPDATED.csv.xlsx")
trainData = pd.read_excel(files, sheet_name='train', nrows = 1000)
gender = {"M": 0, "F": 1}
trainData.gender = [gender[item] for item in trainData.gender]
print(trainData.head().to_string())


