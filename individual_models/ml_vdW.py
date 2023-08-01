#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ranjan Kumar Barik
"""

import pandas as pd
import numpy as np
import sys
import os


import matplotlib.pyplot as plt
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from xgboost import XGBRegressor




features = pd.read_csv('../data/ml_data_vdW.csv')

X=features.iloc[:,1:-1]

Y=features.iloc[:,[-1]]

columns_=X.columns


X_train1, X_test1, y_train, y_test = train_test_split(X,Y,test_size=0.1,random_state=2265)

X_train1 = np.nan_to_num(X_train1)
X_test1 = np.nan_to_num(X_test1)
  
X_train = pd.DataFrame(data=X_train1, columns=columns_)
X_test = pd.DataFrame(data=X_test1, columns=columns_)


X_data = X_train.values
headers=X_train.columns
scalar=StandardScaler()
scalar.fit(X_data)
X_tr=pd.DataFrame(scalar.transform(X_data))  
X_tr.columns=headers


X_data_test=X_test.values                
X_te=pd.DataFrame(scalar.transform(X_data_test))                
X_te.columns=headers
   
model =  XGBRegressor(n_estimators=1000, learning_rate=0.01, max_depth=4)
model.fit(X_tr,y_train.values.ravel())

predicted_train=model.predict(X_tr)
predicted_test=model.predict(X_te)

R2_test=metrics.r2_score(y_test,predicted_test)
R2_train=metrics.r2_score(y_train,predicted_train)

rmse_test=np.sqrt(metrics.mean_squared_error(y_test,predicted_test))
rmse_train=np.sqrt(metrics.mean_squared_error(y_train,predicted_train))

print (" R2 train :",R2_train, "\n R2 test :", R2_test, "\n rmse train :", rmse_train, "\n rmse test :", rmse_test)
