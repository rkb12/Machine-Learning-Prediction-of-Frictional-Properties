#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ranjan Kumar Barik
"""

import pandas as pd
import numpy as np
import sys
import os

# import matplotlib.pyplot as plt
# import shap
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split 
# from sklearn import metrics
# from xgboost import XGBRegressor

import ml_models


data_adh = pd.read_csv('./data/ml_data_adh.csv')
data_cor = pd.read_csv('./data/ml_data_cor.csv')
data_vdw = pd.read_csv('./data/ml_data_vdW.csv')

data_adh_mfr = pd.read_csv('./data/ml_data_adh_for_mean_ranking_score.csv')
data_cor_mfr = pd.read_csv('./data/ml_data_cor_for_mean_ranking_score.csv')
data_vdw_mfr = pd.read_csv('./data/ml_data_vdW_for_mean_ranking_score.csv')


data=[data_adh, data_cor, data_vdw]
data1=[data_adh_mfr, data_cor_mfr, data_vdw_mfr]

target_properties=["adh", "cor", "vdW"]


def ml_metrics(result_model):
    #print (result_model[0], '\n')
    print ("R2 of train :", result_model[11])
    print ("R2 of test :", result_model[12])
    print ("RMSE of train :", result_model[13])
    print ("RMSE of test :", result_model[14], '\n')

def scattered_plot(result_model):
    return ml_models.scattered_plot_ml(result_model[7], result_model[8], result_model[9], result_model[10])

def error_in_ypred(result_model):
    return ml_models.residual_error(result_model[7], result_model[8], result_model[9], result_model[10])

def feature_import_by_model(result_model):
    return ml_models.feature_relative_import(result_model[2], result_model[4])

def shaply_analysis_overall(result_model):
    return ml_models.shap_interpretabilty(result_model[2], result_model[4])
    
def shaply_depend_indi_feature(result_model):
    return ml_models.shap_dependence_plot(result_model[2], result_model[3], result_model[4])

# def shaply_force_ind_mat(result_model, row_num, data):
#     return ml_models.shap_force_plot(result_model[1], result_model[2], row_num, data)

def shaply_force_ind_mat(result_model, row_num, data):
    return ml_models.shap_force_plot(result_model[2], result_model[3], result_model[4], result_model[5], result_model[6], row_num, data)

 
def new_data(result_model, X_new):
    return ml_models.newdata_pred(result_model[1], result_model[2], X_new)

def mfr(X, prop):
    return ml_models.mean_ranking_score(X, prop) 

def input_input_cor(X, prop):
    return ml_models.input_input_correlation(X, prop) 
  

row_num_adh=[625, 273, 223]
row_num_cor=[625, 273, 223]
row_num_vdW=[260, 696, 341]
row_num=[row_num_adh, row_num_cor, row_num_vdW]


sel_opt = input("Which frictional properties ml model do you want to study: select 1, 2, 3, or 4 \n 1 Adhesion energy \n 2 Corrugation energy \n 3 Van der Waals energy \n 4 Exit \n")


#result_model contains [title, scalar, fitted_model, X_train, X_train_std, X_test, X_test_std, y_train, y_train_pred, y_test, y_test_pred, R2_train, R2_test, rmse_train, rmse_test]

if sel_opt == "1" or sel_opt == "2" or sel_opt == "3": 
    sel_opt1 = input("Choose options from 1-10: \n 1 Features \n 2 ML metrics \t 3 scatterd plot  \t 4 residual error \n 5 feature importance by fitted model \t 6 shaply interpretation of features \n 7 shaply dependence plot for individual features \n 8 shaply force plot for invidual bilayers (you can enter row_number of bilayers) \n 9 Input x to predict y (keep your x in test directory with name X_new.csv) \n 10 exit \n")
    
    for _input in sel_opt:
        k=int(_input) - 1
        result_model= ml_models.fitting_ml(data[k], target_properties[k])

        if sel_opt1 == "1":
            sel_opt3 = input("\n 1 Mean feature ranking score \n 2 Pearson correlation among input features \n")
            
            if sel_opt3 == "1":
                mfr(data1[k], target_properties[k])
            if sel_opt3 == "2":
                input_input_cor(data[k], target_properties[k])
            
        if sel_opt1 == "2":
            ml_metrics(result_model)

        if sel_opt1 == "3":
            scattered_plot(result_model)

        if sel_opt1 == "4":
            error_in_ypred(result_model)

        if sel_opt1 == "5":
            feature_import_by_model(result_model)

        if sel_opt1 == "6":
            shaply_analysis_overall(result_model)

        if sel_opt1 == "7":
            shaply_depend_indi_feature(result_model)

        if sel_opt1 == "8":
            sel_opt2 = input(" 1 Want to input bilayer materials row numbers from data file in data directory \n 2 Want to see default bilayers results \n")

            if sel_opt2 == "1":
                    row_num1 = input("\n input the row numbers like 5 6 7 and so on \n") ## choose row_num as bilayers name row number from csv file in the data directory
                    row_num = [int(i) for i in row_num1.split()]
                    shaply_force_ind_mat(result_model, row_num, data[k])

            if sel_opt2 == "2":
                    shaply_force_ind_mat(result_model, row_num[k], data[k])

        if sel_opt1 == "9":
            X_new = pd.read_csv('./test/X_new.csv') ## keep your new data in test directory with name X_new.csv
            new_data(result_model, X_new)

        if sel_opt1 == "10":
            print ("Thank you, please cite us if you used our codes.")



if sel_opt == "4":
    print ("Thank you, please cite us if you used our code.")
    
if sel_opt != "1" and sel_opt != "2" and sel_opt != "3" and sel_opt != "4":
    print ("It is not a valid option, please try again.")

