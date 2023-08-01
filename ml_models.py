#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ranjan Kumar Barik
"""


import pandas as pd
import numpy as np
import sys
import os
import seaborn as sns
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
import matplotlib.pyplot as plt
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE



from scipy import interp
from sklearn.linear_model import LinearRegression,Ridge,LassoCV,Lasso,ARDRegression,ElasticNet
from numpy import set_printoptions
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import RandomizedSearchCV
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared, RBF,Matern, RationalQuadratic,DotProduct
from sklearn.gaussian_process.kernels import ConstantKernel
#import seaborn as sns
import sys, traceback
from sklearn import datasets, linear_model, metrics, gaussian_process,svm 
from sklearn.linear_model import LinearRegression,Ridge,LassoCV,Lasso,ARDRegression,ElasticNet,MultiTaskLasso
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.svm import NuSVR
#import xgboost as xgb
import sys
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis





model_adh = GradientBoostingRegressor(random_state=3, n_estimators=500, learning_rate=0.009529999999999816, max_depth=6, min_samples_split= 3, max_features=12)
model_cor = GradientBoostingRegressor(random_state=214403, learning_rate=0.012799999999999683, n_estimators = 500, max_depth=5, max_features=12)
model_vdw =  XGBRegressor(n_estimators=1000, learning_rate=0.01, max_depth=4)

rs_adh=1700
rs_cor=101528
rs_vdW=2265


def ranking(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))


def fitting_ml(data, prop):    
    
    X=data.iloc[:,0:-1]
    
    Y=data.iloc[:,[-1]]
    
    columns_=X.columns
    
    
    if prop=="adh":
        X_train1, X_test1, y_train, y_test = train_test_split(X,Y,test_size=0.1,random_state=rs_adh)
    
        X_train1 = np.nan_to_num(X_train1)
        X_test1 = np.nan_to_num(X_test1)
          
          
        X_train = pd.DataFrame(data=X_train1, columns=columns_)
        X_test = pd.DataFrame(data=X_test1, columns=columns_)
        
        X_data = X_train.iloc[:,1:].values
        headers=X_train.iloc[:,1:].columns
        scalar=StandardScaler()
        scalar.fit(X_data)
        X_train_std=pd.DataFrame(scalar.transform(X_data))  
        X_train_std.columns=headers
        
        
        X_data_test=X_test.iloc[:,1:].values                
        X_test_std=pd.DataFrame(scalar.transform(X_data_test))                
        X_test_std.columns=headers
           
        model = model_adh
        model.fit(X_train_std,y_train.values.ravel())
        
        title="results of adhesion energy model :"

        
    if prop=="cor":
        X_train1, X_test1, y_train, y_test = train_test_split(X,Y,test_size=0.1,random_state=rs_cor)
    
        X_train1 = np.nan_to_num(X_train1)
        X_test1 = np.nan_to_num(X_test1)
          
          
        X_train = pd.DataFrame(data=X_train1, columns=columns_)
        X_test = pd.DataFrame(data=X_test1, columns=columns_)
        
        X_data = X_train.iloc[:,1:].values
        headers=X_train.iloc[:,1:].columns
        scalar=StandardScaler()
        scalar.fit(X_data)
        X_train_std=pd.DataFrame(scalar.transform(X_data))  
        X_train_std.columns=headers
        
        
        X_data_test=X_test.iloc[:,1:].values                
        X_test_std=pd.DataFrame(scalar.transform(X_data_test))                
        X_test_std.columns=headers
           
        model = model_cor
        model.fit(X_train_std,y_train.values.ravel()) 
        
        title="results of corrugation energy model :"

    if prop=="vdW":
        X_train1, X_test1, y_train, y_test = train_test_split(X,Y,test_size=0.1,random_state=rs_vdW)
    
        X_train1 = np.nan_to_num(X_train1)
        X_test1 = np.nan_to_num(X_test1)
          
          
        X_train = pd.DataFrame(data=X_train1, columns=columns_)
        X_test = pd.DataFrame(data=X_test1, columns=columns_)
        
        X_data = X_train.iloc[:,1:].values
        headers=X_train.iloc[:,1:].columns
        scalar=StandardScaler()
        scalar.fit(X_data)
        X_train_std=pd.DataFrame(scalar.transform(X_data))  
        X_train_std.columns=headers
        
        
        X_data_test=X_test.iloc[:,1:].values                
        X_test_std=pd.DataFrame(scalar.transform(X_data_test))                
        X_test_std.columns=headers
           
        model = model_vdw
        model.fit(X_train_std,y_train.values.ravel())
        
        title="results of van der Waals energy model :"

    
    y_train_pred=model.predict(X_train_std)
    y_test_pred=model.predict(X_test_std)
    
    R2_test=metrics.r2_score(y_test, y_test_pred)
    R2_train=metrics.r2_score(y_train, y_train_pred)
    
    rmse_test=np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))
    rmse_train=np.sqrt(metrics.mean_squared_error(y_train,y_train_pred))
    
    
    #print (R2_train,R2_test,rmse_train,rmse_test)
    
    return title, scalar, model, X_train, X_train_std, X_test, X_test_std, y_train, y_train_pred, y_test, y_test_pred, R2_train, R2_test, rmse_train, rmse_test   




def scattered_plot_ml(y_train, y_train_pred, y_test, y_test_pred):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.scatter(y_train.values.ravel(), y_train_pred, color = "dodgerblue", edgecolors='blue', s = 60, label = 'Train data')
    ax.scatter(y_test.values.ravel(), y_test_pred, color = 'lime', edgecolors='darkgreen', s = 60, label = 'Test data')
    
    ar_max=[float(max(y_train.values.ravel())), float(max(y_train_pred)), float(max(y_test.values.ravel())), float(max(y_test_pred))]
    ar_min=[float(min(y_train.values.ravel())), float(min(y_train_pred)), float(min(y_test.values.ravel())), float(min(y_test_pred))]
    
    y_max=float(max(ar_max))
    y_min=float(min(ar_min))
    
    xl=[y_min-1, y_max + 1]
    yl=[y_min-1, y_max + 1]
    ax.plot(yl,xl,linestyle='dashed',color='grey', linewidth=1.5)
    ax.set_xlim(xl[0], xl[1])
    ax.set_ylim(yl[0], yl[1])
    ax.set_xlabel('DFT calculated Y',fontsize=25)
    ax.set_ylabel('ML predicted Y', fontsize=25)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.legend()
    plt.tight_layout()
    plt.show()
    #plt.savefig('scattered.eps')
    
def residual_error(y_train, y_train_pred, y_test, y_test_pred): 
    
    diff=[]
    x1=[float(y_train_pred[n]-y_train.values.ravel()[n]) for n in range(len(y_train_pred))]
    x2=[float(y_test_pred[n]-y_test.values.ravel()[n]) for n in range(len(y_test_pred))]
    
    for i in x1:
        diff.append(i)
    for i in x2:
        diff.append(i)
        
    #diff_abs=[abs(i) for i in diff]
    
    sns.set(rc={"figure.figsize": (8, 8)}); np.random.seed(0)

    sns_plot=sns.distplot(diff, vertical=True, bins=80, norm_hist=True, kde=True)
    sns_plot.grid(True)
    for axis in ['top','bottom','left','right']:
        sns_plot.spines[axis].set_linewidth(2)
    sns_plot.set(ylim=(-.4, 0.4))
    sns_plot.set_xlabel("Count")
    sns_plot.set_ylabel(r'$|Y_{True} - Y_{Pred}|$', fontsize=20)
    sns_plot.tick_params(labelsize=10)
    #sns_plot.set(xlim=(-0.1, 8))
    #fig = sns_plot.get_figure()
    
    plt.show()
    #fig.savefig("res-elemental.eps")
    
def feature_relative_import(model, X_std):
    feature_names=X_std.columns
    ## Plot feature importanc
    feature_importance = model.feature_importances_
    # make importances relative to max importance
    feature_importance = feature_importance / feature_importance.max()
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, feature_names[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()
    #plt.savefig('feature_importance.eps')
    
def shap_interpretabilty(model, X_std):
    explainerModel = shap.TreeExplainer(model = model)
    shap_values = explainerModel.shap_values(X_std)
    
    shap.summary_plot(shap_values, X_std)
    
def shap_dependence_plot(model, X_train1, X_std):
    X_train=X_train1.iloc[:,1:]
    explainer = shap.TreeExplainer(model, X_std)
    shap_values = explainer(X_std)
    
    for ctr, col in enumerate(X_train.columns):
        shap_values.feature_names[ctr] = col
        
    shap_values.data = X_train.values
    
    for l in range(len(X_train.columns)):
        shap.plots.scatter(shap_values[:, l], hist=False, color=shap_values, show=False)
        plt.tight_layout()
        plt.show()

def shap_force_plot(model, X_train, X_train_std, X_test, X_test_std, row_num, data):
    bilayers=data.bilayers.values
    for _num in row_num:
        sel_bilayer=bilayers[_num - 2]
        if sel_bilayer in X_train.bilayers.values:
            X=X_train.iloc[:,1:]
            _index=int((np.where(X_train.bilayers.values == sel_bilayer))[0])
            explainerModel = shap.TreeExplainer(model = model)
            shap_values = explainerModel.shap_values(X_train_std)
            
            shap.initjs()
            shap.force_plot(explainerModel.expected_value, shap_values[_index], X.iloc[[_index]].astype(float).round(2), show=False, matplotlib=True, figsize=(18, 3), plot_cmap='RdBu')
            plt.title(sel_bilayer, loc='left')
            plt.tight_layout()
            plt.show()
            
        if sel_bilayer in X_test.bilayers.values:
            X=X_test.iloc[:,1:]
            _index=int((np.where(X_test.bilayers.values == sel_bilayer))[0])
            explainerModel = shap.TreeExplainer(model = model)
            shap_values = explainerModel.shap_values(X_test_std)
            
            shap.initjs()
            shap.force_plot(explainerModel.expected_value, shap_values[_index], X.iloc[[_index]].astype(float).round(2), show=False, matplotlib=True, figsize=(18, 3), plot_cmap='RdBu')
            plt.title(sel_bilayer, loc='left')
            plt.tight_layout()
            plt.show()
        


def newdata_pred(scalar, model, X):
    X_std=pd.DataFrame(scalar.transform(X))
    X_std.columns=X.columns
    y_pred=model.predict(X_std)
    X['y_pred']=y_pred
    X.to_csv('./test/X_new_y_pred.csv',index=False)
    return print ('Predicted outputs are \n',y_pred)


def mean_ranking_score(data, prop):
    ranks = {}
    X=data.iloc[:,1:-1]
    y1=data.iloc[:,-1]
    columns_=X.columns
    
    X_data = X.values
    scalar=StandardScaler()
    scalar.fit(X_data)
    X_tr1=pd.DataFrame(scalar.transform(X_data))  
    X_tr1.columns=columns_
    
    X_tr2=np.nan_to_num(X_tr1)
    X_tr = pd.DataFrame(data=X_tr2,columns=columns_)    
    
    rfe = RFE(estimator=DecisionTreeRegressor(), n_features_to_select=6)
    rfe.fit(X_tr,y1)
    ranks["RFE"] = ranking(list(map(float, rfe.ranking_)), columns_, order=-1)
    
    lr = LinearRegression(normalize=True)
    lr.fit(X_tr,y1)
    ranks["LinReg"] = ranking(np.abs(lr.coef_), columns_)
    
    ridge = Ridge()
    ridge.fit(X_tr,y1)
    ranks['Ridge'] = ranking(np.abs(ridge.coef_), columns_)
    
    lasso = Lasso(alpha=.001)
    lasso.fit(X_tr,y1)
    ranks["Lasso"] = ranking(np.abs(lasso.coef_), columns_)
    
    
    rf = RandomForestRegressor(n_estimators=200)
    rf.fit(X_tr,y1)
    ranks["RF"] = ranking(rf.feature_importances_, columns_);
    
    
    xg = XGBRegressor(n_estimators=1000, learning_rate=0.01)
    xg.fit(X_tr,y1)
    ranks["XGB"] = ranking(xg.feature_importances_, columns_)
    
    
    gb = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.01)
    gb.fit(X_tr,y1)
    ranks["XGB"] = ranking(gb.feature_importances_, columns_)
    
    
    r = {}
    for name in columns_:
        r[name] = round(np.mean([ranks[method][name]
                                 for method in ranks.keys()]), 2)
    
    methods = sorted(ranks.keys())
    ranks["Mean"] = r
    methods.append("Mean")    

    meanplot = pd.DataFrame(list(r.items()), columns= ['Feature','Mean Ranking'])
    # Sort the dataframe
    meanplot = meanplot.sort_values('Mean Ranking', ascending=False)
    
    print (meanplot)
    
    print ('\n Features having higher Mean ranking score are taken >= 0.28. \n then high linearly correlated features are eliminated')
    
    if prop == "adh":
        scores=[0.68, 0.66, 0.53, 0.44, 0.41, 0.40, 0.37, 0.34, 0.30, 0.28, 0.28, 0.28]
        col=['PF',  r'$\overline{M}_{int, NN}$', r'$\overline{N}_{p, v}$',
        r'$\overline{EA}_{int}$', r'$\overline{r}_{int, NN}^{atom}$', 'A', r'$\overline{Gr}_{int}$',
        r'$\overline{r}_{int}^{cov}$', r'$\rho$', r'$\overline{\alpha}_{int}$',
        r'$\overline{\kappa}_{int}$', 'd']
        scores.reverse()
        col.reverse()
        plt.barh(col, scores, align='center')
        plt.xlabel('Mean ranking score')
        plt.show()
    
    if prop == "cor":
        scores=[0.53, 0.52, 0.49, 0.46, 0.44, 0.41, 0.41, 0.38, 0.36, 0.35, 0.35, 0.32, 0.32, 0.31, 0.28]
        
        col=[r'$\overline{M}_{int}$', r'$\overline{MV}_{int, NN}$', r'$\epsilon_{r}$', 
             r'$\overline{IE}_{int}$', r'$\overline{\alpha}_{int, NN}$', r'$\overline{BL}_{min}$',
             r'$\overline{BL}_{max}$', r'$\overline{N}_{p, v}$', 'PF', r'$\overline{Gr}_{int, NN}$',
             r'$\overline{C}_{int}^{p, g}$', 'a', r'$\overline{P}_{int, NN}$', r'$\overline{\kappa}_{int}$',
             r'$\rho$']
        scores.reverse()
        col.reverse()
        plt.barh(col, scores, align='center')
        plt.xlabel('Mean ranking score')
        plt.show()
        
    if prop == "vdW":
        scores=[0.62, 0.59, 0.52, 0.42, 0.41, 0.41, 0.36, 0.35, 0.31, 0.30, 0.29, 0.28]
        
        col=['PF', r'$\overline{EA}_{int}$', r'$\overline{N}_{p, v}$', r'$\overline{Gr}_{int}$',
           'A', r'$\overline{\alpha}_{int}$', 'd$_{layer}$', r'$\overline{M}_{int, NN}$',
           r'$\rho$', r'$\overline{C}_{int}^{p, g}$', r'$\overline{MV}_{int, NN}$',
           r'$\overline{AM}_{int, NN}$']
        scores.reverse()
        col.reverse()
        plt.barh(col, scores, align='center')
        plt.xlabel('Mean ranking score')
        plt.show()



def input_input_correlation(data, prop):
    if prop == "cor":
        col=[r'$\overline{M}_{int}$', r'$\overline{MV}_{int, NN}$', r'$\epsilon_{r}$', 
             r'$\overline{IE}_{int}$', r'$\overline{\alpha}_{int, NN}$', r'$\overline{BL}_{min}$',
             r'$\overline{BL}_{max}$', r'$\overline{N}_{p, v}$', 'PF', r'$\overline{Gr}_{int, NN}$',
             r'$\overline{C}_{int}^{p, g}$', 'a', r'$\overline{P}_{int, NN}$', r'$\overline{\kappa}_{int}$',
             r'$\rho$']
        
        X = data[col]
        
        X_data = X.values
        headers=X.columns
        scalar=StandardScaler()
        scalar.fit(X_data)
        X_std=pd.DataFrame(scalar.transform(X_data))  
        X_std.columns=headers
        
        corr = X_std[headers].corr()
        
        plt.figure(figsize=(8, 8))
        #plt.savefig('correlation1.eps')
        sns.heatmap(
            corr, 
            vmin=-1, vmax=1, center=0,
            cmap="coolwarm",
        #    cmap=sns.diverging_palette(20, 220, n=200),
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}
        )
        
        plt.tight_layout()
        plt.show()
        #plt.savefig('correlation-ii-original.eps')
    else :
        X_data = data.iloc[:,1:-1].values
        headers=data.iloc[:,1:-1].columns
        scalar=StandardScaler()
        scalar.fit(X_data)
        X_std=pd.DataFrame(scalar.transform(X_data))  
        X_std.columns=headers
        
        corr = X_std[headers].corr()
        
        plt.figure(figsize=(8, 8))
        #plt.savefig('correlation1.eps')
        sns.heatmap(
            corr, 
            vmin=-1, vmax=1, center=0,
            cmap="coolwarm",
        #    cmap=sns.diverging_palette(20, 220, n=200),
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}
        )
        
        plt.tight_layout()
        plt.show()
        #plt.savefig('correlation-ii-original.eps')
    