# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 17:42:36 2019

@author: MMOHTASHIM
"""
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
import catboost


os.chdir(r'C:\Users\MMOHTASHIM\Anaconda3\libs\Small Data Science projects\Small-Data-Science-Projects\LANL Earthuquake prediction')
bst = xgb.XGBRegressor() #init model
bst.load_model('mainxgboost.model')
m = CatBoostRegressor(iterations=10000, loss_function='MAE', boosting_type='Ordered')
cbt=m.load_model("catboost.model")

os.chdir(r'C:\Users\MMOHTASHIM\Anaconda3\libs\Small Data Science projects\Small-Data-Science-Projects\LANL Earthuquake prediction\test_data')
X_test=[]
y_pred=[]
predicted={}
def gen_features(X):
    strain = []
    strain.append(X.mean())
    strain.append(X.std())
    strain.append(X.min())
    strain.append(X.max())
    strain.append(X.kurtosis())
    strain.append(X.skew())
    strain.append(np.quantile(X,0.01))
    strain.append(np.quantile(X,0.05))
    strain.append(np.quantile(X,0.95))
    strain.append(np.quantile(X,0.99))
    strain.append(np.abs(X).max())
    strain.append(np.abs(X).mean())
    strain.append(np.abs(X).std())
    strain.append(np.abs(X.max()-X.min()))
    strain.append(X.median())
    return strain
for file in tqdm(os.listdir(os.getcwd())):
    df=pd.read_csv(file)
    X_test=np.array(gen_features(df["acoustic_data"])).reshape(1,-1)
    y_pred=np.array(cbt.predict(X_test))        
    predicted[file]=y_pred
    
    
os.chdir(r'C:\Users\MMOHTASHIM\Anaconda3\libs\Small Data Science projects\Small-Data-Science-Projects\LANL Earthuquake prediction')
df_sub=pd.read_csv("sample_submission.csv")
s=-1
for i in df_sub["seg_id"]:
         s+=1
         df_sub.iloc[s,1]=predicted[str(i)+".csv"]
df_sub.to_csv("submission.csv")

    

    