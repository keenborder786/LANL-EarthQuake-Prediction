# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 01:59:02 2019

@author: MMOHTASHIM
"""
###loaded the data
from tensorflow import keras
from tensorflow.keras.models import load_model
import os
import pickle
import xgboost as xgb
os.chdir(r'C:\Users\MMOHTASHIM\Anaconda3\libs\Small Data Science projects\Small-Data-Science-Projects\LANL Earthuquake prediction')
bst = xgb.Booster({'nthread':4}) #init model
bst.load_model('mainxgboost.model')



###predicted the data
from tqdm import tqdm
from scipy.stats import kurtosis,skew
from statistics import mode,mean
import pandas as pd
import numpy as np
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


os.chdir(r'C:\Users\MMOHTASHIM\Anaconda3\libs\Small Data Science projects\Small-Data-Science-Projects\LANL Earthuquake prediction\test_data')
predicted={}
f=[]
for file in os.listdir(os.getcwd()):
    f.append(file)
for filename in tqdm(f):
    df=pd.read_csv(filename)
    for newcol in ["Mean per 150k","std per 150k","min per 150k","max per 150k","Median per 150k",
              "Mean","std","min","max","Median","Moving Average Overallwindow5",
              "Moving Average Overallwindow10",'Ratio Succession_window5','Ratio Succession_window10',
              'kurtosis per 150k',"Mode per 150k","skew per 150k","quantile0.95 per 150k","quantile0.05 per 150k","quantile0.25 per 150k","quantile0.75 per 150k"]:
        df[newcol]=0
    df['Mean per 150k']=df["acoustic_data"].rolling(15).mean()
    df['std per 150k']=df["acoustic_data"].rolling(15).std()
    df['min per 150k']=df["acoustic_data"].rolling(15).max()
    df['max per 150k']=df["acoustic_data"].rolling(15).min()
    df['Median per 150k']=df["acoustic_data"].rolling(15).median()
    df['Mean']=np.mean(df["acoustic_data"])
    df['std']=np.std(df["acoustic_data"])
    df['min']=np.min(df["acoustic_data"])
    df['max']=np.max(df["acoustic_data"])
    df['Median']=np.median(df["acoustic_data"])
    df["Moving Average Overallwindow5"]=df["acoustic_data"].rolling(5).mean()
    df["Moving Average Overallwindow10"]=df["acoustic_data"].rolling(10).mean()
    df['Ratio Succession_window5']=df["acoustic_data"].pct_change(periods=5)
    df['Ratio Succession_window10']=df["acoustic_data"].pct_change(periods=10)
    df['kurtosis per 150k']=kurtosis(df["acoustic_data"])
    try:
         df["Mode per 150k"]=mode(df["acoustic_data"])
    except:
         df["Mode per 150k"]=0
    df['skew per 150k']=skew(df["acoustic_data"])
    df['quantile0.95 per 150k']=np.quantile(df["acoustic_data"],0.95)
    df['quantile0.05 per 150k']=np.quantile(df["acoustic_data"],0.05)
    df['quantile0.25 per 150k']=np.quantile(df["acoustic_data"],0.25)
    df['quantile0.75 per 150k']=np.quantile(df["acoustic_data"],0.75)
    
    
    df=df.replace([np.inf, -np.inf], np.nan).dropna()
    X=np.array(df)
    preprocessing_pipeline = Pipeline(steps=[
      ('scaler', StandardScaler())])
    X_transformed=preprocessing_pipeline.fit_transform(X)  
    dtest=xgb.DMatrix(X_transformed)  
    y_pred=np.array(bst.predict(dtest))
    
    predicted[str(filename)]=np.mean(y_pred)
#####################################make the submission file for kaggle
os.chdir(r'C:\Users\MMOHTASHIM\Anaconda3\libs\Small Data Science projects\Small-Data-Science-Projects\LANL Earthuquake prediction')
df_sub=pd.read_csv("sample_submission.csv")
s=-1
for i in df_sub["seg_id"]:
         s+=1
         df_sub.iloc[s,1]=predicted[str(i)+".csv"]
df_sub.to_csv("submission.csv")
##shutdown the system
#import os
#os.system('shutdown -s')