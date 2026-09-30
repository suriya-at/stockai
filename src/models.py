from __future__ import annotations
import numpy as np, pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_recall_fscore_support
from settings import XGB_PARAMS

TECH_FEATURES=["Close","Volume","EMA_20","EMA_50","EMA_200","SMA_20","SMA_50","SMA_200","RSI_14","MACD","MACD_Signal","MACD_Histogram","BB_Upper","BB_Middle","BB_Lower","ATR_14","Volume_MA_20","Volume_Ratio","Volume_Change","Return_1D","Return_5D","Return_20D","Volatility_20"]
TARGETS={"1-Day":"Next_Day_Return","5-Day":"Return_5D_Target","20-Day":"Return_20D_Target"}
def metrics(actual, predicted, baseline=None):
    direction=actual > 0; pred_dir=predicted > 0; p,r,f,_=precision_recall_fscore_support(direction,pred_dir,average="binary",zero_division=0)
    answer={"MAE":mean_absolute_error(actual,predicted),"RMSE":mean_squared_error(actual,predicted)**.5,"Directional Accuracy":accuracy_score(direction,pred_dir),"Precision":p,"Recall":r,"F1":f}
    if baseline is not None: answer.update({"Baseline MAE":mean_absolute_error(actual,baseline),"Baseline RMSE":mean_squared_error(actual,baseline)**.5,"Baseline Directional Accuracy":accuracy_score(direction,baseline>0)})
    return answer
def train_evaluate(data, features, target):
    clean=data[features+[target]].replace([np.inf,-np.inf],np.nan).dropna()
    if len(clean)<180: raise ValueError("At least 180 complete trading rows are needed to train a model.")
    split=max(120,int(len(clean)*.8)); train,test=clean.iloc[:split],clean.iloc[split:]
    model=XGBRegressor(**XGB_PARAMS).fit(train[features],train[target]); pred=model.predict(test[features])
    # zero return is a transparent naive baseline.
    return model,metrics(test[target].to_numpy(),pred,np.zeros(len(test))),clean
def predict_latest(data, features):
    out={}
    for horizon,target in TARGETS.items():
        model,score,clean=train_evaluate(data,features,target); row=data[features].replace([np.inf,-np.inf],np.nan).dropna().iloc[[-1]]
        ret=float(model.predict(row)[0]); close=float(data.loc[row.index[0],"Close"])
        out[horizon]={"return":ret,"price":close*(1+ret),"metrics":score,"model":model}
    return out
