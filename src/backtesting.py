from __future__ import annotations
import numpy as np, pandas as pd
from xgboost import XGBRegressor
from .models import metrics
from settings import XGB_PARAMS
def walk_forward(data, features, target, min_train=252):
    clean=data[features+[target]].replace([np.inf,-np.inf],np.nan).dropna(); results=[]
    if len(clean)<min_train+60:return pd.DataFrame()
    # Expanding-window, chronological folds: no random shuffling.
    for end in range(min_train,len(clean)-1,max(60,(len(clean)-min_train)//3 or 1)):
        test=clean.iloc[end:min(end+60,len(clean))]; train=clean.iloc[:end]
        if test.empty:continue
        pred=XGBRegressor(**XGB_PARAMS).fit(train[features],train[target]).predict(test[features]); m=metrics(test[target].to_numpy(),pred,np.zeros(len(test)))
        results.append({"Train through":train.index[-1].date(),"Test start":test.index[0].date(),"Test end":test.index[-1].date(),**m})
    return pd.DataFrame(results)
