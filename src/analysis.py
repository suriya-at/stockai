from __future__ import annotations
import pandas as pd
from .market_data import download_history
from .indicators import add_indicators
from .features import merge_news, NEWS_FEATURES
from .models import TECH_FEATURES, TARGETS, predict_latest, train_evaluate
from .backtesting import walk_forward

def run_analysis(ticker, start=None, end=None, period="5y", include_news=False):
    raw=download_history(ticker,start,end,period); data=merge_news(add_indicators(raw),ticker)
    selected_features = TECH_FEATURES + NEWS_FEATURES if include_news else TECH_FEATURES
    tech_results=predict_latest(data,selected_features)
    comparison=[]
    for name,features in (("Technical only",TECH_FEATURES),("Technical + news",TECH_FEATURES+NEWS_FEATURES)):
        for horizon,target in TARGETS.items():
            try: _,m,_=train_evaluate(data,features,target); comparison.append({"Model":name,"Horizon":horizon,**m})
            except ValueError as e: comparison.append({"Model":name,"Horizon":horizon,"Error":str(e)})
    return {"raw":raw,"data":data,"predictions":tech_results,"feature_set":"Technical + news" if include_news else "Technical only","comparison":pd.DataFrame(comparison),"backtest":walk_forward(data,selected_features,"Next_Day_Return")}
