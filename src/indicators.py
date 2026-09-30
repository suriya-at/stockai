import numpy as np
import pandas as pd

def add_indicators(data: pd.DataFrame) -> pd.DataFrame:
    d = data.copy()
    c, h, l, v = d.Close, d.High, d.Low, d.Volume
    for kind, spans in (("EMA", (20,50,200)), ("SMA", (20,50,200))):
        for n in spans: d[f"{kind}_{n}"] = c.ewm(span=n, adjust=False).mean() if kind == "EMA" else c.rolling(n).mean()
    delta = c.diff(); gain = delta.clip(lower=0).rolling(14).mean(); loss = -delta.clip(upper=0).rolling(14).mean()
    d["RSI_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    fast, slow = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    d["MACD"] = fast - slow; d["MACD_Signal"] = d.MACD.ewm(span=9, adjust=False).mean(); d["MACD_Histogram"] = d.MACD-d.MACD_Signal
    d["BB_Middle"] = c.rolling(20).mean(); std=c.rolling(20).std(); d["BB_Upper"] = d.BB_Middle+2*std; d["BB_Lower"] = d.BB_Middle-2*std
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1); d["ATR_14"] = tr.rolling(14).mean()
    d["Volume_MA_20"] = v.rolling(20).mean(); d["Volume_Ratio"] = v/d.Volume_MA_20; d["Volume_Change"] = v.pct_change()
    d["Return_1D"] = c.pct_change(); d["Return_5D"] = c.pct_change(5); d["Return_20D"] = c.pct_change(20); d["Volatility_20"] = d.Return_1D.rolling(20).std()
    for days, name in ((1,"Next_Day_Return"),(5,"Return_5D_Target"),(20,"Return_20D_Target")): d[name] = c.shift(-days)/c-1
    return d.replace([np.inf,-np.inf], np.nan)
