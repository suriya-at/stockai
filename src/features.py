import pandas as pd
from .sentiment import sentiment_file
from .news import ticker_path

NEWS_FEATURES=["News_Sentiment","News_Count","Positive_News","Negative_News","Neutral_News","Positive_Ratio","Negative_Ratio","News_Available"]

def _empty_daily_news() -> pd.DataFrame:
    """Return an empty, date-indexed numeric frame for a ticker with no valid news."""
    return pd.DataFrame(
        {column: pd.Series(dtype="float64") for column in NEWS_FEATURES},
        index=pd.DatetimeIndex([], name="Date"),
    )

def daily_news(ticker):
    path=sentiment_file(ticker)
    output=ticker_path(ticker) / "daily_news.csv"
    if not path.exists(): return _empty_daily_news()
    d=pd.read_csv(path); d=d[d.Status.eq("ok")].copy()
    if d.empty:return _empty_daily_news()
    d["Date"]=pd.to_datetime(d.Date,utc=True).dt.tz_convert("Asia/Kolkata")
    # After 15:30 IST, news is only available to the next session.
    d["SessionDate"]=(d.Date.dt.normalize()+pd.to_timedelta((d.Date.dt.hour*60+d.Date.dt.minute>930).astype(int),unit="D")).dt.tz_localize(None)
    g=d.groupby("SessionDate"); out=g.agg(News_Sentiment=("Score","mean"),News_Count=("Score","count"),Positive_News=("Sentiment",lambda x:(x=="positive").sum()),Negative_News=("Sentiment",lambda x:(x=="negative").sum()),Neutral_News=("Sentiment",lambda x:(x=="neutral").sum()))
    out["Positive_Ratio"]=out.Positive_News/out.News_Count; out["Negative_Ratio"]=out.Negative_News/out.News_Count; out["News_Available"]=1; out.index.name="Date"
    out.reset_index().to_csv(output, index=False)
    return out
def merge_news(market, ticker):
    d=market.join(daily_news(ticker),how="left"); d["News_Available"]=d.News_Available.fillna(0)
    # Missing news stays missing; an explicit availability flag lets models distinguish it.
    for col in NEWS_FEATURES:
        # Empty CSV frames otherwise cause pandas to retain object dtype, which XGBoost rejects.
        d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0.0).astype("float64")
    return d
