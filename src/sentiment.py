from __future__ import annotations
import logging
from pathlib import Path
import pandas as pd
from .news import ticker_path, news_file
from .gemini_sentiment import analyze_sentiment, get_gemini_status

log = logging.getLogger(__name__)

def sentiment_file(ticker: str) -> Path:
    return ticker_path(ticker) / "sentiment.csv"

def analyze_unprocessed(ticker: str, company: str, limit: int = 20) -> pd.DataFrame:
    """Analyze news headlines that have not yet succeeded with Gemini API.
    
    Failed articles are intentionally retried on subsequent calls.
    Only articles with Status == 'ok' are considered complete.
    """
    n_file = news_file(ticker)
    if not n_file.exists():
        return pd.DataFrame(columns=["Link", "Ticker", "Date", "Title", "Source", "Sentiment", "Score", "Model", "Status", "Error"])

    news = pd.read_csv(n_file)
    if news.empty:
        return pd.DataFrame(columns=["Link", "Ticker", "Date", "Title", "Source", "Sentiment", "Score", "Model", "Status", "Error"])

    path = sentiment_file(ticker)
    if path.exists():
        old = pd.read_csv(path)
    else:
        old = pd.DataFrame(columns=["Link", "Ticker", "Date", "Title", "Source", "Sentiment", "Score", "Model", "Status", "Error"])

    # Successfully analyzed links
    done = set(old.loc[old.Status.eq("ok"), "Link"].dropna())
    unprocessed = news.loc[~news.Link.isin(done)].head(limit)

    if unprocessed.empty:
        return old

    rows = []
    for _, a in unprocessed.iterrows():
        title = a.Title
        link = a.Link
        date = a.Date
        source = a.Source

        label, score, model_used, error = analyze_sentiment(title, company)
        if error:
            rows.append({
                "Link": link,
                "Ticker": ticker,
                "Date": date,
                "Title": title,
                "Source": source,
                "Sentiment": None,
                "Score": None,
                "Model": model_used,
                "Status": "failed",
                "Error": error
            })
        else:
            rows.append({
                "Link": link,
                "Ticker": ticker,
                "Date": date,
                "Title": title,
                "Source": source,
                "Sentiment": label,
                "Score": score,
                "Model": model_used,
                "Status": "ok",
                "Error": ""
            })

    new_df = pd.DataFrame(rows)
    if not old.empty:
        combined = pd.concat([old, new_df], ignore_index=True)
    else:
        combined = new_df

    result = combined.drop_duplicates("Link", keep="last")
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)
    return result
