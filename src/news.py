from __future__ import annotations
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote
import feedparser, pandas as pd
from settings import NEWS_DIR

def ticker_path(ticker: str) -> Path: return NEWS_DIR / ticker.replace("/", "_").replace(".", "_")
def news_file(ticker: str) -> Path: return ticker_path(ticker) / "news.csv"
def collect_news(ticker: str, company: str, days: int) -> pd.DataFrame:
    path = news_file(ticker); path.parent.mkdir(parents=True, exist_ok=True)
    query=quote(f'"{company}" stock when:{days}d'); feed=feedparser.parse(f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en")
    cutoff=datetime.now(timezone.utc)-timedelta(days=days); rows=[]
    for e in feed.entries:
        parsed=e.get("published_parsed")
        if not parsed: continue
        date=datetime(*parsed[:6],tzinfo=timezone.utc)
        if date >= cutoff: rows.append({"Ticker":ticker,"Date":date,"Title":e.get("title",""),"Source":getattr(e.get("source",{}),"get",lambda *_:"")("title", ""),"Link":e.get("link","")})
    old=pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["Ticker","Date","Title","Source","Link"])
    merged=pd.concat([old,pd.DataFrame(rows)],ignore_index=True).drop_duplicates("Link", keep="last")
    if not merged.empty: merged["Date"]=pd.to_datetime(merged.Date,utc=True); merged=merged.sort_values("Date",ascending=False)
    merged.to_csv(path,index=False); return merged
