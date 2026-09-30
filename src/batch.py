"""Batch orchestration for the supported-stock research universe."""
from __future__ import annotations

import pandas as pd

from .analysis import run_analysis
from .market_data import get_company_name, resolve_ticker
from .news import collect_news
from .sentiment import analyze_unprocessed


def analyze_selected_stocks(
    tickers: list[str], period: str, news_days: int, include_news: bool,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Collect Gemini-scored news and run research models for selected tickers.

    One failing ticker is reported in the table and does not stop the rest of
    the batch.  This intentionally presents metrics, not trading rankings.
    """
    rows: list[dict] = []
    details: dict[str, dict] = {}
    for requested_ticker in tickers:
        try:
            ticker, valid, message = resolve_ticker(requested_ticker)
            if not valid:
                raise ValueError(message)
            company = get_company_name(ticker)
            articles = collect_news(ticker, company, news_days)
            sentiments = analyze_unprocessed(ticker, company)
            result = run_analysis(ticker, period=period, include_news=include_news)
            details[ticker] = result
            latest = result["raw"].iloc[-1]
            one_day = result["predictions"]["1-Day"]
            valid = int(sentiments.Status.eq("ok").sum())
            rows.append({
                "Ticker": ticker,
                "Company": company,
                "Latest Date": result["raw"].index[-1].date(),
                "Latest Close": float(latest.Close),
                "1D Estimated Return": one_day["return"],
                "1D Estimated Price": one_day["price"],
                "1D Model MAE": one_day["metrics"]["MAE"],
                "1D Directional Accuracy": one_day["metrics"]["Directional Accuracy"],
                "Valid Gemini Headlines": valid,
                "Collected Articles": len(articles),
                "Status": "ok",
            })
        except Exception as exc:
            rows.append({"Ticker": requested_ticker, "Status": "failed", "Error": str(exc)})
    return pd.DataFrame(rows), details
