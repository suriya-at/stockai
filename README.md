# Stock AI Research Platform

An educational Streamlit application for researching multiple Yahoo Finance tickers with historical market data, technical indicators, Google News RSS, Gemini headline sentiment, and XGBoost return estimates. It does **not** provide investment advice or guaranteed price predictions.

## Architecture

```
Ticker → Yahoo Finance OHLCV → indicators → optional daily news features
      → XGBoost return models → chronological walk-forward testing → dashboard
Google News RSS → per-ticker CSV → Gemini → valid-only daily aggregation ┘
```

The legacy scripts remain in `venv/` because that was their original location. The application source is deliberately outside the virtual environment, in `src/`.

| Module | Responsibility |
|---|---|
| `src/market_data.py` | Robust download, MultiIndex normalisation, ticker validation |
| `src/indicators.py` | Point-in-time technical features and trading-day return targets |
| `src/news.py` | Dynamic Google News RSS collection and link de-duplication |
| `src/sentiment.py` | Retried Gemini JSON sentiment; failures remain `failed` |
| `src/features.py` | Market-session-aware daily aggregation and feature joining |
| `src/models.py` | XGBoost and transparent zero-return baseline metrics |
| `src/backtesting.py` | Expanding-window chronological validation |
| `app.py` | Streamlit dashboard |

## Install (Windows PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Gemini API setup

Create a restricted Gemini API key in [Google AI Studio](https://aistudio.google.com/app/apikey), then set it for the current PowerShell window:

```powershell
$env:GEMINI_API_KEY = "your-key-here"
# Optional: choose another enabled Gemini model
$env:GEMINI_MODEL = "gemini-2.5-flash"
```

Restart Streamlit after setting the variable. Do not add the key to `settings.py`, a CSV file, or Git. Gemini requests use the API-key header and JSON-mode content generation, as described in the [official Gemini API reference](https://ai.google.dev/api).

Before starting Streamlit, run the key-safe connection diagnostic:

```powershell
venv\Scripts\python.exe test_gemini.py
```

It lists only models available to the configured key and performs one minimal generation request. It never prints a key. If both `GEMINI_API_KEY` and `GOOGLE_API_KEY` are set, the application reports the conflict and explicitly uses `GEMINI_API_KEY`; remove an obsolete `GOOGLE_API_KEY` if it is not needed elsewhere.

## Run

```powershell
streamlit run app.py
```

Choose a preset ticker or type any Yahoo Finance-supported symbol. Click **Refresh News** to fetch and analyse recent articles with Gemini, then **Analyze Stock**. The app never downloads data solely because Streamlit reran; analysis occurs only after the button click.

To research several stocks, choose the extra tickers under **Additional stocks to analyse** and click **Analyze Selected Stocks**. It processes only your selected tickers: Yahoo Finance data, Google News collection, Gemini headline sentiment, and its own XGBoost/backtest workflow. This may take several minutes and uses Gemini API quota for every new article. The resulting table deliberately shows metrics without ranking or investment recommendations. A bare NSE code such as `CUPID` is automatically tried as `CUPID.NS`.

## Data and leakage rules

- Yahoo `end` is treated as exclusive, so a requested custom end date is advanced one calendar day. The dashboard uses the final downloaded row as the latest available trading date—it does not claim today is a trading session.
- `yfinance` empty responses and changing MultiIndex column shapes produce a readable error rather than a crash.
- Indicators use current and prior OHLCV values only. Targets use `shift(-1/-5/-20)`, which represents future trading days and is never included in features.
- Article timestamps are converted to Asia/Kolkata. Articles after 15:30 IST are assigned to the following calendar session; this avoids treating after-close news as known at that day’s close. A production deployment should replace the calendar-day rollover with an exchange holiday calendar for exact holiday handling.
- News failures are stored with `Status=failed` and an error message. They are excluded from aggregation—not labelled neutral.
- Missing news has `News_Available=0`; zero-valued aggregate columns should never be interpreted as historical neutral sentiment. The technical model is the default because short RSS history cannot support a credible multi-year news model.

## Modeling and evaluation

Three separate XGBoost regressors estimate next-day, 5-trading-day, and 20-trading-day returns. Estimated price is `latest close × (1 + estimated return)`. Metrics include MAE, RMSE, directional accuracy, precision, recall, and F1; each is shown beside a zero-return baseline.

Walk-forward testing uses expanding historical training windows followed by later test blocks. There is no random shuffle. The model comparison table shows technical-only and technical-plus-news feature sets when enough valid, aligned news exists; it does not declare a winner.

## Storage

News files are separated by ticker, e.g. `data/news/RELIANCE_NS/news.csv` and `data/news/TCS_NS/sentiment.csv`, preventing cross-company contamination. `data/models/` and `data/stocks/` are reserved for future persisted model/data caches.

## Limitations

Yahoo Finance can be delayed or temporarily unavailable. Google News RSS provides limited recent history, not a multi-year news archive. Gemini scores headlines only and can fail if the API key, quota, model, or network is unavailable. Market data and predictive metrics are historical research outputs, not a recommendation to buy, sell, or hold any security.
