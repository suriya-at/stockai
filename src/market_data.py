from __future__ import annotations
import functools
import logging
import pandas as pd
import requests
import yfinance as yf

log = logging.getLogger(__name__)

class MarketDataError(Exception):
    """Exception raised when market data retrieval or ticker resolution fails."""
    pass

@functools.lru_cache(maxsize=256)
def search_tickers(query: str) -> list[dict]:
    """Search Yahoo Finance for tickers matching a query string."""
    query = query.strip()
    if not query:
        return []
    try:
        url = "https://query2.finance.yahoo.com/v1/finance/search"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        params = {"q": query, "quotesCount": 10, "newsCount": 0}
        resp = requests.get(url, params=params, headers=headers, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            results = []
            for q in data.get("quotes", []):
                sym = q.get("symbol")
                if sym:
                    name = q.get("shortname") or q.get("longname") or sym
                    exchange = q.get("exchange", "")
                    qtype = q.get("quoteType", "")
                    results.append({"symbol": sym, "name": name, "exchange": exchange, "type": qtype})
            return results
    except Exception as exc:
        log.warning("Yahoo search request failed for query '%s': %s", query, exc)
    return []

@functools.lru_cache(maxsize=256)
def get_company_info(symbol: str) -> dict:
    """Fetch company metadata (name, country, exchange) for a symbol."""
    symbol = symbol.strip().upper()
    info = {"symbol": symbol, "name": symbol, "exchange": "", "country": ""}
    try:
        t = yf.Ticker(symbol)
        t_info = t.info or {}
        name = t_info.get("longName") or t_info.get("shortName") or symbol
        country = t_info.get("country") or ("India" if symbol.endswith((".NS", ".BO")) else "United States" if "." not in symbol else "")
        exchange = t_info.get("exchange") or ("NSE" if symbol.endswith(".NS") else "BSE" if symbol.endswith(".BO") else "")
        info = {"symbol": symbol, "name": name, "exchange": exchange, "country": country}
    except Exception:
        pass
    return info

def get_company_name(symbol: str) -> str:
    """Return the company name for a given ticker symbol."""
    info = get_company_info(symbol)
    return info.get("name") or symbol

@functools.lru_cache(maxsize=256)
def verify_ticker(symbol: str) -> tuple[bool, dict | None]:
    """Verify if a ticker has valid historical price data on Yahoo Finance."""
    symbol = symbol.strip().upper()
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period="5d")
        if hist is not None and not hist.empty and len(hist) > 0:
            info = get_company_info(symbol)
            return True, info
    except Exception:
        pass
    return False, None

def resolve_ticker(query: str) -> tuple[str, bool, str]:
    """Dynamically resolve a user search query to a valid Yahoo Finance ticker.
    
    Returns:
        (resolved_ticker, is_valid, message)
    """
    query = query.strip()
    if not query:
        return "", False, "Please enter a stock ticker or company name."

    cleaned = query.upper()
    candidates = []

    # 1. Exact cleaned input
    if cleaned:
        candidates.append(cleaned)

    # 2. Yahoo Finance search results
    search_res = search_tickers(query)
    for s in search_res:
        sym = s["symbol"].strip().upper()
        if sym not in candidates:
            candidates.append(sym)

    # 3. Append Indian market suffixes (.NS, .BO) if no suffix present
    if "." not in cleaned:
        ns_sym = f"{cleaned}.NS"
        bo_sym = f"{cleaned}.BO"
        if ns_sym not in candidates:
            candidates.append(ns_sym)
        if bo_sym not in candidates:
            candidates.append(bo_sym)

    # Validate candidates in sequence
    for cand in candidates:
        ok, info = verify_ticker(cand)
        if ok and info:
            name = info.get("name", cand)
            country = info.get("country", "")
            exchange = info.get("exchange", "")
            meta = f"{name}"
            if exchange:
                meta += f" ({exchange})"
            if country:
                meta += f" · {country}"
            return cand, True, f"Resolved: {cand} — {meta}"

    return (
        query,
        False,
        f"Could not find a valid Yahoo Finance ticker for '{query}'.\n"
        f"Try entering a ticker such as TCS.NS, AAPL, INFY.NS, or RELIANCE.NS.",
    )

def download_history(ticker: str, start=None, end=None, period="5y") -> pd.DataFrame:
    """Download OHLCV history from Yahoo Finance and validate non-emptiness."""
    ticker = ticker.strip().upper()
    try:
        t = yf.Ticker(ticker)
        if start and end:
            df = t.history(start=start, end=end, auto_adjust=True)
        else:
            df = t.history(period=period, auto_adjust=True)

        if df is None or df.empty:
            raise MarketDataError(f"No historical price data returned for ticker '{ticker}'. The ticker may be invalid or delisted.")

        required = ["Open", "High", "Low", "Close", "Volume"]
        for col in required:
            if col not in df.columns:
                raise MarketDataError(f"Historical price data for '{ticker}' is missing required column '{col}'.")

        df = df[required].copy()
        df = df.dropna()
        if len(df) < 30:
            raise MarketDataError(f"Insufficient historical data for '{ticker}' ({len(df)} trading days available). At least 30 trading days are required.")

        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df
    except Exception as exc:
        if isinstance(exc, MarketDataError):
            raise exc
        raise MarketDataError(f"Failed to download market data for '{ticker}': {exc}")
