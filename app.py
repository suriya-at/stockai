from __future__ import annotations
import logging
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from settings import DEFAULT_TICKER
from src.market_data import resolve_ticker, get_company_name, get_company_info, search_tickers, MarketDataError
from src.news import collect_news, news_file
from src.sentiment import analyze_unprocessed, sentiment_file
from src.gemini_sentiment import get_gemini_status
from src.analysis import run_analysis
from src.batch import analyze_selected_stocks

st.set_page_config(
    page_title="Stock AI Research Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO)

@st.cache_data(ttl=900, show_spinner="Preparing market data and machine learning analysis...")
def cached_analysis(ticker: str, start=None, end=None, period: str = "5y", include_news: bool = False):
    """Avoid repeated downloads and model training during Streamlit reruns."""
    return run_analysis(ticker, start=start, end=end, period=period, include_news=include_news)

# Sidebar UI
with st.sidebar:
    st.title("📈 Stock AI")
    st.caption("AI-powered Stock Analysis & Gemini News Sentiment")
    
    st.subheader("Stock Search")
    user_query = st.text_input("Enter Ticker or Company Name", value="TCS", help="Examples: TCS, TCS.NS, AAPL, Apple, RELIANCE.NS")
    
    # Dynamic ticker resolution preview
    resolved_ticker, is_valid, resolve_msg = resolve_ticker(user_query)
    if is_valid:
        info = get_company_info(resolved_ticker)
        st.success(f"**{info['name']}**\n\nTicker: `{resolved_ticker}` | {info['country']} ({info['exchange']})")
    else:
        st.warning(resolve_msg)
        
    st.divider()
    st.subheader("Analysis Settings")
    range_name = st.selectbox("Historical Range", ["1y", "3y", "5y", "10y", "max", "Custom"], index=2)
    custom_start = custom_end = None
    if range_name == "Custom":
        custom_start = st.date_input("Start Date", value=pd.Timestamp.today() - pd.DateOffset(years=5))
        custom_end = st.date_input("End Date", value=pd.Timestamp.today())

    news_days = st.selectbox("News Lookback (Days)", [1, 3, 7, 14, 30], index=2)
    feature_mode = st.selectbox("Model Feature Set", ["Technical only", "Technical + news"])
    include_news = (feature_mode == "Technical + news")

    st.divider()
    analyze_btn = st.button("🚀 Analyze Stock", type="primary", use_container_width=True)
    refresh_btn = st.button("🔄 Refresh Data & News", use_container_width=True)

    st.divider()
    st.subheader("🤖 Gemini AI Status")
    gem_status = get_gemini_status()
    if gem_status["connected"]:
        st.success(f"**Gemini API:** Connected\n\n**Model:** `{gem_status['model']}`")
    else:
        st.error(f"**Gemini API:** Failed\n\n**Error type:** `{gem_status['error_type']}`\n\n{gem_status['error_message']}")
    st.caption(
        f"GEMINI_API_KEY: {'configured' if gem_status['api_key_configured'] else 'not configured'} | "
        f"GOOGLE_API_KEY: {'configured' if gem_status['google_api_key_configured'] else 'not configured'}"
    )
    if gem_status["key_conflict"]:
        st.warning("Both key variables are configured. This app explicitly uses GEMINI_API_KEY so GOOGLE_API_KEY cannot override it.")

    # Show headline statistics for resolved ticker if available
    if is_valid and sentiment_file(resolved_ticker).exists():
        s_df = pd.read_csv(sentiment_file(resolved_ticker))
        ok_count = int(s_df.Status.eq("ok").sum())
        fail_count = int(s_df.Status.eq("failed").sum())
        st.metric("Articles Analyzed", ok_count)
        if fail_count > 0:
            st.caption(f"⚠️ {fail_count} headlines pending/failed retry.")

    st.divider()
    st.subheader("Multi-Stock Comparison")
    compare_input = st.text_input("Add tickers to compare (comma separated)", value="AAPL, RELIANCE.NS")
    compare_list = [t.strip() for t in compare_input.split(",") if t.strip()]

# Action handling: Refresh Data & News
if refresh_btn:
    if not is_valid:
        st.error(resolve_msg)
    else:
        cached_analysis.clear()
        company = get_company_name(resolved_ticker)
        with st.spinner(f"Fetching news for {company} ({resolved_ticker})..."):
            articles = collect_news(resolved_ticker, company, news_days)
            sentiments = analyze_unprocessed(resolved_ticker, company)
            ok_cnt = int(sentiments.Status.eq("ok").sum()) if not sentiments.empty else 0
            fail_cnt = int(sentiments.Status.eq("failed").sum()) if not sentiments.empty else 0
        st.success(f"Collected {len(articles)} articles. {ok_cnt} analyzed with Gemini ({gem_status['model']}). {fail_cnt} failed/pending.")

# Main Dashboard
st.title("Stock AI Research Platform")
st.caption("Research and education platform combining XGBoost predictive modeling with Google Gemini financial sentiment.")

if not is_valid:
    st.error(f"### Invalid Stock Selection\n\n{resolve_msg}")
    st.stop()

# Auto-run analysis on load or when Analyze clicked
try:
    with st.spinner(f"Analyzing {resolved_ticker}..."):
        result = cached_analysis(resolved_ticker, custom_start, custom_end, range_name.lower(), include_news)
except (MarketDataError, ValueError) as exc:
    st.error(f"### Market Data Error\n\n{exc}")
    st.stop()

raw = result["raw"]
data = result["data"]
latest = raw.iloc[-1]
company_name = get_company_name(resolved_ticker)
info = get_company_info(resolved_ticker)

# Stock Header Card
m1, m2, m3, m4 = st.columns(4)
m1.metric("Company", company_name, f"Ticker: {resolved_ticker}")
m2.metric("Latest Close Price", f"{latest.Close:,.2f}", f"Volume: {int(latest.Volume):,}")
m3.metric("Exchange / Country", f"{info['exchange'] or '—'}", info['country'] or '—')
m4.metric("Latest Trading Date", str(raw.index[-1].date()))

st.divider()

# Interactive Price Chart
st.subheader("Price Chart & Moving Averages")
fig = go.Figure([
    go.Candlestick(
        x=raw.index, open=raw.Open, high=raw.High, low=raw.Low, close=raw.Close, name="OHLC Price"
    )
])
for col, color in [("EMA_20", "#1f77b4"), ("EMA_50", "#ff7f0e"), ("EMA_200", "#2ca02c")]:
    if col in data.columns:
        fig.add_scatter(x=data.index, y=data[col], name=col, line=dict(width=1.5))

fig.update_layout(
    height=500,
    xaxis_rangeslider_visible=False,
    template="plotly_dark",
    margin=dict(l=20, r=20, t=30, b=20),
)
st.plotly_chart(fig, use_container_width=True)

# Technical Indicators Tabs
st.subheader("Technical Analysis Indicators")
tab1, tab2, tab3 = st.tabs(["Bollinger Bands & ATR", "RSI & MACD Momentum", "Volume Analysis"])

with tab1:
    fig_bb = go.Figure()
    for name, color in [("Close", "#ffffff"), ("BB_Upper", "#e377c2"), ("BB_Middle", "#7f7f7f"), ("BB_Lower", "#e377c2")]:
        if name in data.columns:
            fig_bb.add_scatter(x=data.index, y=data[name], name=name, line=dict(width=1))
    if "ATR_14" in data.columns:
        fig_bb.add_scatter(x=data.index, y=data["ATR_14"], name="ATR 14", yaxis="y2", line=dict(color="#bcbd22", width=1.5))
    fig_bb.update_layout(
        yaxis2=dict(overlaying="y", side="right", title="ATR 14"),
        height=380,
        template="plotly_dark",
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(fig_bb, use_container_width=True)

with tab2:
    fig_mom = go.Figure()
    if "RSI_14" in data.columns:
        fig_mom.add_scatter(x=data.index, y=data["RSI_14"], name="RSI 14", line=dict(color="#17becf"))
        fig_mom.add_hline(y=70, line_dash="dot", line_color="red", annotation_text="Overbought (70)")
        fig_mom.add_hline(y=30, line_dash="dot", line_color="green", annotation_text="Oversold (30)")
    if "MACD_Histogram" in data.columns:
        colors = ["green" if val >= 0 else "red" for val in data["MACD_Histogram"].fillna(0)]
        fig_mom.add_bar(x=data.index, y=data["MACD_Histogram"], name="MACD Histogram", yaxis="y2", marker_color=colors)
    if "MACD" in data.columns:
        fig_mom.add_scatter(x=data.index, y=data["MACD"], name="MACD", yaxis="y2", line=dict(color="#ff7f0e"))
    if "MACD_Signal" in data.columns:
        fig_mom.add_scatter(x=data.index, y=data["MACD_Signal"], name="Signal", yaxis="y2", line=dict(color="#2ca02c"))
    fig_mom.update_layout(
        yaxis2=dict(overlaying="y", side="right", title="MACD"),
        height=380,
        template="plotly_dark",
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(fig_mom, use_container_width=True)

with tab3:
    fig_vol = go.Figure()
    fig_vol.add_bar(x=data.index, y=data["Volume"], name="Volume", marker_color="#1f77b4")
    if "Volume_MA_20" in data.columns:
        fig_vol.add_scatter(x=data.index, y=data["Volume_MA_20"], name="Volume MA 20", line=dict(color="#ff7f0e", width=2))
    fig_vol.update_layout(
        height=380,
        template="plotly_dark",
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(fig_vol, use_container_width=True)

st.divider()

# News & Gemini Sentiment Section
st.subheader("📰 Real News & Gemini AI Sentiment")
n_path = news_file(resolved_ticker)
s_path = sentiment_file(resolved_ticker)

if not s_path.exists() and not n_path.exists():
    st.info("No local news collected yet for this stock. Click **Refresh Data & News** in the sidebar to collect and analyze headlines.")
else:
    if s_path.exists():
        sent_df = pd.read_csv(s_path)
        ok_df = sent_df[sent_df.Status.eq("ok")]
        fail_df = sent_df[sent_df.Status.eq("failed")]
        
        st.caption(f"**Gemini Model:** `{gem_status['model']}` | **Valid Sentiment Headlines:** {len(ok_df)} | **Failed/Pending Headlines:** {len(fail_df)}")
        if not ok_df.empty:
            st.dataframe(
                ok_df[["Date", "Title", "Source", "Sentiment", "Score", "Link"]].head(25),
                use_container_width=True,
                column_config={
                    "Link": st.column_config.LinkColumn("Article Link"),
                    "Score": st.column_config.NumberColumn("Score", format="%.2f"),
                }
            )
        elif not fail_df.empty:
            st.warning("All recent news attempts failed. Check Gemini API key in `.env` and click Refresh News.")
            st.dataframe(fail_df[["Date", "Title", "Error"]].head(10), use_container_width=True)

st.divider()

# XGBoost AI Predictions Section
st.subheader(f"🤖 Machine Learning Return Predictions ({result['feature_set']})")
p_cols = st.columns(3)
for col, (horizon, pred_data) in zip(p_cols, result["predictions"].items()):
    est_return = pred_data["return"]
    est_price = pred_data["price"]
    mae = pred_data["metrics"]["MAE"]
    base_mae = pred_data["metrics"]["Baseline MAE"]
    acc = pred_data["metrics"]["Directional Accuracy"]
    
    col.metric(
        f"{horizon} Predicted Return",
        f"{est_return:+.2%}",
        f"Est Price: {est_price:,.2f}"
    )
    col.caption(f"**Model MAE:** {mae:.2%} | **Baseline MAE:** {base_mae:.2%}\n\n**Directional Accuracy:** {acc:.1%}")

st.caption("⚠️ Predictions are statistical estimates based on historical XGBoost regression. They are not guaranteed or investment advice.")

st.divider()

# Model Comparison & Backtesting
col_comp, col_back = st.columns(2)

with col_comp:
    st.subheader("Model Feature Comparison")
    st.caption("Technical Only vs Technical + News Sentiment")
    if "comparison" in result and isinstance(result["comparison"], pd.DataFrame):
        st.dataframe(result["comparison"], use_container_width=True)

with col_back:
    st.subheader("Walk-Forward Cross-Validation")
    st.caption("Chronological out-of-sample backtest")
    if "backtest" in result and isinstance(result["backtest"], pd.DataFrame):
        st.dataframe(result["backtest"], use_container_width=True)

# Multi-stock Comparison if specified
if compare_list:
    st.divider()
    st.subheader("🌐 Multi-Stock Research Comparison")
    comp_rows = []
    all_targets = list(dict.fromkeys([resolved_ticker] + compare_list))
    for t_query in all_targets:
        t_resolved, t_ok, _ = resolve_ticker(t_query)
        if not t_ok:
            comp_rows.append({"Ticker": t_query, "Status": "Invalid Ticker"})
            continue
        try:
            t_res = result if t_resolved == resolved_ticker else cached_analysis(t_resolved, custom_start, custom_end, range_name.lower(), include_news)
            t_raw = t_res["raw"]
            t_latest = t_raw.iloc[-1]
            t_pred = t_res["predictions"]["1-Day"]
            comp_rows.append({
                "Ticker": t_resolved,
                "Company": get_company_name(t_resolved),
                "Latest Close": float(t_latest.Close),
                "1D Predicted Return": t_pred["return"],
                "1D Predicted Price": t_pred["price"],
                "1D Model MAE": t_pred["metrics"]["MAE"],
                "1D Directional Accuracy": t_pred["metrics"]["Directional Accuracy"],
                "Status": "OK"
            })
        except Exception as exc:
            comp_rows.append({"Ticker": t_resolved, "Status": f"Error: {exc}"})
            
    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)
