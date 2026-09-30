import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
NEWS_DIR = DATA_DIR / "news"
MODEL_DIR = DATA_DIR / "models"
DEFAULT_TICKER = "RELIANCE.NS"

def get_secret(name: str, default: str = "") -> str:
    val = os.getenv(name, "").strip()
    if not val:
        try:
            import streamlit as st
            if hasattr(st, "secrets") and name in st.secrets:
                val = str(st.secrets[name]).strip()
        except Exception:
            pass
    return val or default

GEMINI_API_KEY = get_secret("GEMINI_API_KEY", "")
GEMINI_MODEL = get_secret("GEMINI_MODEL", "gemini-3.5-flash")

MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MINUTE = 30
XGB_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "objective": "reg:squarederror",
    "n_jobs": 2,
}
