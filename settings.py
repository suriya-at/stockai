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
    value = os.getenv(name, "").strip()
    if not value:
        try:
            import streamlit as st
            if name in st.secrets:
                value = str(st.secrets[name]).strip()
        except Exception:
            pass
    return value or default


GEMINI_MODEL = get_secret("GEMINI_MODEL", "gemini-2.5-flash")
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
