import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.market_data import resolve_ticker, get_company_name, download_history, MarketDataError
from src.gemini_sentiment import analyze_sentiment, get_gemini_status

def test_ticker_resolution():
    print("\n--- Test 1: RELIANCE.NS ---")
    sym, valid, msg = resolve_ticker("RELIANCE.NS")
    assert valid, f"Expected valid for RELIANCE.NS, got {msg}"
    assert sym == "RELIANCE.NS", f"Expected RELIANCE.NS, got {sym}"
    print(f"PASSED: {msg}")

    print("\n--- Test 2: TCS ---")
    sym, valid, msg = resolve_ticker("TCS")
    assert valid, f"Expected valid for TCS, got {msg}"
    assert sym == "TCS.NS", f"Expected TCS.NS, got {sym}"
    print(f"PASSED: {msg}")

    print("\n--- Test 3: AAPL ---")
    sym, valid, msg = resolve_ticker("AAPL")
    assert valid, f"Expected valid for AAPL, got {msg}"
    assert sym == "AAPL", f"Expected AAPL, got {sym}"
    print(f"PASSED: {msg}")

    print("\n--- Test 4: Apple ---")
    sym, valid, msg = resolve_ticker("Apple")
    assert valid, f"Expected valid for Apple, got {msg}"
    assert sym == "AAPL", f"Expected AAPL, got {sym}"
    print(f"PASSED: {msg}")

    print("\n--- Test 5: Invalid Ticker XYZABC ---")
    sym, valid, msg = resolve_ticker("XYZABC")
    assert not valid, "Expected invalid for XYZABC"
    assert "Could not find a valid Yahoo Finance ticker" in msg
    print(f"PASSED: Handled invalid ticker with error: {msg}")

def test_market_download():
    print("\n--- Market Data Download Test ---")
    df = download_history("TCS.NS", period="1y")
    assert not df.empty, "DataFrame should not be empty"
    assert len(df) > 100, f"Expected >100 rows, got {len(df)}"
    print(f"PASSED: Downloaded {len(df)} rows for TCS.NS.")

def test_gemini_sentiment_error_handling():
    print("\n--- Test 7 & 8: Missing/Invalid Gemini API Key Error Handling ---")
    # Backup key if set
    orig_key = os.environ.get("GEMINI_API_KEY")
    os.environ["GEMINI_API_KEY"] = ""
    
    label, score, model, err = analyze_sentiment("Reliance announces renewable energy investment", "Reliance")
    assert label is None, "Should not return neutral label on error"
    assert score is None, "Should not return 0.0 score on error"
    assert "GEMINI_API_KEY" in err, f"Expected key error, got '{err}'"
    print(f"PASSED: Missing API key returned error without fake neutral: '{err}'")

    os.environ["GEMINI_API_KEY"] = "invalid_key_123"
    label, score, model, err = analyze_sentiment("Reliance announces renewable energy investment", "Reliance", retries=0)
    assert label is None, "Should not return neutral on invalid key"
    assert score is None, "Should not return 0.0 on invalid key"
    assert err != "", "Error message must be populated"
    print(f"PASSED: Invalid API key returned error: '{err}'")

    if orig_key:
        os.environ["GEMINI_API_KEY"] = orig_key

if __name__ == "__main__":
    print("=== Running Complete Stock AI Test Suite ===")
    test_ticker_resolution()
    test_market_download()
    test_gemini_sentiment_error_handling()
    print("\nALL BACKEND TESTS PASSED SUCCESSFULLY!")
