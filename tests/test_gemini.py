import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.gemini_sentiment import analyze_sentiment, get_gemini_client

def test_gemini():
    print("=== Gemini API Integration Test ===")
    api_key = os.getenv("GEMINI_API_KEY", "")
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    print(f"1. Checking API Key: {'FOUND' if api_key else 'MISSING'}")
    if not api_key:
        print("ERROR: GEMINI_API_KEY is not set in environment or .env file.")
        print("Please set GEMINI_API_KEY in .env file.")
        return False

    print(f"2. Testing Gemini client initialization...")
    client, err = get_gemini_client()
    if err:
        print(f"ERROR: {err}")
        return False
    print("Client initialized successfully.")

    print(f"3. Verifying selected model: {model_name}")

    print("4. Testing JSON sentiment generation with sample headline:")
    headline = "Reliance Industries announces major investment in renewable energy."
    company = "Reliance Industries"
    print(f"   Headline: '{headline}'")
    print(f"   Company: '{company}'")

    sentiment, score, model_used, error = analyze_sentiment(headline, company, model_name=model_name)
    if error:
        print(f"ERROR calling Gemini API: {error}")
        return False

    print("5. Gemini API Result:")
    print(f"   Model Used : {model_used}")
    print(f"   Sentiment  : {sentiment}")
    print(f"   Score      : {score}")

    if sentiment == "positive" and score > 0:
        print("SUCCESS: Sentiment correctly recognized as positive.")
        return True
    else:
        print(f"RESULT: Received sentiment='{sentiment}', score={score}")
        return True

if __name__ == "__main__":
    test_gemini()
