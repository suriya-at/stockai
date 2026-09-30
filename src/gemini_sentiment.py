from __future__ import annotations
import json
import logging
import os
import time
from dotenv import load_dotenv

# Load environment variables from .env if available locally
load_dotenv()

log = logging.getLogger(__name__)

FALLBACK_MODELS = ["gemini-3.5-flash", "gemini-3.8-flash", "gemini-3.5-flash-lite"]

def get_api_key() -> str:
    """Retrieve GEMINI_API_KEY from environment variables or Streamlit Cloud Secrets."""
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        try:
            import streamlit as st
            if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
                key = str(st.secrets["GEMINI_API_KEY"]).strip()
        except Exception:
            pass
    return key

def get_gemini_model_name() -> str:
    """Retrieve GEMINI_MODEL from environment variables, Streamlit Cloud Secrets, or default."""
    model = os.getenv("GEMINI_MODEL", "").strip()
    if not model:
        try:
            import streamlit as st
            if hasattr(st, "secrets") and "GEMINI_MODEL" in st.secrets:
                model = str(st.secrets["GEMINI_MODEL"]).strip()
        except Exception:
            pass
    return model or "gemini-3.5-flash"

def get_gemini_client():
    """Initialize and return google-genai Client if GEMINI_API_KEY is configured."""
    api_key = get_api_key()
    if not api_key:
        return None, "GEMINI_API_KEY is missing. Set it in .env (local) or Streamlit Secrets (cloud)."
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        return client, ""
    except Exception as exc:
        return None, f"Failed to initialize Gemini client: {exc}"

def get_gemini_status() -> dict:
    """Return status information about the Gemini API connection."""
    api_key = get_api_key()
    model_name = get_gemini_model_name()
    if not api_key:
        return {
            "connected": False,
            "api_key_configured": False,
            "model": model_name,
            "error": "GEMINI_API_KEY is missing. Add it in Streamlit Cloud Secrets (or .env locally).",
        }
    client, err = get_gemini_client()
    if err or not client:
        return {
            "connected": False,
            "api_key_configured": True,
            "model": model_name,
            "error": err,
        }
    return {
        "connected": True,
        "api_key_configured": True,
        "model": model_name,
        "error": "",
    }

def analyze_sentiment(
    title: str,
    company_name: str,
    model_name: str | None = None,
    retries: int = 1
) -> tuple[str | None, float | None, str, str]:
    """Analyze financial sentiment of a headline towards a company using Gemini API.
    
    Returns:
        (sentiment, score, model_used, error_message)
        sentiment: "positive", "negative", or "neutral"
        score: float between -1.0 and 1.0
    """
    api_key = get_api_key()
    primary_model = model_name or get_gemini_model_name()
    
    if not api_key:
        return None, None, primary_model, "GEMINI_API_KEY is missing. Add it to Streamlit Secrets or .env."

    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=api_key)
    except Exception as exc:
        return None, None, primary_model, f"Gemini SDK error: {exc}"

    prompt = (
        f"Analyze the financial sentiment of this news article toward the specified company.\n\n"
        f"Company:\n{company_name}\n\n"
        f"Headline:\n{title}\n\n"
        f"Return JSON:\n"
        f"{{\n"
        f'    "sentiment": "positive",\n'
        f'    "score": 0.5\n'
        f"}}\n\n"
        f"Rules:\n"
        f"sentiment must be:\npositive\nnegative\nneutral\n\n"
        f"score must be between -1 and 1.\n\n"
        f"Return only JSON."
    )

    models_to_try = [primary_model] + [m for m in FALLBACK_MODELS if m != primary_model]
    last_error = "Unknown error"

    for target_model in models_to_try:
        for attempt in range(retries + 1):
            try:
                response = client.models.generate_content(
                    model=target_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.0,
                    ),
                )
                raw_text = (response.text or "").strip()
                log.debug("Gemini raw response from %s: %s", target_model, raw_text)

                if not raw_text:
                    raise ValueError("Received empty response from Gemini API.")

                cleaned = raw_text.replace("```json", "").replace("```", "").strip()
                parsed = json.loads(cleaned)

                label = str(parsed.get("sentiment", "")).lower()
                score = float(parsed.get("score", 0.0))

                if label not in {"positive", "negative", "neutral"}:
                    raise ValueError(f"Invalid sentiment label '{label}' returned from Gemini.")
                if not (-1.0 <= score <= 1.0):
                    raise ValueError(f"Score {score} out of valid range [-1, 1].")

                return label, score, target_model, ""
            except Exception as exc:
                last_error = str(exc)
                log.warning("Gemini sentiment attempt %d on %s failed: %s", attempt + 1, target_model, last_error)
                # If model not found or unavailable, break retry loop to try next fallback model immediately
                if "404" in last_error or "503" in last_error or "NOT_FOUND" in last_error or "UNAVAILABLE" in last_error:
                    break
                if attempt < retries:
                    time.sleep(1)

    return None, None, primary_model, last_error
