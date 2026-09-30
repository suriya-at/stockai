from __future__ import annotations
import json
import logging
import os
import time
from dotenv import load_dotenv

# Load environment variables from .env if available
load_dotenv()

log = logging.getLogger(__name__)

def get_gemini_model_name() -> str:
    return os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

def get_gemini_client():
    """Initialize and return google-genai Client if GEMINI_API_KEY is configured."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None, "GEMINI_API_KEY is not configured in environment or .env file."
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        return client, ""
    except Exception as exc:
        return None, f"Failed to initialize Gemini client: {exc}"

def get_gemini_status() -> dict:
    """Return status information about the Gemini API connection."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model_name = get_gemini_model_name()
    if not api_key:
        return {
            "connected": False,
            "api_key_configured": False,
            "model": model_name,
            "error": "GEMINI_API_KEY is missing. Set it in .env or environment.",
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
    retries: int = 2
) -> tuple[str | None, float | None, str, str]:
    """Analyze financial sentiment of a headline towards a company using Gemini API.
    
    Returns:
        (sentiment, score, model_used, error_message)
        sentiment: "positive", "negative", or "neutral"
        score: float between -1.0 and 1.0
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    target_model = model_name or get_gemini_model_name()
    
    if not api_key:
        return None, None, target_model, "GEMINI_API_KEY environment variable is missing"

    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=api_key)
    except Exception as exc:
        return None, None, target_model, f"Gemini SDK error: {exc}"

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

    last_error = "Unknown error"
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
            log.debug("Gemini raw response: %s", raw_text)

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
            log.warning("Gemini sentiment attempt %d failed: %s", attempt + 1, last_error)
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))

    return None, None, target_model, last_error
