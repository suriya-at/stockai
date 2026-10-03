"""Gemini configuration, diagnostics, and structured headline sentiment."""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


def _secret(name: str) -> str:
    """Return a secret without logging its value."""
    value = os.getenv(name, "").strip()
    if value:
        return value
    try:
        import streamlit as st
        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:
        pass
    return ""


def get_api_key() -> str:
    return _secret("GEMINI_API_KEY")


def get_gemini_model_name() -> str:
    return _secret("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL


def get_key_status() -> dict[str, Any]:
    gemini_key, google_key = get_api_key(), _secret("GOOGLE_API_KEY")
    return {
        "gemini_api_key_configured": bool(gemini_key),
        "google_api_key_configured": bool(google_key),
        "key_conflict": bool(gemini_key and google_key),
        # Explicit Client(api_key=...) prevents a stale GOOGLE_API_KEY override.
        "selected_key": "GEMINI_API_KEY" if gemini_key else None,
    }


def classify_gemini_error(error: Exception | str) -> tuple[str, str]:
    """Return a stable category plus an actionable, key-safe message."""
    message = str(error).strip()
    lower = message.lower()
    match = re.search(r"\b(401|403|404|408|429|500|502|503|504)\b", message)
    status = match.group(1) if match else ""
    if status == "401" or "unauthenticated" in lower or "api key not valid" in lower or "invalid api key" in lower:
        return "authentication", "Authentication failed. The Gemini API key may be invalid, revoked, or expired. " + message
    if status == "403" or "permission_denied" in lower or "permission denied" in lower:
        if "restrict" in lower or "api target" in lower or "referer" in lower:
            return "key_restriction", "The API key restriction rejected this request. In Google AI Studio, update the key to allow the Gemini API. " + message
        return "permission", "The key is recognized but lacks Gemini API access. Check its Google AI Studio project, API access, and restrictions. " + message
    if status == "404" or "not_found" in lower or "not found" in lower or "unsupported model" in lower:
        return "model_unavailable", "The configured Gemini model is not available to this API key. Set GEMINI_MODEL to a model returned by the diagnostic. " + message
    if status == "429" or "resource_exhausted" in lower or "quota" in lower or "rate limit" in lower:
        return "quota_or_rate_limit", "Gemini rejected the request because of a rate limit, quota, or billing limit. Check usage and billing in Google AI Studio. " + message
    if any(token in lower for token in ("timeout", "connection", "dns", "network", "connecterror", "readerror")):
        return "network", "Could not reach the Gemini API. Check internet access, proxy settings, and try again. " + message
    if isinstance(error, (ImportError, ModuleNotFoundError)) or "google-genai" in lower:
        return "sdk", "The Google GenAI SDK could not be imported or initialized. Run `pip install -U google-genai`. " + message
    if status in {"500", "502", "503", "504"} or "unavailable" in lower:
        return "service", "The Gemini service is temporarily unavailable. Retry shortly. " + message
    return "request", message or error.__class__.__name__


def get_gemini_client():
    """Create the current google-genai client using GEMINI_API_KEY explicitly."""
    api_key = get_api_key()
    if not api_key:
        return None, "GEMINI_API_KEY not found. Add it to .env locally or Streamlit Secrets in deployment."
    try:
        from google import genai
        return genai.Client(api_key=api_key), ""
    except Exception as exc:
        return None, classify_gemini_error(exc)[1]


def _model_names(client: Any) -> list[str]:
    return [model.name for model in client.models.list() if getattr(model, "name", None)]


def check_gemini_connection() -> dict[str, Any]:
    """Make a live list-models + generation health check; client creation is not success."""
    model_name, keys = get_gemini_model_name(), get_key_status()
    result: dict[str, Any] = {
        "connected": False, "api_key_configured": keys["gemini_api_key_configured"],
        "google_api_key_configured": keys["google_api_key_configured"], "key_conflict": keys["key_conflict"],
        "model": model_name, "model_available": None, "available_models": [],
        "error_type": "", "error_message": "", "error": "",
    }
    if not keys["gemini_api_key_configured"]:
        result.update(error_type="missing_key", error_message="GEMINI_API_KEY not found.", error="GEMINI_API_KEY not found.")
        return result
    client, client_error = get_gemini_client()
    if client is None:
        result.update(error_type="sdk", error_message=client_error, error=client_error)
        return result
    try:
        result["available_models"] = _model_names(client)
        requested = model_name.removeprefix("models/")
        result["model_available"] = any(name.removeprefix("models/") == requested for name in result["available_models"])
        if not result["model_available"]:
            message = f"Configured model '{model_name}' was not returned for this API key."
            result.update(error_type="model_unavailable", error_message=message, error=message)
            return result
        client.models.generate_content(model=model_name, contents="Reply with OK.")
        result["connected"] = True
        return result
    except Exception as exc:
        kind, message = classify_gemini_error(exc)
        result.update(error_type=kind, error_message=message, error=message)
        return result


def get_gemini_status() -> dict[str, Any]:
    return check_gemini_connection()


def analyze_sentiment(title: str, company_name: str, model_name: str | None = None, retries: int = 1) -> tuple[str | None, float | None, str, str]:
    """Analyze a headline in Gemini JSON mode; errors are never neutral scores."""
    selected_model = model_name or get_gemini_model_name()
    client, client_error = get_gemini_client()
    if client is None:
        return None, None, selected_model, client_error
    try:
        from google.genai import types
    except Exception as exc:
        return None, None, selected_model, classify_gemini_error(exc)[1]
    prompt = f'''Analyze the financial sentiment of this news headline toward the specified company.

Company:
{company_name}

Headline:
{title}

Return only valid JSON:

{{
    "sentiment": "positive",
    "score": 0.5
}}

Rules:
- sentiment must be positive, negative, or neutral
- score must be between -1 and 1
- positive sentiment should have a positive score
- negative sentiment should have a negative score
- neutral sentiment should have a score near 0
- return JSON only'''
    schema = {"type": "object", "properties": {"sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]}, "score": {"type": "number", "minimum": -1, "maximum": 1}}, "required": ["sentiment", "score"]}
    last_error = "Gemini request did not return a result."
    for _ in range(retries + 1):
        try:
            response = client.models.generate_content(model=selected_model, contents=prompt, config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=schema, temperature=0.0))
            parsed = json.loads((response.text or "").strip())
            label, score = str(parsed["sentiment"]).lower(), float(parsed["score"])
            if label not in {"positive", "negative", "neutral"} or not -1.0 <= score <= 1.0:
                raise ValueError("Gemini returned invalid sentiment JSON values.")
            return label, score, selected_model, ""
        except Exception as exc:
            kind, last_error = classify_gemini_error(exc)
            log.warning("Gemini sentiment request failed (%s): %s", kind, last_error)
            if kind in {"authentication", "permission", "key_restriction", "model_unavailable", "quota_or_rate_limit"}:
                break
    return None, None, selected_model, last_error
