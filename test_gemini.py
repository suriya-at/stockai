"""Run `venv\\Scripts\\python.exe test_gemini.py` before starting Streamlit."""
from dotenv import load_dotenv

load_dotenv()

from src.gemini_sentiment import check_gemini_connection, get_key_status


def main() -> int:
    keys = get_key_status()
    print(f"GEMINI_API_KEY: {'configured' if keys['gemini_api_key_configured'] else 'not configured'}")
    print(f"GOOGLE_API_KEY: {'configured' if keys['google_api_key_configured'] else 'not configured'}")
    if keys["key_conflict"]:
        print("Key conflict: both variables are set; this project explicitly uses GEMINI_API_KEY.")
    status = check_gemini_connection()
    print(f"Model: {status['model']}")
    if status["available_models"]:
        print("Available models:")
        for model in status["available_models"]:
            print(f"- {model}")
    if status["connected"]:
        print("Gemini API: Connected\nModel: Available\nResponse: OK")
        return 0
    print("Gemini API: Failed")
    print(f"API key: {'Detected' if status['api_key_configured'] else 'Not detected'}")
    print(f"Model: {'Available' if status['model_available'] else 'Unknown'}")
    print(f"Error type: {status['error_type']}\nError: {status['error_message']}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
