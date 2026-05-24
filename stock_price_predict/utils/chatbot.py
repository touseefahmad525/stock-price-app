import os

from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "gemini-2.5-flash"


def _get_api_key():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add it to your .env file, then restart Streamlit."
        )

    return api_key


def _generate_with_google_genai(prompt, api_key):
    try:
        from google import genai
    except ImportError as exc:
        return None, exc

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )
    return response.text, None


def _generate_with_google_generativeai(prompt, api_key):
    try:
        import google.generativeai as genai
    except ImportError as exc:
        return None, exc

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(prompt)
    return response.text, None


def get_ai_response(user_message):
    prompt = f"""
    You are a Stock Market AI Assistant.

    You are NOT a financial advisor.

    Your job is to ANALYZE stocks and give a decision score.

    Always respond in this format:

    1. Summary (2-3 lines)
    2. Pros
    3. Cons
    4. Risk Level (Low / Medium / High)
    5. Final Score (0-100)
    6. Recommendation:
    - Buy
    - Hold
    - Avoid

    Be logical and data-driven.

    User Question:
    {user_message}
    """

    api_key = _get_api_key()
    text, new_sdk_error = _generate_with_google_genai(prompt, api_key)
    if text:
        return text

    text, old_sdk_error = _generate_with_google_generativeai(prompt, api_key)
    if text:
        return text

    raise RuntimeError(
        "Gemini SDK is not installed in the Python environment running this app. "
        "Run Streamlit with the project venv, or install dependencies with: "
        "python -m pip install -r requirements.txt"
    ) from old_sdk_error or new_sdk_error
