"""Multi-backend LLM client: Local Ollama, xAI (Grok), or Groq Cloud."""
import os
import requests

URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL = os.getenv("MODEL", "llama3.2:3b")


def get_secret(key_name: str) -> str | None:
    """Retrieve key from environment or Streamlit secrets."""
    val = os.getenv(key_name)
    if val:
        return val.strip()
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key_name in st.secrets:
            return str(st.secrets[key_name]).strip()
    except Exception:
        pass
    return None


def list_available_models() -> list[str]:
    """Return all models currently pulled in Ollama."""
    try:
        r = requests.get(f"{URL}/api/tags", timeout=2)
        if r.status_code == 200:
            return [m.get("name", "") for m in r.json().get("models", [])]
    except Exception:
        pass
    return []


def get_active_model() -> str:
    """Return configured model or auto-detect first available model."""
    if "MODEL" in os.environ:
        return os.environ["MODEL"]
    available = list_available_models()
    if available:
        for preferred in ["llama3.2:3b", "llama3.2", "llama3.1:8b", "llama3.1"]:
            for m in available:
                if m.lower().startswith(preferred.lower()):
                    return m
        return available[0]
    return MODEL


def _call_xai(api_key: str, prompt: str, json_mode: bool) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": "grok-2-latest",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "stream": False,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    r = requests.post("https://api.x.ai/v1/chat/completions", headers=headers, json=body, timeout=60)
    if r.status_code == 403:
        err = r.json().get("error", "Permission denied / No active credits")
        raise RuntimeError(f"xAI API Error: {err}")
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def _call_groq(api_key: str, prompt: str, json_mode: bool) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    for model_name in ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]:
        body = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "stream": False,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        r = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=body, timeout=60)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]



def _call_ollama(prompt: str, json_mode: bool) -> str:
    active_model = get_active_model()
    body = {"model": active_model, "prompt": prompt, "stream": False, "options": {"temperature": 0}}
    if json_mode:
        body["format"] = "json"
    r = requests.post(f"{URL}/api/generate", json=body, timeout=120)
    r.raise_for_status()
    return r.json()["response"]


def check_status() -> tuple[bool, str]:
    """Check if Ollama server is running, or if cloud API keys are ready."""
    # 1. Check Local Ollama
    try:
        r = requests.get(f"{URL}/api/tags", timeout=2)
        if r.status_code == 200:
            models = [m.get("name", "") for m in r.json().get("models", [])]
            if models:
                active = get_active_model()
                return True, f"Connected to Ollama ({active})"
            return False, "Ollama running, but no models found. Run: ollama pull llama3.2:3b"
    except Exception:
        pass

    # 2. Check xAI Cloud
    if get_secret("XAI_API_KEY"):
        return True, "Connected to xAI Cloud (Grok)"

    # 3. Check Groq Cloud
    if get_secret("GROQ_API_KEY"):
        return True, "Connected to Groq Cloud (Llama 3.2)"

    return False, f"Ollama is offline and no cloud API key detected."


def chat(prompt: str, json_mode: bool = False) -> str:
    """Route chat request to Ollama first, falling back to xAI or Groq."""
    # Attempt local Ollama
    try:
        r = requests.get(f"{URL}/api/tags", timeout=1)
        if r.status_code == 200:
            return _call_ollama(prompt, json_mode)
    except Exception:
        pass

    # Fallback: xAI API
    xai_key = get_secret("XAI_API_KEY")
    if xai_key:
        try:
            return _call_xai(xai_key, prompt, json_mode)
        except Exception as e:
            if not get_secret("GROQ_API_KEY"):
                raise

    # Fallback: Groq Cloud API
    groq_key = get_secret("GROQ_API_KEY")
    if groq_key:
        return _call_groq(groq_key, prompt, json_mode)

    raise ConnectionError(
        f"Could not connect to Ollama at {URL} and no valid cloud API key was configured. "
        "Please ensure Ollama is running (`ollama serve`) or set XAI_API_KEY / GROQ_API_KEY."
    )
