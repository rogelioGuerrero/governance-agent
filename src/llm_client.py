"""
Cliente LLM multi-provider con failover automatico.

Orden de providers:
1. Groq (mas rapido) — gpt-oss-120b / gpt-oss-20b
2. Gemini 2.0 Flash (alto free tier) — via OpenAI-compatible endpoint
3. SambaNova (alto free tier) — via OpenAI-compatible endpoint

El agente no sabe ni le importa cual LLM responde.
Solo llama call_groq() o call_groq_with_tools() y recibe la respuesta.

Los nombres de funciones se mantienen (call_groq, call_groq_with_tools) para
compatibilidad con los 4 archivos que las importan.
"""

import os
import time
import threading
import httpx
from dotenv import load_dotenv

from .log_config import get_logger

log = get_logger("llm")

load_dotenv()

# === RATE LIMITER GLOBAL ===
# Evita saturar los providers con llamadas concurrentes (MoA dispara 3 threads).
# Configurable via env: LLM_MIN_DELAY (segundos entre llamadas), LLM_MAX_CONCURRENT

_LLM_MIN_DELAY = float(os.getenv("LLM_MIN_DELAY", "2.0"))
_LLM_MAX_CONCURRENT = int(os.getenv("LLM_MAX_CONCURRENT", "1"))
_LLM_SEMAPHORE = threading.Semaphore(_LLM_MAX_CONCURRENT)
_LLM_LAST_CALL_TIME = 0.0
_LLM_RATE_LOCK = threading.Lock()


def _acquire_llm_slot():
    """Adquirir un slot para llamar al LLM. Bloquea si hay llamadas en curso.
    
    Enforcea un delay mínimo entre llamadas consecutivas para no saturar el rate limit.
    Thread-safe: usa semaphore + lock para coordinar threads del MoA.
    """
    _LLM_SEMAPHORE.acquire()
    global _LLM_LAST_CALL_TIME
    with _LLM_RATE_LOCK:
        elapsed = time.monotonic() - _LLM_LAST_CALL_TIME
        wait = _LLM_MIN_DELAY - elapsed
        if wait > 0:
            time.sleep(wait)
        _LLM_LAST_CALL_TIME = time.monotonic()


def _release_llm_slot():
    """Liberar el slot del LLM."""
    _LLM_SEMAPHORE.release()

# === PROVIDERS ===

PROVIDERS = []

# 1. Groq
_groq_key = os.getenv("GROQ_API_KEY", "")
if _groq_key:
    PROVIDERS.append({
        "name": "groq",
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "key": _groq_key,
        "models": [
            os.getenv("GROQ_MODEL_PRIMARY", "llama-3.3-70b-versatile"),
            os.getenv("GROQ_MODEL_FALLBACK", "llama-3.1-8b-instant"),
        ],
        "max_retries": 2,
        "backoff_base": 2,
    })

# 2. Gemini (via OpenAI-compatible endpoint)
_gemini_key = os.getenv("GEMINI_API_KEY", "")
if _gemini_key:
    PROVIDERS.append({
        "name": "gemini",
        "url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "key": _gemini_key,
        "models": [os.getenv("GEMINI_MODEL", "gemini-2.0-flash")],
        "max_retries": 2,
        "backoff_base": 2,
    })

# 3. SambaNova
_sambanova_key = os.getenv("SAMBANOVA_API_KEY", "")
if _sambanova_key:
    PROVIDERS.append({
        "name": "sambanova",
        "url": "https://api.sambanova.ai/v1/chat/completions",
        "key": _sambanova_key,
        "models": [os.getenv("SAMBANOVA_MODEL", "Meta-Llama-3.1-70B-Instruct")],
        "max_retries": 2,
        "backoff_base": 2,
    })

if not PROVIDERS:
    log.warning("No hay providers LLM configurados. Verifica .env (GROQ_API_KEY, GEMINI_API_KEY, SAMBANOVA_API_KEY)")


def _try_provider(provider: dict, messages: list[dict], tools: list[dict] | None = None,
                  temperature: float = 0.3, max_tokens: int = 4000, timeout: int = 45,
                  json_mode: bool = False) -> dict | str | None:
    """Intenta llamar un provider. Retorna resultado o None si falla.
    
    Rate-limited: respeta delay mínimo entre llamadas y concurrencia máxima.
    """
    headers = {
        "Authorization": f"Bearer {provider['key']}",
        "Content-Type": "application/json",
    }

    for mdl in provider["models"]:
        for retry in range(provider["max_retries"] + 1):
            _acquire_llm_slot()
            try:
                body = {
                    "model": mdl,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if tools:
                    body["tools"] = tools
                    body["tool_choice"] = "auto"
                if json_mode:
                    body["response_format"] = {"type": "json_object"}

                with httpx.Client(timeout=timeout) as client:
                    resp = client.post(provider["url"], headers=headers, json=body)

                if resp.status_code == 429:
                    if retry < provider["max_retries"]:
                        wait = provider["backoff_base"] ** (retry + 1)
                        log.warning("[%s] 429 rate limit (%s), esperando %ss...", provider["name"], mdl, wait)
                        time.sleep(wait)
                        continue
                    else:
                        log.warning("[%s] 429 agotado para %s, saltando al siguiente provider", provider["name"], mdl)
                        break

                if resp.status_code >= 500:
                    if retry < provider["max_retries"]:
                        wait = provider["backoff_base"] ** (retry + 1)
                        log.warning("[%s] %s (%s), reintentando en %ss...", provider["name"], resp.status_code, mdl, wait)
                        time.sleep(wait)
                        continue
                    else:
                        break

                resp.raise_for_status()
                data = resp.json()
                message = data["choices"][0]["message"]

                if tools:
                    if not message.get("content") and not message.get("tool_calls"):
                        log.warning("[%s] Respuesta vacia de %s (sin content ni tool_calls)", provider["name"], mdl)
                        continue
                    log.info("[%s] OK con %s (tool calling)", provider["name"], mdl)
                    return {
                        "role": message.get("role", "assistant"),
                        "content": message.get("content") or "",
                        "tool_calls": message.get("tool_calls"),
                    }
                else:
                    content = message.get("content", "")
                    if not content:
                        log.warning("[%s] Respuesta vacia de %s", provider["name"], mdl)
                        continue
                    log.info("[%s] OK con %s", provider["name"], mdl)
                    return content

            except (httpx.HTTPError, KeyError, IndexError) as e:
                if retry < provider["max_retries"]:
                    wait = provider["backoff_base"] ** (retry + 1)
                    log.warning("[%s] Error: %s (%s), reintentando en %ss...", provider["name"], e, mdl, wait)
                    time.sleep(wait)
                else:
                    log.error("[%s] %s fallo tras %d intentos: %s", provider["name"], mdl, provider["max_retries"] + 1, e)
            finally:
                _release_llm_slot()

    return None


def call_groq(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 4000,
    json_mode: bool = False,
    timeout: int = 30,
    session_max_retries: int = 2,
    session_backoff: int = 30,
) -> str:
    """
    Llamar al LLM con retry y failover multi-provider.

    Mantiene compatibilidad con call_groq original.
    Intenta Groq primero, luego Gemini, luego SambaNova.
    Si todos fallan, espera session_backoff segundos y reintenta toda la ronda.
    """
    last_error = None
    for session_attempt in range(session_max_retries + 1):
        for provider in PROVIDERS:
            if model and model not in provider["models"]:
                continue

            result = _try_provider(
                provider, messages, tools=None,
                temperature=temperature, max_tokens=max_tokens,
                timeout=timeout, json_mode=json_mode,
            )
            if result is not None:
                return result

        last_error = "todos los providers fallaron"
        if session_attempt < session_max_retries:
            log.warning("Todos los providers fallaron (intento %d/%d), esperando %ss antes de reintentar...",
                        session_attempt + 1, session_max_retries + 1, session_backoff)
            time.sleep(session_backoff)
        else:
            log.error("Sesion agotada tras %d intentos. %s", session_max_retries + 1, last_error)

    raise RuntimeError(f"LLM: {last_error}")


def call_groq_with_tools(
    messages: list[dict],
    tools: list[dict] | None = None,
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 4000,
    timeout: int = 45,
    session_max_retries: int = 2,
    session_backoff: int = 30,
) -> dict:
    """
    Llamar al LLM con tool calling y failover multi-provider.

    Mantiene compatibilidad con call_groq_with_tools original.
    Intenta Groq primero, luego Gemini, luego SambaNova.
    Si todos fallan, espera session_backoff segundos y reintenta toda la ronda.
    """
    last_error = None
    for session_attempt in range(session_max_retries + 1):
        for provider in PROVIDERS:
            if model and model not in provider["models"]:
                continue

            result = _try_provider(
                provider, messages, tools=tools,
                temperature=temperature, max_tokens=max_tokens,
                timeout=timeout, json_mode=False,
            )
            if result is not None:
                return result

        last_error = "todos los providers fallaron (tool calling)"
        if session_attempt < session_max_retries:
            log.warning("Todos los providers fallaron (intento %d/%d), esperando %ss antes de reintentar...",
                        session_attempt + 1, session_max_retries + 1, session_backoff)
            time.sleep(session_backoff)
        else:
            log.error("Sesion agotada tras %d intentos. %s", session_max_retries + 1, last_error)

    raise RuntimeError(f"LLM: {last_error}")
