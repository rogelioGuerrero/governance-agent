"""
Cliente Groq para el agente de governance.

Usa la API key configurada en .env.
Modelos: gpt-oss-120b (primario) + gpt-oss-20b (fallback).
"""

import os
import time
import httpx
from dotenv import load_dotenv

from .log_config import get_logger

log = get_logger("groq")

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
PRIMARY_MODEL = os.getenv("GROQ_MODEL_PRIMARY", "openai/gpt-oss-120b")
FALLBACK_MODEL = os.getenv("GROQ_MODEL_FALLBACK", "openai/gpt-oss-20b")


def call_groq(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 4000,
    json_mode: bool = False,
    timeout: int = 30,
) -> str:
    """
    Llamar a Groq con retry y fallback automático.

    Args:
        messages: lista de mensajes en formato OpenAI
        model: modelo a usar (default: PRIMARY_MODEL)
        temperature: temperatura (default: 0.3)
        max_tokens: máximo de tokens (default: 4000, por reasoning tokens)
        json_mode: forzar response en JSON
        timeout: timeout en segundos

    Returns:
        contenido del mensaje de respuesta
    """
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY no configurada. Verifica .env o variables de entorno.")

    models_to_try = [model or PRIMARY_MODEL]
    if model is None:
        models_to_try.append(FALLBACK_MODEL)

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    for attempt, mdl in enumerate(models_to_try):
        for retry in range(3):
            try:
                body = {
                    "model": mdl,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if json_mode:
                    body["response_format"] = {"type": "json_object"}

                with httpx.Client(timeout=timeout) as client:
                    resp = client.post(GROQ_URL, headers=headers, json=body)

                if resp.status_code == 429:
                    wait = 2 ** (retry + 1)
                    log.warning("429 rate limit, esperando %ss...", wait)
                    time.sleep(wait)
                    continue

                if resp.status_code >= 500:
                    wait = 2 ** (retry + 1)
                    log.warning("%s, reintentando en %ss...", resp.status_code, wait)
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                if not content:
                    log.warning("Respuesta vacía de %s", mdl)
                    continue
                return content

            except (httpx.HTTPError, KeyError, IndexError) as e:
                if retry < 2:
                    wait = 2 ** (retry + 1)
                    log.warning("Error: %s, reintentando en %ss...", e, wait)
                    time.sleep(wait)
                else:
                    log.error("%s falló tras 3 intentos: %s", mdl, e)

    raise RuntimeError("Groq: todos los modelos fallaron")


def call_groq_with_tools(
    messages: list[dict],
    tools: list[dict] | None = None,
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 4000,
    timeout: int = 45,
) -> dict:
    """
    Llamar a Groq con soporte nativo de tool calling (function calling).

    A diferencia de call_groq(), retorna el mensaje completo del asistente
    incluyendo tool_calls (no solo content).

    Args:
        messages: lista de mensajes en formato OpenAI (incluyendo role=tool)
        tools: lista de tool schemas en formato OpenAI function calling
        model: modelo a usar (default: PRIMARY_MODEL)
        temperature: temperatura
        max_tokens: máximo de tokens
        timeout: timeout en segundos

    Returns:
        dict con: content (str), tool_calls (list|None), role (str)
        Ejemplo: {"role": "assistant", "content": "...", "tool_calls": [...]}
    """
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY no configurada. Verifica .env o variables de entorno.")

    models_to_try = [model or PRIMARY_MODEL]
    if model is None:
        models_to_try.append(FALLBACK_MODEL)

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    for attempt, mdl in enumerate(models_to_try):
        for retry in range(3):
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

                with httpx.Client(timeout=timeout) as client:
                    resp = client.post(GROQ_URL, headers=headers, json=body)

                if resp.status_code == 429:
                    wait = 2 ** (retry + 1)
                    log.warning("429 rate limit, esperando %ss...", wait)
                    time.sleep(wait)
                    continue

                if resp.status_code >= 500:
                    wait = 2 ** (retry + 1)
                    log.warning("%s, reintentando en %ss...", resp.status_code, wait)
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                data = resp.json()
                message = data["choices"][0]["message"]

                # message puede tener content=None cuando hay tool_calls
                if not message.get("content") and not message.get("tool_calls"):
                    log.warning("Respuesta vacía de %s (sin content ni tool_calls)", mdl)
                    continue

                return {
                    "role": message.get("role", "assistant"),
                    "content": message.get("content") or "",
                    "tool_calls": message.get("tool_calls"),
                }

            except (httpx.HTTPError, KeyError, IndexError) as e:
                if retry < 2:
                    wait = 2 ** (retry + 1)
                    log.warning("Error: %s, reintentando en %ss...", e, wait)
                    time.sleep(wait)
                else:
                    log.error("%s falló tras 3 intentos: %s", mdl, e)

    raise RuntimeError("Groq: todos los modelos fallaron (tool calling)")
