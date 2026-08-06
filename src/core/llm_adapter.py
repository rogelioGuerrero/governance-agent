"""
Adapter LLM para el ValidationEngine.

Conecta el llm_client.py existente (multi-provider con failover)
con la interfaz que espera el ValidationEngine (Capa 2 semántica).

El validator espera:
    llm_client.call(messages=[...], temperature=0.1) -> Result(ok=True, content="...")

Este adapter traduce call_groq() → Result con manejo de errores.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMResult:
    """Resultado estandarizado de una llamada LLM."""
    ok: bool
    content: Optional[str] = None
    error: Optional[str] = None
    provider: str = ""


class LLMAdapter:
    """
    Adapter que envuelve llm_client.call_groq() para el ValidationEngine.

    Uso:
        from src.core.llm_adapter import LLMAdapter
        llm = LLMAdapter()
        engine = ValidationEngine(pack=pack, llm_client=llm)
    """

    def __init__(self, json_mode: bool = True, temperature: float = 0.1,
                 max_tokens: int = 2000, timeout: int = 30):
        """
        Args:
            json_mode: forzar respuesta en JSON (recomendado para validación)
            temperature: baja temperatura para respuestas deterministas
            max_tokens: límite de tokens de respuesta
            timeout: timeout por llamada en segundos
        """
        self.json_mode = json_mode
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def call(self, messages: list[dict], temperature: float = None,
             max_tokens: int = None) -> LLMResult:
        """
        Llamar al LLM con los messages dados.

        Args:
            messages: lista de {"role": "...", "content": "..."}
            temperature: override de temperatura (opcional)
            max_tokens: override de max_tokens (opcional)

        Returns:
            LLMResult con ok=True y content=respuesta, o ok=False y error
        """
        from src.llm_client import call_groq

        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        try:
            content = call_groq(
                messages=messages,
                temperature=temp,
                max_tokens=tokens,
                json_mode=self.json_mode,
                timeout=self.timeout,
            )
            if content:
                return LLMResult(ok=True, content=content, provider="multi")
            else:
                return LLMResult(ok=False, error="Respuesta vacía del LLM")
        except RuntimeError as e:
            logger.error("LLM falló: %s", e)
            return LLMResult(ok=False, error=str(e))
        except Exception as e:
            logger.error("LLM error inesperado: %s", e)
            return LLMResult(ok=False, error=str(e))

    def call_with_tools(self, messages: list[dict], tools: list[dict],
                        temperature: float = None, max_tokens: int = None) -> LLMResult:
        """
        Llamar al LLM con tool calling.

        Returns:
            LLMResult con content=JSON del tool_call response
        """
        from src.llm_client import call_groq_with_tools

        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        try:
            result = call_groq_with_tools(
                messages=messages,
                tools=tools,
                temperature=temp,
                max_tokens=tokens,
                timeout=self.timeout,
            )
            if result:
                # result es un dict con role, content, tool_calls
                content = result.get("content", "")
                tool_calls = result.get("tool_calls")
                if tool_calls:
                    # Serializar tool_calls para que el caller los procese
                    return LLMResult(
                        ok=True,
                        content=json.dumps({"content": content, "tool_calls": tool_calls}),
                        provider="multi",
                    )
                elif content:
                    return LLMResult(ok=True, content=content, provider="multi")
                else:
                    return LLMResult(ok=False, error="LLM respondió sin content ni tool_calls")
            else:
                return LLMResult(ok=False, error="Respuesta vacía del LLM")
        except RuntimeError as e:
            logger.error("LLM (tools) falló: %s", e)
            return LLMResult(ok=False, error=str(e))
        except Exception as e:
            logger.error("LLM (tools) error inesperado: %s", e)
            return LLMResult(ok=False, error=str(e))
