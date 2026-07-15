"""
Sistema configurable de inferencia de contexto de captura.

Reemplaza la logica hardcoded de _infer_context en cli.py y nomenclar.py
con un sistema basado en reglas que puede extenderse via archivo JSON.

Reglas por defecto cubren dominios comunes (censo, hospital, seguro, etc.).
Para agregar reglas especificas, crear context_rules.json junto al nomenclador:

[
  {
    "keywords": ["mi_dominio", "midominio"],
    "context_label": "mi_dominio",
    "population": "mi poblacion",
    "capture_method": "mi metodo"
  }
]
"""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from .log_config import get_logger

log = get_logger("context_rules")

_CONFIG_PATH = Path(__file__).parent.parent / "nomenclador" / "context_rules.json"


@dataclass
class ContextRule:
    keywords: list[str]
    context_label: str
    population: str
    capture_method: str

    def matches(self, source_name_lower: str) -> bool:
        return any(kw in source_name_lower for kw in self.keywords)


_DEFAULT_RULES: list[ContextRule] = [
    ContextRule(
        keywords=["censo", "agri"],
        context_label="censo_agricola",
        population="productores agricolas",
        capture_method="censo",
    ),
    ContextRule(
        keywords=["monitoreo", "ambiental", "marn"],
        context_label="monitoreo_ambiental",
        population="ecosistemas nacionales",
        capture_method="monitoreo remoto",
    ),
    ContextRule(
        keywords=["agri", "mag"],
        context_label="agricola",
        population="productores agricolas",
        capture_method="censo",
    ),
    ContextRule(
        keywords=["censo"],
        context_label="censo",
        population="poblacion general",
        capture_method="auto-reporte",
    ),
    ContextRule(
        keywords=["hospital", "clinica"],
        context_label="hospital",
        population="pacientes atendidos",
        capture_method="registro clinico",
    ),
    ContextRule(
        keywords=["seguro", "aseguradora", "afiliacion"],
        context_label="seguro",
        population="afiliados",
        capture_method="formulario administrativo",
    ),
    ContextRule(
        keywords=["encuesta", "survey", "muestra"],
        context_label="encuesta",
        population="muestra poblacional",
        capture_method="entrevista",
    ),
    ContextRule(
        keywords=["registro", "padron"],
        context_label="registro",
        population="poblacion registrada",
        capture_method="registro administrativo",
    ),
    ContextRule(
        keywords=["financiero", "contable", "presupuesto"],
        context_label="financiero",
        population="transacciones",
        capture_method="sistema transaccional",
    ),
    ContextRule(
        keywords=["educacion", "escolar", "matricula"],
        context_label="educacion",
        population="estudiantes matriculados",
        capture_method="registro academico",
    ),
]

_rules_cache: Optional[list[ContextRule]] = None


def _load_rules() -> list[ContextRule]:
    global _rules_cache
    if _rules_cache is not None:
        return _rules_cache

    rules = list(_DEFAULT_RULES)

    if _CONFIG_PATH.exists():
        try:
            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                custom = json.load(f)
            for entry in custom:
                rules.append(ContextRule(
                    keywords=entry.get("keywords", []),
                    context_label=entry.get("context_label", ""),
                    population=entry.get("population", "no definida"),
                    capture_method=entry.get("capture_method", "no definida"),
                ))
            log.debug("Reglas de contexto custom cargadas desde %s (%d reglas)", _CONFIG_PATH, len(custom))
        except Exception as e:
            log.warning("Error cargando context_rules.json: %s — usando reglas por defecto", e)

    _rules_cache = rules
    return rules


def infer_context(source_name: str) -> dict:
    """Inferir contexto de captura basándose en el nombre de la fuente.

    Usa reglas configurables (defaults + context_rules.json si existe).
    Retorna dict con context_label, population y capture_method.
    """
    name_lower = source_name.lower()
    for rule in _load_rules():
        if rule.matches(name_lower):
            return {
                "context_label": rule.context_label,
                "population": rule.population,
                "capture_method": rule.capture_method,
            }
    return {
        "context_label": source_name,
        "population": "no definida",
        "capture_method": "no definida",
    }


def clear_rules_cache():
    """Limpiar cache de reglas (para tests o recarga)."""
    global _rules_cache
    _rules_cache = None
