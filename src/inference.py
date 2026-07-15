"""
Motor de inferencia semántica — capa intermedia entre detección de estándares y LLM.

3 mecanismos en orden de confianza:
1. Patrones de tipo semántico (regex/heurísticas) → high confidence
2. Listas de referencia (soft standards) → high/medium confidence
3. Huella de valores (overlap coefficient contra conceptos existentes) → high/medium confidence

El objetivo es resolver ~75% de las columnas sin usar LLM, reduciendo costo
y carga humana. Solo lo que no encaja llega al LLM con --llm.
"""

import re
import csv
import unicodedata
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from .log_config import get_logger

log = get_logger("inference")

REFERENCE_LISTS_DIR = Path(__file__).parent / "reference_lists"

_RE_WHITESPACE = re.compile(r"\s+")
_RE_DIGITS_ONLY = re.compile(r"^-?[\d.]+$")
_RE_DIGITS_COMMA = re.compile(r"^-?[\d.,]+$")

# === Normalización ===

def _normalize(value: str) -> str:
    """Normalizar valor: lowercase, sin acentos, sin espacios extra."""
    if value is None:
        return ""
    s = str(value).strip().lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = _RE_WHITESPACE.sub(" ", s)
    return s


def _normalize_set(values: list[str]) -> set[str]:
    """Normalizar lista de valores a un set."""
    return {_normalize(v) for v in values if v is not None and str(v).strip()}


# === Resultado ===

@dataclass
class InferenceResult:
    matched_concept: Optional[str] = None
    semantic_type: Optional[str] = None
    reference_match: Optional[str] = None
    confidence: str = "low"
    reason: str = ""
    suggested_concept_name: Optional[str] = None
    suggested_standard_id: Optional[str] = None
    overlap_score: float = 0.0


# === Mecanismo 1: Patrones de tipo semántico ===

_PATTERNS = {
    "date_iso": re.compile(r"^\d{4}-\d{2}-\d{2}$"),
    "date_latino": re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$"),
    "dui_sv": re.compile(r"^\d{8}-\d$"),
    "nit_sv": re.compile(r"^\d{4}-\d{6}-\d{3}-\d{3}$"),
    "phone_sv": re.compile(r"^(\+?503)?\s?\d{4}-?\d{4}$"),
    "email": re.compile(r"^[\w.-]+@[\w.-]+\.\w+$"),
    "year": re.compile(r"^(19|20)\d{2}$"),
}

_BOOLEAN_VALUES = {
    "s", "n", "si", "no", "y", "t", "f", "true", "false",
    "verdadero", "falso", "0", "1",
}

# Tipos semánticos definitivos (alta confianza)
_DEFINITIVE_TYPES = {"date", "identifier_dui", "identifier_nit", "email", "boolean", "age", "percentage", "year"}


def _detect_semantic_type(column_name: str, sample_values: list[str], unique_count: int) -> Optional[dict]:
    """Detectar tipo semántico por patrones y heurísticas."""
    name_lower = _normalize(column_name)
    samples = [str(v).strip() for v in sample_values if v is not None and str(v).strip()]

    if not samples:
        return None

    # Fecha
    for s in samples:
        if _PATTERNS["date_iso"].match(s) or _PATTERNS["date_latino"].match(s):
            return {"type": "date", "reason": "Valores coinciden con patron de fecha"}

    # Año (entero de 4 dígitos 1900-2099 + nombre sugiere año)
    if any(k in name_lower for k in ("anio", "ano", "year", "periodo", "ejercicio")):
        year_matches = [s for s in samples if _PATTERNS["year"].match(s)]
        if year_matches and len(year_matches) / len(samples) > 0.8:
            return {"type": "year", "reason": f"Valores de año ({year_matches[0]}-{year_matches[-1]}) y nombre sugiere año"}

    # DUI
    for s in samples:
        if _PATTERNS["dui_sv"].match(s):
            return {"type": "identifier_dui", "reason": "Valores coinciden con formato DUI (El Salvador)"}

    # NIT
    for s in samples:
        if _PATTERNS["nit_sv"].match(s):
            return {"type": "identifier_nit", "reason": "Valores coinciden con formato NIT (El Salvador)"}

    # Email
    for s in samples:
        if _PATTERNS["email"].match(s):
            return {"type": "email", "reason": "Valores coinciden con formato email"}

    # Booleano
    if unique_count <= 2:
        normalized = {_normalize(s) for s in samples}
        if normalized.issubset(_BOOLEAN_VALUES):
            return {"type": "boolean", "reason": f"Valores binarios: {', '.join(sorted(normalized))}"}

    # Edad
    if 0 < unique_count <= 120:
        try:
            nums = [int(float(s)) for s in samples if _RE_DIGITS_ONLY.match(s)]
            if nums and all(0 <= n <= 120 for n in nums):
                if any(k in name_lower for k in ("edad", "age", "anos")):
                    return {"type": "age", "reason": "Valores enteros 0-120 y nombre sugiere edad"}
        except ValueError:
            pass

    # Porcentaje
    try:
        nums = [float(s) for s in samples if _RE_DIGITS_COMMA.match(s)]
        if nums and all(0 <= n <= 100 for n in nums):
            if any(k in name_lower for k in ("porcentaje", "percent", "rate", "tasa", "%")):
                return {"type": "percentage", "reason": "Valores 0-100 y nombre sugiere porcentaje"}
    except ValueError:
        pass

    # Geográfico (heurística: capitaliza, cardinalidad baja-media, no numérico)
    if 2 < unique_count <= 500:
        non_numeric = [s for s in samples if not _RE_DIGITS_COMMA.match(s)]
        if len(non_numeric) == len(samples):
            capitalized = [s for s in non_numeric if s and s[0].isupper()]
            if len(capitalized) / len(samples) > 0.7:
                return {
                    "type": "geographic_candidate",
                    "reason": f"Valores capitalizados, cardinalidad {unique_count}, no numericos — candidato geografico",
                }

    return None


# === Mecanismo 2: Listas de referencia ===

_reference_cache: dict[str, set[str]] = {}


def _load_reference_lists() -> dict[str, set[str]]:
    """Cargar listas de referencia desde CSVs en reference_lists/."""
    if _reference_cache:
        return _reference_cache

    ref_dir = REFERENCE_LISTS_DIR
    if not ref_dir.exists():
        log.warning("Directorio de listas de referencia no encontrado: %s", ref_dir)
        return {}

    for csv_file in ref_dir.glob("*.csv"):
        name = csv_file.stem
        values = set()
        try:
            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if row and row[0].strip():
                        values.add(_normalize(row[0]))
        except Exception as e:
            log.warning("Error cargando lista de referencia %s: %s", csv_file, e)
            continue

        if values:
            _reference_cache[name] = values
            log.debug("Lista de referencia cargada: %s (%d valores)", name, len(values))

    return _reference_cache


def _match_reference_list(values: set[str]) -> Optional[dict]:
    """Comparar valores contra listas de referencia."""
    ref_lists = _load_reference_lists()
    if not ref_lists or not values:
        return None

    best_match = None
    best_overlap = 0.0

    for name, ref_values in ref_lists.items():
        if not ref_values:
            continue

        denom = min(len(values), len(ref_values))
        if denom == 0:
            continue

        overlap = len(values & ref_values) / denom

        if overlap > best_overlap:
            best_overlap = overlap
            best_match = name

        if overlap >= 0.80:
            return {
                "reference": name,
                "overlap": overlap,
                "confidence": "high",
                "reason": f"Valores coinciden con lista '{name}' ({overlap:.0%} overlap)",
            }

    if best_match and best_overlap >= 0.60:
        return {
            "reference": best_match,
            "overlap": best_overlap,
            "confidence": "medium",
            "reason": f"Valores coinciden parcialmente con lista '{best_match}' ({best_overlap:.0%} overlap)",
        }

    return None


# === Mecanismo 3: Huella de valores (overlap contra conceptos existentes) ===

def _fingerprint_match(
    values: set[str],
    existing_concepts: list[dict],
) -> Optional[dict]:
    """Comparar valores contra campos de conceptos existentes en el grafo."""
    if not values or not existing_concepts:
        return None

    best_match = None
    best_overlap = 0.0

    for concept in existing_concepts:
        concept_values = set()
        for field in concept.get("fields", []):
            for sv in field.get("sample_values", []):
                concept_values.add(_normalize(sv))

        if not concept_values:
            continue

        denom = min(len(values), len(concept_values))
        if denom == 0:
            continue

        overlap = len(values & concept_values) / denom

        if overlap > best_overlap:
            best_overlap = overlap
            best_match = concept

    if best_match and best_overlap >= 0.80:
        return {
            "concept_id": best_match.get("id"),
            "concept_name": best_match.get("name"),
            "overlap": best_overlap,
            "confidence": "high",
            "reason": f"Valores coinciden con concepto existente '{best_match.get('name')}' ({best_overlap:.0%} overlap)",
        }

    if best_match and best_overlap >= 0.60:
        return {
            "concept_id": best_match.get("id"),
            "concept_name": best_match.get("name"),
            "overlap": best_overlap,
            "confidence": "medium",
            "reason": f"Valores coinciden parcialmente con concepto '{best_match.get('name')}' ({best_overlap:.0%} overlap)",
        }

    return None


# === Mapeo de tipos a conceptos sugeridos ===

_PATTERN_CONCEPT_MAP = {
    "date": ("fecha", "ISO_8601"),
    "identifier_dui": ("dui", None),
    "identifier_nit": ("nit", None),
    "email": ("email", None),
    "boolean": ("indicador_booleano", None),
    "age": ("edad", None),
    "percentage": ("porcentaje", None),
    "year": ("anio", None),
}

_REFERENCE_CONCEPT_MAP = {
    "departamentos_sv": "departamento",
    "municipios_sv": "municipio",
    "meses_es": "mes",
    "dias_semana_es": "dia_semana",
    "genero_binario": "sexo",
    "estado_civil": "estado_civil",
    "nivel_educativo": "nivel_educativo",
    "tipo_sangre": "tipo_sangre",
}


# === Función principal ===

def infer_semantic_type(
    column_name: str,
    sample_values: list[str],
    unique_count: int,
    existing_concepts: list[dict] | None = None,
) -> InferenceResult:
    """
    Inferir tipo semántico de una columna sin usar LLM.

    3 mecanismos en orden de confianza:
    1. Patrones de tipo semántico (regex/heurísticas) → high
    2. Listas de referencia (soft standards) → high/medium
    3. Huella de valores (overlap contra conceptos existentes) → high/medium

    Returns InferenceResult con confidence high/medium/low.
    """
    existing_concepts = existing_concepts or []
    normalized_values = _normalize_set(sample_values)

    # 1. Patrones de tipo semántico
    pattern_result = _detect_semantic_type(column_name, sample_values, unique_count)

    # Patrones definitivos → alta confianza inmediata
    if pattern_result and pattern_result["type"] in _DEFINITIVE_TYPES:
        sem_type = pattern_result["type"]
        suggested_name, suggested_std = _PATTERN_CONCEPT_MAP.get(sem_type, (None, None))
        return InferenceResult(
            semantic_type=sem_type,
            confidence="high",
            reason=pattern_result["reason"],
            suggested_concept_name=suggested_name,
            suggested_standard_id=suggested_std,
        )

    # 2. Listas de referencia
    ref_result = _match_reference_list(normalized_values)
    if ref_result:
        suggested_name = _REFERENCE_CONCEPT_MAP.get(ref_result["reference"], ref_result["reference"])
        return InferenceResult(
            reference_match=ref_result["reference"],
            confidence=ref_result["confidence"],
            reason=ref_result["reason"],
            suggested_concept_name=suggested_name,
            overlap_score=ref_result["overlap"],
        )

    # 3. Huella de valores contra conceptos existentes
    fp_result = _fingerprint_match(normalized_values, existing_concepts)
    if fp_result:
        return InferenceResult(
            matched_concept=fp_result["concept_id"],
            confidence=fp_result["confidence"],
            reason=fp_result["reason"],
            suggested_concept_name=fp_result.get("concept_name"),
            overlap_score=fp_result["overlap"],
        )

    # 4. Fallback: candidato geográfico si se detectó
    if pattern_result and pattern_result["type"] == "geographic_candidate":
        return InferenceResult(
            semantic_type="geographic_candidate",
            confidence="low",
            reason=pattern_result["reason"],
        )

    return InferenceResult(confidence="low", reason="Sin inferencia posible")


def clear_reference_cache():
    """Limpiar cache de listas de referencia (para tests o recarga)."""
    global _reference_cache
    _reference_cache = {}
