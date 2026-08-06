"""Deteccion de anomalias y limpieza automatica de datos. Sin LLM."""

import re
import math
import statistics
from collections import Counter

from .statistics import _to_float_list, _percentile


def column_quality_score(values: list[str]) -> dict:
    """Calcular score de calidad 0-100 de una columna.

    Metricas: completitud, consistencia, validez.
    Retorna dict con score, grade, completeness, consistency, validity, issues.
    """
    total = len(values) if values else 0
    if total == 0:
        return {"score": 0, "grade": "F", "completeness": 0.0, "consistency": 0.0, "validity": 0.0, "issues": ["columna vacia"]}

    non_null = [v for v in values if v is not None and str(v).strip() and str(v).strip().lower() not in ("", "na", "n/a", "null", "none")]
    completeness = len(non_null) / total

    # Consistencia: valores unicos vs total (baja cardinalidad = mas consistente para categoricos)
    unique = set(str(v).strip().upper() for v in non_null)
    cardinality = len(unique)

    # Detectar tipo
    floats = _to_float_list(non_null)
    is_numeric = len(floats) > len(non_null) * 0.8 if non_null else False

    issues = []

    # Completitud
    if completeness < 0.5:
        issues.append(f"alta nulidad: {1 - completeness:.0%}")
    elif completeness < 0.9:
        issues.append(f"nulidad moderada: {1 - completeness:.0%}")

    # Validez: para numericos, verificar que se pueden parsear
    if is_numeric:
        validity = len(floats) / max(len(non_null), 1)
    else:
        # Para categoricos, validez = 1 - proporcion de valores con caracteres raros
        weird = sum(1 for v in non_null if re.search(r"[^\x00-\x7F]", str(v)) and not re.match(r"[\w\s\-\.\,\(\)]+", str(v)))
        validity = 1 - (weird / max(len(non_null), 1))

    # Consistencia: para categoricos de baja cardinalidad, verificar valores consistentes
    if not is_numeric and cardinality < 20:
        # Verificar si hay variantes del mismo valor (case, whitespace)
        lower_unique = set(str(v).strip().lower() for v in non_null)
        consistency = len(lower_unique) / max(cardinality, 1)
        if consistency < 0.8:
            issues.append(f"posibles duplicados por case/whitespace: {cardinality} -> {len(lower_unique)} valores")
    else:
        consistency = 1.0 if cardinality > 0 else 0.0

    # Score ponderado
    score = int(completeness * 40 + consistency * 30 + validity * 30)

    if score >= 90:
        grade = "A"
    elif score >= 80:
        grade = "B"
    elif score >= 70:
        grade = "C"
    elif score >= 60:
        grade = "D"
    else:
        grade = "F"

    return {
        "score": score,
        "grade": grade,
        "completeness": completeness,
        "consistency": consistency,
        "validity": validity,
        "issues": issues,
    }


def detect_numeric_anomalies(values: list[str], method: str = "iqr") -> dict:
    """Detectar anomalias en datos numericos (IQR o zscore).

    Retorna dict con anomaly_count, anomaly_ratio, bounds, anomalies.
    """
    floats = _to_float_list(values)
    if len(floats) < 4:
        return {"anomaly_count": 0, "anomaly_ratio": 0.0, "bounds": {"lower": 0, "upper": 0}, "anomalies": []}

    sorted_vals = sorted(floats)

    if method == "zscore":
        mean = statistics.mean(floats)
        std = statistics.stdev(floats) if len(floats) > 1 else 0
        if std == 0:
            return {"anomaly_count": 0, "anomaly_ratio": 0.0, "bounds": {"lower": mean, "upper": mean}, "anomalies": []}
        threshold = 3.0
        lower = mean - threshold * std
        upper = mean + threshold * std
    else:  # IQR
        q1 = _percentile(sorted_vals, 25)
        q3 = _percentile(sorted_vals, 75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

    anomalies = []
    for i, v in enumerate(values):
        try:
            f = float(str(v).strip())
        except (ValueError, TypeError):
            continue
        if f < lower or f > upper:
            reason = "fuera de rango IQR" if method == "iqr" else f"zscore > {threshold}"
            anomalies.append({"index": i, "value": f, "reason": reason})

    return {
        "anomaly_count": len(anomalies),
        "anomaly_ratio": len(anomalies) / max(len(floats), 1),
        "bounds": {"lower": lower, "upper": upper},
        "anomalies": anomalies,
    }


def detect_categorical_anomalies(values: list[str]) -> dict:
    """Detectar anomalias en datos categoricos.

    Retorna dict con rare_values, high_cardinality, cardinality, suspicious_patterns.
    """
    non_null = [str(v).strip() for v in values if v is not None and str(v).strip() and str(v).strip().lower() not in ("", "na", "n/a", "null", "none")]
    if not non_null:
        return {"rare_values": [], "high_cardinality": False, "cardinality": 0, "suspicious_patterns": []}

    counts = Counter(non_null)
    total = len(non_null)
    cardinality = len(counts)

    # Valores raros: frecuencia < 1% del total
    rare_values = [
        {"value": v, "freq": c}
        for v, c in counts.items()
        if c / total < 0.01 and total > 10
    ]
    rare_values.sort(key=lambda x: x["freq"])

    high_cardinality = cardinality > total * 0.9 and total > 20

    # Patrones sospechosos: mojibake, espacios raros, caracteres no imprimibles
    suspicious_patterns = []
    for i, v in enumerate(non_null):
        issues = []
        if re.search(r"[^\x00-\x7F]", v):
            issues.append("mojibake")
        if v != v.strip():
            issues.append("whitespace")
        if re.search(r"\s{2,}", v):
            issues.append("espacios multiples")
        if issues:
            suspicious_patterns.append({"index": i, "value": v, "issues": issues})

    return {
        "rare_values": rare_values[:20],
        "high_cardinality": high_cardinality,
        "cardinality": cardinality,
        "suspicious_patterns": suspicious_patterns[:20],
    }


_ENCODING_FIXES = {
    "a~o": "ano",
    "Ã±": "n",
    "Ã¡": "a",
    "Ã©": "e",
    "Ã­": "i",
    "Ã³": "o",
    "Ãº": "u",
    "Ã": "A",
    "Ã‰": "E",
    "Ã": "I",
    "Ã“": "O",
    "Ãš": "U",
}


def auto_clean(values: list[str]) -> dict:
    """Limpiar una columna: whitespace, encoding mojibake, fill de valores faltantes.

    Retorna dict con total_changes, fixes_applied, changes.
    """
    changes = []
    fixes_applied = set()

    for i, v in enumerate(values):
        if v is None:
            changes.append({"index": i, "original": "", "cleaned": "", "fixes": ["null_to_empty"]})
            fixes_applied.add("null_fill")
            continue

        original = str(v)
        cleaned = original
        fixes = []

        # Fix encoding
        for bad, good in _ENCODING_FIXES.items():
            if bad in cleaned:
                cleaned = cleaned.replace(bad, good)
                fixes.append("encoding")

        # Strip whitespace
        stripped = cleaned.strip()
        if stripped != cleaned:
            fixes.append("whitespace")
            cleaned = stripped

        # Fix multiple spaces
        if re.search(r"\s{2,}", cleaned):
            cleaned = re.sub(r"\s+", " ", cleaned)
            fixes.append("multiple_spaces")

        if fixes:
            changes.append({"index": i, "original": original, "cleaned": cleaned, "fixes": fixes})
            fixes_applied.update(fixes)

    return {
        "total_changes": len(changes),
        "fixes_applied": list(fixes_applied),
        "changes": changes,
    }
