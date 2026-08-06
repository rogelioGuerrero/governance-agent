"""Matcheo automatico de columnas entre dos datasets. Sin LLM."""

import csv
from pathlib import Path
from collections import Counter

from .similarity import cosine_similarity, jaccard_similarity, composite_similarity
from .statistics import _to_float_list, welch_t_test, chi_square_test
from .quality import column_quality_score
from .profiling import profile_csv


def _load_columns(csv_path: str, max_rows: int = 500) -> dict:
    """Cargar columnas de un CSV como dict {name: list[str]}."""
    p = Path(csv_path)
    with open(p, "r", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        rows = []
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            rows.append(row)
    if not rows:
        return {}
    headers = list(rows[0].keys())
    return {h: [row.get(h, "") for row in rows] for h in headers}


def auto_match(csv_a: str, csv_b: str) -> dict:
    """Matchear automaticamente columnas entre dos datasets.

    Ejecuta similarity + statistics + quality en todos los pares.
    Retorna dict con high_confidence, medium_confidence, low_confidence, total_pairs.
    """
    cols_a = _load_columns(csv_a)
    cols_b = _load_columns(csv_b)

    if not cols_a or not cols_b:
        return {"high_confidence": [], "medium_confidence": [], "low_confidence": [], "total_pairs": 0}

    results = []
    for name_a, vals_a in cols_a.items():
        for name_b, vals_b in cols_b.items():
            # Skip ID-like columns
            if name_a.lower() in ("id", "id_paciente", "index") or name_b.lower() in ("id", "id_paciente", "index"):
                continue

            # Similarity scores
            cos = cosine_similarity(vals_a, vals_b)
            jac = jaccard_similarity(vals_a, vals_b)

            # Statistical test
            floats_a = _to_float_list(vals_a)
            floats_b = _to_float_list(vals_b)
            is_numeric = len(floats_a) > len(vals_a) * 0.8 and len(floats_b) > len(vals_b) * 0.8

            stat_score = 0.0
            if is_numeric and len(floats_a) >= 3 and len(floats_b) >= 3:
                t_result = welch_t_test(floats_a, floats_b)
                if not t_result["significant"]:
                    stat_score = 1.0 - abs(t_result["t_stat"]) / 10.0
                    stat_score = max(0.0, min(1.0, stat_score))
            else:
                chi_result = chi_square_test(vals_a, vals_b)
                if not chi_result["significant"]:
                    stat_score = 0.7

            # Composite confidence
            confidence = cos * 0.4 + jac * 0.3 + stat_score * 0.3

            results.append({
                "column_a": name_a,
                "column_b": name_b,
                "cosine": round(cos, 4),
                "jaccard": round(jac, 4),
                "stat_score": round(stat_score, 4),
                "confidence": round(confidence, 4),
            })

    # Clasificar por confianza
    high = [r for r in results if r["confidence"] >= 0.7]
    medium = [r for r in results if 0.4 <= r["confidence"] < 0.7]
    low = [r for r in results if r["confidence"] < 0.4]

    high.sort(key=lambda x: x["confidence"], reverse=True)
    medium.sort(key=lambda x: x["confidence"], reverse=True)

    return {
        "high_confidence": high,
        "medium_confidence": medium,
        "low_confidence": low,
        "total_pairs": len(results),
    }


def format_match_result(result: dict) -> str:
    """Formatear el resultado de auto_match como texto legible."""
    lines = [
        f"Auto-match: {result['total_pairs']} pares evaluados",
        f"  Alta confianza (>=0.7): {len(result['high_confidence'])}",
        f"  Media confianza (0.4-0.7): {len(result['medium_confidence'])}",
        f"  Baja confianza (<0.4): {len(result['low_confidence'])}",
        "",
    ]

    if result["high_confidence"]:
        lines.append("=== ALTA CONFIANZA ===")
        for m in result["high_confidence"][:10]:
            lines.append(
                f"  {m['column_a']} <-> {m['column_b']} "
                f"(conf={m['confidence']:.2f}, cos={m['cosine']:.2f}, jac={m['jaccard']:.2f})"
            )

    if result["medium_confidence"]:
        lines.append("\n=== MEDIA CONFIANZA ===")
        for m in result["medium_confidence"][:10]:
            lines.append(
                f"  {m['column_a']} <-> {m['column_b']} "
                f"(conf={m['confidence']:.2f})"
            )

    return "\n".join(lines)
