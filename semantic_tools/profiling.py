"""Profiling de datasets CSV. Sin pandas, usa stdlib csv."""

import csv
import re
from collections import Counter
from pathlib import Path
from .statistics import _to_float_list, distribution_summary
from .quality import column_quality_score


def profile_csv(file_path: str, max_rows: int = 10000) -> dict:
    """Perfilar todas las columnas de un CSV.

    Retorna dict con source, total_rows, columns (lista de dicts con stats por columna).
    """
    p = Path(file_path)
    source_name = p.stem

    with open(p, "r", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        rows = []
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            rows.append(row)

    if not rows:
        return {"source": source_name, "total_rows": 0, "columns": []}

    headers = list(rows[0].keys())
    columns = []

    for h in headers:
        values = [row.get(h, "") for row in rows]
        dist = distribution_summary(values)
        quality = column_quality_score(values)

        floats = _to_float_list([v for v in values if v and str(v).strip()])
        is_numeric = len(floats) > len([v for v in values if v and str(v).strip()]) * 0.8 if values else False

        col_info = {
            "name": h,
            "type": dist["type"],
            "count": dist["count"],
            "unique": dist["unique"],
            "missing": dist["missing"],
            "quality_score": quality["score"],
            "quality_grade": quality["grade"],
        }

        if is_numeric and len(floats) >= 2:
            col_info.update({
                "mean": round(dist.get("mean", 0), 4),
                "median": round(dist.get("median", 0), 4),
                "std": round(dist.get("std", 0), 4),
                "min": dist.get("min", 0),
                "max": dist.get("max", 0),
            })
        else:
            col_info["top_values"] = dist.get("top_5", {})

        columns.append(col_info)

    return {
        "source": source_name,
        "total_rows": len(rows),
        "columns": columns,
    }


def format_profile_summary(result: dict) -> str:
    """Formatear el resultado de profile_csv como texto legible."""
    lines = [
        f"Perfil de dataset: {result['source']}",
        f"  Filas: {result['total_rows']} | Columnas: {len(result['columns'])}",
        "",
    ]

    for col in result["columns"]:
        type_icon = "N" if col["type"] == "numeric" else ("D" if col["type"] == "date" else "C")
        quality_icon = col["quality_grade"]
        lines.append(
            f"  [{type_icon}] {col['name']} | "
            f"tipo={col['type']} | "
            f"unicos={col['unique']} | "
            f"missing={col['missing']} | "
            f"calidad={col['quality_score']}/100 ({quality_icon})"
        )
        if col["type"] == "numeric":
            lines.append(f"       media={col.get('mean', '?')} | mediana={col.get('median', '?')} | std={col.get('std', '?')} | rango=[{col.get('min', '?')}, {col.get('max', '?')}]")
        elif col.get("top_values"):
            top = ", ".join(f"{v}({c})" for v, c in list(col["top_values"].items())[:3])
            lines.append(f"       top: {top}")

    # Resumen de calidad
    grades = [c["quality_grade"] for c in result["columns"]]
    avg_score = sum(c["quality_score"] for c in result["columns"]) / max(len(grades), 1)
    lines.append("")
    lines.append(f"  Calidad promedio: {avg_score:.0f}/100 | Grades: A={grades.count('A')} B={grades.count('B')} C={grades.count('C')} D={grades.count('D')} F={grades.count('F')}")

    return "\n".join(lines)
