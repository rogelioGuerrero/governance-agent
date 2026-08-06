"""
Rapid Assessment — Pass 1 del flujo de dos pasadas.

Diagnostico automatico de un dataset sin contexto humano:
1. Perfilar columnas (tipos, nulos, distribuciones)
2. Calcular quality score por columna (A-F)
3. Detectar anomalias (outliers IQR)
4. Inferir tipo semantico (inference engine)
5. Detectar PII / datos sensibles
6. Detectar problemas (encoding, constantes, alta nulidad)
7. Matching contra conceptos existentes en el nomenclador
8. Generar reporte markdown estructurado

Entregable: reporte rapid assessment con seccion de "accion requerida"
listando variables que necesitan contexto humano.

Uso:
    uv run python -m src.cli rapid-assessment <csv> [--output report.md]
"""

import csv
import os
import re
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from .profiler import profile_csv, TableProfile, ColumnProfile
from .inference import infer_semantic_type
from .standards import detect_standard, STANDARDS
from .graph.catalog import load_graph_cached
from .rag_factory import detect_issues, clean_column_name, RawColumn
from .nomenclar import _PII_KEYWORDS, _SENSIBLE_KEYWORDS

try:
    from semantic_tools.quality import column_quality_score, detect_numeric_anomalies
except ImportError:
    from ..semantic_tools.quality import column_quality_score, detect_numeric_anomalies


@dataclass
class ColumnAssessment:
    """Resultado del assessment de una columna."""
    name: str
    clean_name: str
    data_type: str
    total_count: int
    null_count: int
    unique_count: int
    sample_values: list[str] = field(default_factory=list)
    min_value: Optional[str] = None
    max_value: Optional[str] = None

    # Quality
    quality_score: int = 0
    quality_grade: str = "F"
    completeness: float = 0.0
    consistency: float = 0.0
    validity: float = 0.0
    issues: list[str] = field(default_factory=list)

    # Anomalies
    anomaly_count: int = 0
    anomaly_ratio: float = 0.0

    # Inference
    inferred_type: Optional[str] = None
    inferred_confidence: str = "low"
    inferred_reason: str = ""
    suggested_concept: Optional[str] = None
    suggested_standard: Optional[str] = None
    matched_concept_id: Optional[str] = None

    # Sensitivity
    is_pii: bool = False
    is_sensitive: bool = False
    sensitivity_reason: str = ""

    # Matching
    match_status: str = "unmatched"  # matched, inferred, unmatched
    match_detail: str = ""


@dataclass
class RapidAssessmentReport:
    """Reporte completo del rapid assessment."""
    source_file: str
    source_name: str
    generated_at: str
    total_rows: int
    total_columns: int

    # Global metrics
    avg_quality_score: float = 0.0
    global_grade: str = "F"

    # Column assessments
    columns: list[ColumnAssessment] = field(default_factory=list)

    # Summary counts
    matched_count: int = 0
    inferred_count: int = 0
    unmatched_count: int = 0
    pii_count: int = 0
    sensitive_count: int = 0
    issues_count: int = 0

    # Interop potential
    interop_candidates: list[dict] = field(default_factory=list)


def _check_sensitivity(column_name: str) -> tuple[bool, bool, str]:
    """Verificar si una columna es PII o sensible por nombre.

    Returns: (is_pii, is_sensitive, reason)
    """
    name_lower = column_name.lower().strip()
    clean = re.sub(r"[_\-\s]", "", name_lower)

    for kw in _PII_KEYWORDS:
        if kw in name_lower or kw in clean:
            return True, False, f"Nombre coincide con keyword PII: '{kw}'"

    for kw in _SENSIBLE_KEYWORDS:
        if kw in name_lower or kw in clean:
            return False, True, f"Nombre coincide con keyword sensible: '{kw}'"

    return False, False, ""


def _get_existing_concepts_for_matching() -> list[dict]:
    """Obtener conceptos existentes del grafo para fingerprint matching."""
    try:
        g = load_graph_cached()
        concepts = []
        for node_id, node_data in g.graph.nodes(data=True):
            if node_data.get("type") == "concept":
                fields = []
                for fn_id, fn_data in g.graph.nodes(data=True):
                    if fn_data.get("type") == "field":
                        fields.append({
                            "sample_values": fn_data.get("sample_values", []),
                        })
                concepts.append({
                    "id": node_id,
                    "name": node_data.get("name", ""),
                    "fields": fields,
                })
        return concepts
    except Exception:
        return []


def _detect_interop_candidates(
    source_name: str,
    columns: list[ColumnAssessment],
) -> list[dict]:
    """Detectar columnas que podrian puentear con otros datasets en el grafo."""
    candidates = []
    seen = set()
    try:
        g = load_graph_cached()
        # Obtener fuentes distintas a la actual
        other_sources = []
        for node_id, node_data in g.graph.nodes(data=True):
            if node_data.get("type") == "source" and node_data.get("name", "") != source_name:
                other_sources.append({
                    "id": node_id,
                    "name": node_data.get("name", ""),
                })

        if not other_sources:
            return []

        # Por cada columna matched o inferred, verificar si el concepto existe en otras fuentes
        for col in columns:
            if col.matched_concept_id or col.suggested_concept:
                concept_name = col.suggested_concept or col.matched_concept_id
                for src in other_sources:
                    key = (col.name, src["name"])
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append({
                        "column": col.name,
                        "concept": concept_name,
                        "target_source": src["name"],
                        "confidence": col.inferred_confidence,
                    })
    except Exception:
        pass

    return candidates


def assess_csv(csv_path: str) -> RapidAssessmentReport:
    """Ejecutar rapid assessment sobre un CSV.

    No requiere LLM. No requiere contexto humano.
    Usa profiling + inference engine + quality scoring + PII detection.
    """
    source_name = Path(csv_path).stem
    tables = profile_csv(csv_path)

    if not tables:
        return RapidAssessmentReport(
            source_file=csv_path,
            source_name=source_name,
            generated_at=datetime.now().isoformat(timespec="seconds"),
            total_rows=0,
            total_columns=0,
        )

    table = tables[0]
    existing_concepts = _get_existing_concepts_for_matching()

    report = RapidAssessmentReport(
        source_file=csv_path,
        source_name=source_name,
        generated_at=datetime.now().isoformat(timespec="seconds"),
        total_rows=table.row_count,
        total_columns=len(table.columns),
    )

    for col_profile in table.columns:
        # Crear RawColumn para detect_issues
        raw_col = RawColumn(
            raw_name=col_profile.column,
            clean_name=clean_column_name(col_profile.column),
            data_type=col_profile.data_type,
            sample_values=col_profile.sample_values,
            null_count=col_profile.null_count,
            total_count=col_profile.total_count,
            unique_count=col_profile.unique_count,
        )

        # Quality score
        all_values = []
        try:
            with open(csv_path, "r", encoding="utf-8-sig", errors="replace") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    if i >= 10000:
                        break
                    all_values.append(row.get(col_profile.column, ""))
        except Exception:
            all_values = [str(v) for v in col_profile.sample_values]

        q = column_quality_score(all_values)

        # Anomalies (solo numericos)
        anomalies = {"anomaly_count": 0, "anomaly_ratio": 0.0}
        if col_profile.data_type in ("integer", "float"):
            anomalies = detect_numeric_anomalies(all_values)

        # Issues
        col_issues = detect_issues(raw_col)

        # Inference
        inf = infer_semantic_type(
            column_name=col_profile.column,
            sample_values=col_profile.sample_values,
            unique_count=col_profile.unique_count,
            existing_concepts=existing_concepts,
        )

        # Standard detection
        std_results = detect_standard(
            col_profile.column,
            col_profile.sample_values,
        )
        std_result = std_results[0] if std_results else None

        # Sensitivity
        is_pii, is_sensitive, sens_reason = _check_sensitivity(col_profile.column)

        # Match status
        match_status = "unmatched"
        match_detail = ""
        if inf.matched_concept:
            match_status = "matched"
            match_detail = f"Concepto: {inf.suggested_concept_name or inf.matched_concept}"
        elif inf.confidence in ("high", "medium") and inf.suggested_concept_name:
            match_status = "inferred"
            match_detail = f"Inferido: {inf.suggested_concept_name}"
            if inf.suggested_standard_id:
                match_detail += f" (estandar: {inf.suggested_standard_id})"
        elif std_result and std_result.get("standard"):
            match_status = "inferred"
            match_detail = f"Estandar: {std_result['standard']}"
            inf.suggested_standard = std_result["standard"]

        col_assessment = ColumnAssessment(
            name=col_profile.column,
            clean_name=raw_col.clean_name,
            data_type=col_profile.data_type,
            total_count=col_profile.total_count,
            null_count=col_profile.null_count,
            unique_count=col_profile.unique_count,
            sample_values=col_profile.sample_values[:10],
            min_value=col_profile.min_value,
            max_value=col_profile.max_value,
            quality_score=q["score"],
            quality_grade=q["grade"],
            completeness=q["completeness"],
            consistency=q["consistency"],
            validity=q["validity"],
            issues=col_issues + q.get("issues", []),
            anomaly_count=anomalies["anomaly_count"],
            anomaly_ratio=anomalies["anomaly_ratio"],
            inferred_type=inf.semantic_type or inf.reference_match,
            inferred_confidence=inf.confidence,
            inferred_reason=inf.reason,
            suggested_concept=inf.suggested_concept_name,
            suggested_standard=inf.suggested_standard_id,
            matched_concept_id=inf.matched_concept,
            is_pii=is_pii,
            is_sensitive=is_sensitive,
            sensitivity_reason=sens_reason,
            match_status=match_status,
            match_detail=match_detail,
        )

        report.columns.append(col_assessment)

        if match_status == "matched":
            report.matched_count += 1
        elif match_status == "inferred":
            report.inferred_count += 1
        else:
            report.unmatched_count += 1

        if is_pii:
            report.pii_count += 1
        if is_sensitive:
            report.sensitive_count += 1
        if col_assessment.issues:
            report.issues_count += 1

    # Global metrics
    if report.columns:
        report.avg_quality_score = sum(c.quality_score for c in report.columns) / len(report.columns)
        avg = report.avg_quality_score
        if avg >= 90:
            report.global_grade = "A"
        elif avg >= 80:
            report.global_grade = "B"
        elif avg >= 70:
            report.global_grade = "C"
        elif avg >= 60:
            report.global_grade = "D"
        else:
            report.global_grade = "F"

    # Interop candidates
    report.interop_candidates = _detect_interop_candidates(source_name, report.columns)

    return report


def format_report_markdown(report: RapidAssessmentReport) -> str:
    """Formatear el reporte como markdown para entregar al cliente."""
    lines = []
    p = lines.append

    p(f"# Rapid Assessment — {report.source_name}")
    p("")
    p(f"**Archivo:** `{report.source_file}`  ")
    p(f"**Fecha:** {report.generated_at}  ")
    p(f"**Filas:** {report.total_rows:,} | **Columnas:** {report.total_columns}")
    p("")

    # === RESUMEN EJECUTIVO ===
    p("## Resumen Ejecutivo")
    p("")
    p(f"| Metrica | Valor |")
    p(f"|---|---|")
    p(f"| Calidad global | **{report.avg_quality_score:.0f}/100** (Grade {report.global_grade}) |")
    p(f"| Columnas matcheadas a conceptos | {report.matched_count}/{report.total_columns} ({report.matched_count/max(report.total_columns,1)*100:.0f}%) |")
    p(f"| Columnas inferidas sin LLM | {report.inferred_count}/{report.total_columns} ({report.inferred_count/max(report.total_columns,1)*100:.0f}%) |")
    p(f"| Columnas sin match | **{report.unmatched_count}/{report.total_columns}** ({report.unmatched_count/max(report.total_columns,1)*100:.0f}%) |")
    p(f"| Columnas PII detectadas | {report.pii_count} |")
    p(f"| Columnas sensibles detectadas | {report.sensitive_count} |")
    p(f"| Columnas con problemas | {report.issues_count} |")
    p(f"| Candidatos interop | {len(report.interop_candidates)} |")
    p("")

    # === CALIDAD POR COLUMNA ===
    p("## Calidad por Columna")
    p("")
    p("| Columna | Tipo | Grade | Compl. | Consist. | Validez | Nulos | Unicos | Anomalias |")
    p("|---|---|---|---|---|---|---|---|---|")
    for c in report.columns:
        null_pct = f"{c.null_count}/{c.total_count}" if c.total_count else "0"
        anom = f"{c.anomaly_count} ({c.anomaly_ratio:.0%})" if c.anomaly_count > 0 else "-"
        p(f"| `{c.name}` | {c.data_type} | {c.quality_grade} | {c.completeness:.0%} | {c.consistency:.0%} | {c.validity:.0%} | {null_pct} | {c.unique_count} | {anom} |")
    p("")

    # === MATCHING SEMANTICO ===
    p("## Matching Semantico")
    p("")
    matched = [c for c in report.columns if c.match_status == "matched"]
    inferred = [c for c in report.columns if c.match_status == "inferred"]
    unmatched = [c for c in report.columns if c.match_status == "unmatched"]

    if matched:
        p(f"### Matcheados a conceptos existentes ({len(matched)})")
        p("")
        p("| Columna | Concepto | Confianza | Razon |")
        p("|---|---|---|---|")
        for c in matched:
            p(f"| `{c.name}` | {c.suggested_concept or c.matched_concept_id} | {c.inferred_confidence} | {c.inferred_reason} |")
        p("")

    if inferred:
        p(f"### Inferidos sin LLM ({len(inferred)})")
        p("")
        p("| Columna | Concepto sugerido | Estandar | Confianza | Razon |")
        p("|---|---|---|---|---|")
        for c in inferred:
            std = c.suggested_standard or "-"
            p(f"| `{c.name}` | {c.suggested_concept or '-'} | {std} | {c.inferred_confidence} | {c.inferred_reason} |")
        p("")

    if unmatched:
        p(f"### Sin match — REQUIEREN CONTEXTO HUMANO ({len(unmatched)})")
        p("")
        p("| Columna | Tipo | Valores muestra | Sugerencia |")
        p("|---|---|---|---|")
        for c in unmatched:
            samples = ", ".join(c.sample_values[:5])
            suggestion = c.inferred_reason if c.inferred_reason != "Sin inferencia posible" else "Necesita definicion humana"
            p(f"| `{c.name}` | {c.data_type} | {samples} | {suggestion} |")
        p("")

    # === PROBLEMAS DETECTADOS ===
    cols_with_issues = [c for c in report.columns if c.issues]
    if cols_with_issues:
        p("## Problemas Detectados")
        p("")
        for c in cols_with_issues:
            p(f"### `{c.name}`")
            for issue in c.issues:
                p(f"- {issue}")
            if c.anomaly_count > 0:
                p(f"- Outliers: {c.anomaly_count} ({c.anomaly_ratio:.0%}) via IQR")
            p("")

    # === PII Y SENSIBLES ===
    pii_cols = [c for c in report.columns if c.is_pii]
    sens_cols = [c for c in report.columns if c.is_sensitive]
    if pii_cols or sens_cols:
        p("## Datos Personales y Sensibles")
        p("")
        if pii_cols:
            p(f"### PII ({len(pii_cols)})")
            p("")
            for c in pii_cols:
                p(f"- **`{c.name}`** — {c.sensitivity_reason}")
            p("")
        if sens_cols:
            p(f"### Sensibles ({len(sens_cols)})")
            p("")
            for c in sens_cols:
                p(f"- **`{c.name}`** — {c.sensitivity_reason}")
            p("")

    # === INTEROP POTENCIAL ===
    if report.interop_candidates:
        p("## Potencial de Interoperabilidad")
        p("")
        p("| Columna | Concepto | Fuente destino | Confianza |")
        p("|---|---|---|---|")
        for cand in report.interop_candidates:
            p(f"| `{cand['column']}` | {cand['concept']} | {cand['target_source']} | {cand['confidence']} |")
        p("")

    # === ACCION REQUERIDA ===
    if unmatched:
        p("## Accion Requerida")
        p("")
        p(f"Para completar el analisis se necesita contexto humano para **{len(unmatched)} variables**.")
        p("")
        p("Puede entregar la metadata de dos formas:")
        p("")
        p("1. **Documento/diccionario de datos** (PDF, Excel, Word) — lo procesamos con RAG")
        p("2. **Formulario variable por variable** — le enviamos un link interactivo")
        p("")
        p("### Variables que necesitan contexto:")
        p("")
        p("| # | Variable | Tipo | Valores muestra | Pregunta sugerida |")
        p("|---|---|---|---|---|")
        for i, c in enumerate(unmatched, 1):
            samples = ", ".join(c.sample_values[:5])
            question = f"Que significa '{c.name}'? Es {c.suggested_concept or '...'}?"
            p(f"| {i} | `{c.name}` | {c.data_type} | {samples} | {question} |")
        p("")
        p("---")
        p("")
        p("**Siguiente paso:** Entregue la metadata y ejecute `enriched-analysis` para el reporte completo.")
    else:
        p("## Accion Requerida")
        p("")
        p("Todas las columnas fueron matcheadas o inferidas. No se requiere contexto humano adicional.")
        p("")
        p("**Siguiente paso:** Ejecute `enriched-analysis` para el reporte completo con MoA.")

    return "\n".join(lines)


def format_report_plain(report: RapidAssessmentReport) -> str:
    """Formatear el reporte como texto plano para terminal."""
    lines = []
    p = lines.append

    p(f"RAPID ASSESSMENT — {report.source_name}")
    p(f"Filas: {report.total_rows:,} | Columnas: {report.total_columns}")
    p(f"Calidad global: {report.avg_quality_score:.0f}/100 (Grade {report.global_grade})")
    p("")

    p(f"Matching: {report.matched_count} matched | {report.inferred_count} inferred | {report.unmatched_count} sin match")
    p(f"PII: {report.pii_count} | Sensibles: {report.sensitive_count} | Con problemas: {report.issues_count}")
    p(f"Interop candidates: {len(report.interop_candidates)}")
    p("")

    p("CALIDAD POR COLUMNA:")
    for c in report.columns:
        status_icon = "OK" if c.match_status == "matched" else ("~" if c.match_status == "inferred" else "?")
        pii_icon = " [PII]" if c.is_pii else (" [SENS]" if c.is_sensitive else "")
        p(f"  [{status_icon}] {c.name:30s} {c.data_type:10s} {c.quality_grade} ({c.quality_score:3d}){pii_icon}")
        if c.issues:
            for issue in c.issues:
                p(f"       ! {issue}")
    p("")

    if report.unmatched_count > 0:
        p(f"ACCION REQUERIDA: {report.unmatched_count} variables necesitan contexto humano")
        for c in report.columns:
            if c.match_status == "unmatched":
                samples = ", ".join(c.sample_values[:5])
                p(f"  [?] {c.name:30s} valores: {samples}")
        p("")
        p("Entregue diccionario de datos o use formulario para completar metadata.")
        p("Luego ejecute: enriched-analysis <csv> --metadata <doc>")

    return "\n".join(lines)
