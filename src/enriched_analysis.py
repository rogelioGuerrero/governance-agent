"""
Enriched Analysis — Pass 2 del flujo de dos pasadas.

Toma el reporte de Rapid Assessment (Pass 1) y:
1. Si hay metadata humana (--metadata file.json), la usa para matchear variables sin match
2. Si NO hay metadata, usa el LLM para inferir metadata de las variables sin match
3. Re-ejecuta matching con la nueva metadata
4. Ejecuta MoA (juridico + tecnico + estadistico) sobre el dataset
5. Genera reporte completo con:
   - Resumen ejecutivo enriquecido
   - Matching completo (todas las variables resueltas)
   - Analisis MoA (interoperabilidad, calidad, sesgos)
   - Recomendaciones accionables
   - Linaje de transformaciones sugeridas

Uso:
    uv run python -m src.cli enriched-analysis <csv> [--metadata file.json] [--output report.md]
"""

import csv
import json
import os
import re
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from .rapid_assessment import (
    assess_csv,
    ColumnAssessment,
    RapidAssessmentReport,
    format_report_markdown,
    _check_sensitivity,
)
from .profiler import profile_csv
from .inference import infer_semantic_type
from .standards import detect_standard
from .graph.catalog import load_graph_cached, NomencladorGraph, clear_graph_cache
from .graph.schema import SourceNode, FieldNode, ConceptNode, DataClassification, ReviewStatus
from .rag_factory import detect_issues, clean_column_name, RawColumn
from .llm_client import call_groq
from .log_config import get_logger

log = get_logger("enriched")

NOMENCLADOR_PATH = str(Path(__file__).parent.parent / "nomenclador" / "nomenclador.json")


@dataclass
class EnrichedColumn:
    """Columna enriquecida con metadata humana o LLM-inferida."""
    name: str
    data_type: str
    sample_values: list[str] = field(default_factory=list)

    # From Pass 1
    pass1_status: str = "unmatched"  # matched, inferred, unmatched
    pass1_concept: Optional[str] = None

    # Enriched
    enriched_concept: Optional[str] = None
    enriched_description: Optional[str] = None
    enriched_type: Optional[str] = None  # pii, sensible, normal, identifier
    enriched_standard: Optional[str] = None
    enriched_confidence: str = "low"  # high (human), medium (llm), low (none)
    enrichment_source: str = "none"  # human, llm, none

    # Final status
    final_status: str = "unmatched"  # matched, inferred, unmatched
    final_concept: Optional[str] = None


@dataclass
class EnrichedReport:
    """Reporte completo del enriched analysis."""
    source_file: str
    source_name: str
    generated_at: str
    total_rows: int
    total_columns: int

    # Pass 1 summary
    pass1_grade: str = "F"
    pass1_matched: int = 0
    pass1_inferred: int = 0
    pass1_unmatched: int = 0

    # Enrichment
    enrichment_source: str = "none"  # human, llm, none
    enriched_count: int = 0
    still_unmatched: int = 0

    # Final matching
    final_matched: int = 0
    final_inferred: int = 0
    final_unmatched: int = 0

    # Columns
    columns: list[EnrichedColumn] = field(default_factory=list)

    # MoA analysis
    moa_analysis: Optional[str] = None
    moa_juridico: Optional[str] = None
    moa_tecnico: Optional[str] = None
    moa_estadistico: Optional[str] = None

    # Quality
    avg_quality_score: float = 0.0
    pii_count: int = 0
    sensitive_count: int = 0
    issues_count: int = 0

    # Interop
    interop_candidates: list[dict] = field(default_factory=list)

    # Pass 1 report (embedded)
    pass1_report: Optional[RapidAssessmentReport] = None


def _load_metadata(metadata_path: str) -> dict:
    """Cargar metadata humana desde archivo JSON."""
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _infer_metadata_via_llm(unmatched_columns: list[ColumnAssessment]) -> dict:
    """Usar LLM para inferir metadata de columnas sin match.

    Retorna dict: {column_name: {concept, description, type, standard}}
    """
    cols_info = []
    for c in unmatched_columns:
        samples = ", ".join(c.sample_values[:8])
        cols_info.append({
            "name": c.name,
            "data_type": c.data_type,
            "unique_count": c.unique_count,
            "sample_values": samples,
        })

    prompt = f"""Eres un experto en governance de datos. Analiza estas columnas de un dataset
que no pudieron ser matcheadas automaticamente a conceptos existentes.

Para cada columna, inferir:
- concept: nombre del concepto semantico (ej: nombre_completo, diagnostico_cie10, fecha_nacimiento)
- description: descripcion breve de que representa
- type: clasificacion (pii, sensible, normal, identifier)
- standard: estandar si aplica (ej: ISO_8601, CIE10, DUI), o null

Columnas a analizar:
{json.dumps(cols_info, indent=2, ensure_ascii=False)}

Responde SOLO con JSON valido, sin markdown:
{{"columns": [{{"name": "...", "concept": "...", "description": "...", "type": "...", "standard": "..."}}]}}"""

    messages = [
        {"role": "system", "content": "Eres un experto en governance de datos y nomencladores institucionales. Respondes SOLO con JSON valido."},
        {"role": "user", "content": prompt},
    ]

    try:
        response = call_groq(messages, temperature=0.2, max_tokens=2000, json_mode=True)
        if isinstance(response, dict):
            return response
        parsed = json.loads(response)
        return parsed
    except Exception as e:
        log.warning("LLM metadata inference fallo: %s", e)
        return {"columns": []}


def _enrich_columns(
    pass1_report: RapidAssessmentReport,
    metadata: Optional[dict] = None,
) -> list[EnrichedColumn]:
    """Enriquecer columnas sin match usando metadata humana o LLM."""
    enriched = []
    unmatched = [c for c in pass1_report.columns if c.match_status == "unmatched"]

    # Build metadata lookup
    meta_lookup = {}
    if metadata:
        vars_meta = metadata.get("variables", metadata.get("columns", {}))
        if isinstance(vars_meta, list):
            for v in vars_meta:
                meta_lookup[v.get("name", "").lower()] = v
        elif isinstance(vars_meta, dict):
            for k, v in vars_meta.items():
                meta_lookup[k.lower()] = v

    # LLM inference si no hay metadata humana
    llm_lookup = {}
    if not meta_lookup and unmatched:
        log.info("Sin metadata humana — inferiendo via LLM para %d columnas", len(unmatched))
        llm_result = _infer_metadata_via_llm(unmatched)
        for col_meta in llm_result.get("columns", []):
            llm_lookup[col_meta.get("name", "").lower()] = col_meta

    for c in pass1_report.columns:
        ec = EnrichedColumn(
            name=c.name,
            data_type=c.data_type,
            sample_values=c.sample_values,
            pass1_status=c.match_status,
            pass1_concept=c.suggested_concept or c.matched_concept_id,
        )

        # Si ya estaba matcheado/inferred en Pass 1
        if c.match_status in ("matched", "inferred"):
            ec.final_status = c.match_status
            ec.final_concept = c.suggested_concept or c.matched_concept_id
            ec.enrichment_source = "pass1"
            ec.enriched_confidence = c.inferred_confidence
        else:
            # Intentar enriquecer
            col_key = c.name.lower().strip()
            meta = meta_lookup.get(col_key) or llm_lookup.get(col_key)

            if meta:
                ec.enriched_concept = meta.get("concept")
                ec.enriched_description = meta.get("description")
                ec.enriched_type = meta.get("type", "normal")
                ec.enriched_standard = meta.get("standard")
                ec.enrichment_source = "human" if meta_lookup else "llm"
                ec.enriched_confidence = "high" if meta_lookup else "medium"
                ec.final_status = "inferred"
                ec.final_concept = ec.enriched_concept
            else:
                ec.final_status = "unmatched"
                ec.enrichment_source = "none"

        enriched.append(ec)

    return enriched


def _build_moa_query(report: EnrichedReport, pass1: RapidAssessmentReport) -> str:
    """Construir consulta para el MoA basada en el dataset analizado."""
    cols_summary = []
    for ec in report.columns:
        status_icon = "OK" if ec.final_status == "matched" else ("~" if ec.final_status == "inferred" else "?")
        concept = ec.final_concept or "?"
        sensitivity = ""
        for c in pass1.columns:
            if c.name == ec.name:
                if c.is_pii:
                    sensitivity = " [PII]"
                elif c.is_sensitive:
                    sensitivity = " [SENSIBLE]"
                break
        cols_summary.append(f"  - {ec.name} ({ec.data_type}) -> {concept} [{status_icon}]{sensitivity}")

    return f"""Analiza el dataset '{report.source_name}' ({report.total_rows} filas, {report.total_columns} columnas).

Columnas y matching:
{chr(10).join(cols_summary)}

Calidad global: {pass1.avg_quality_score:.0f}/100 (Grade {pass1.global_grade})
PII: {report.pii_count} | Sensibles: {report.sensitive_count} | Problemas: {report.issues_count}

Evalua:
1. Interoperabilidad: que columnas pueden puentear con otros datasets?
2. Calidad: hay problemas criticos que impiden usar los datos?
3. Sesgos: hay sesgos potenciales en muestreo o cobertura?
4. Recomendaciones: que transformaciones o limpieza se necesita?"""


def _run_moa_analysis(query: str) -> dict:
    """Ejecutar MoA para analisis del dataset."""
    try:
        from .moa_agent import run_moa
        result = run_moa(query, max_iterations=3)
        return {
            "analysis": result.get("final_answer", ""),
            "juridico": result.get("juridico", ""),
            "tecnico": result.get("tecnico", ""),
            "estadistico": result.get("estadistico", ""),
        }
    except Exception as e:
        log.warning("MoA fallo: %s", e)
        return {
            "analysis": f"MoA no disponible: {e}",
            "juridico": "",
            "tecnico": "",
            "estadistico": "",
        }


def format_enriched_report_markdown(report: EnrichedReport) -> str:
    """Formatear reporte enriquecido como markdown."""
    lines = []
    p = lines.append

    p(f"# Enriched Analysis — {report.source_name}")
    p("")
    p(f"**Archivo:** `{report.source_file}`  ")
    p(f"**Fecha:** {report.generated_at}  ")
    p(f"**Filas:** {report.total_rows:,} | **Columnas:** {report.total_columns}")
    p(f"**Enrichment:** {report.enrichment_source}")
    p("")

    # === RESUMEN ===
    p("## Resumen")
    p("")
    p("| Metrica | Pass 1 | Pass 2 |")
    p("|---|---|---|")
    p(f"| Matched | {report.pass1_matched} | {report.final_matched} |")
    p(f"| Inferred | {report.pass1_inferred} | {report.final_inferred} |")
    p(f"| Sin match | {report.pass1_unmatched} | {report.final_unmatched} |")
    p(f"| Calidad | {report.avg_quality_score:.0f}/100 ({report.pass1_grade}) | - |")
    p(f"| PII | {report.pii_count} | - |")
    p(f"| Sensibles | {report.sensitive_count} | - |")
    p("")

    # === MATCHING COMPLETO ===
    p("## Matching Completo")
    p("")
    p("| Columna | Tipo | Pass 1 | Pass 2 | Concepto | Fuente | Confianza |")
    p("|---|---|---|---|---|---|---|")
    for ec in report.columns:
        p1_icon = "OK" if ec.pass1_status == "matched" else ("~" if ec.pass1_status == "inferred" else "?")
        p2_icon = "OK" if ec.final_status == "matched" else ("~" if ec.final_status == "inferred" else "?")
        concept = ec.final_concept or "-"
        source = ec.enrichment_source
        conf = ec.enriched_confidence
        p(f"| `{ec.name}` | {ec.data_type} | {p1_icon} | {p2_icon} | {concept} | {source} | {conf} |")
    p("")

    # === VARIABLES AUN SIN MATCH ===
    still_unmatched = [ec for ec in report.columns if ec.final_status == "unmatched"]
    if still_unmatched:
        p(f"### Variables sin resolver ({len(still_unmatched)})")
        p("")
        p("Estas variables no pudieron ser matcheadas ni con metadata humana ni LLM.")
        p("Requieren definicion manual explicita.")
        p("")
        for ec in still_unmatched:
            samples = ", ".join(ec.sample_values[:5])
            p(f"- **`{ec.name}`** ({ec.data_type}): {samples}")
        p("")

    # === DESCRIPCIONES DE CONCEPTOS ===
    enriched_cols = [ec for ec in report.columns if ec.enriched_description]
    if enriched_cols:
        p("## Definiciones Inferidas")
        p("")
        p("| Variable | Concepto | Descripcion | Tipo | Estandar |")
        p("|---|---|---|---|---|")
        for ec in enriched_cols:
            p(f"| `{ec.name}` | {ec.enriched_concept or '-'} | {ec.enriched_description or '-'} | {ec.enriched_type or '-'} | {ec.enriched_standard or '-'} |")
        p("")

    # === ANALISIS MoA ===
    if report.moa_analysis:
        p("## Analisis Multi-Agente (MoA)")
        p("")
        p("### Sintesis")
        p("")
        p(report.moa_analysis)
        p("")

        if report.moa_juridico:
            p("<details>")
            p("<summary>Analisis Juridico</summary>")
            p("")
            p(report.moa_juridico)
            p("")
            p("</details>")
            p("")

        if report.moa_tecnico:
            p("<details>")
            p("<summary>Analisis Tecnico</summary>")
            p("")
            p(report.moa_tecnico)
            p("")
            p("</details>")
            p("")

        if report.moa_estadistico:
            p("<details>")
            p("<summary>Analisis Estadistico</summary>")
            p("")
            p(report.moa_estadistico)
            p("")
            p("</details>")
            p("")

    # === INTEROP ===
    if report.interop_candidates:
        p("## Potencial de Interoperabilidad")
        p("")
        p("| Columna | Concepto | Fuente destino | Confianza |")
        p("|---|---|---|---|")
        seen = set()
        for cand in report.interop_candidates:
            key = (cand["column"], cand["target_source"])
            if key in seen:
                continue
            seen.add(key)
            p(f"| `{cand['column']}` | {cand['concept']} | {cand['target_source']} | {cand['confidence']} |")
        p("")

    # === PASS 1 REPORT (embedded) ===
    if report.pass1_report:
        p("---")
        p("")
        p("## Rapid Assessment (Pass 1)")
        p("")
        p(format_report_markdown(report.pass1_report))

    return "\n".join(lines)


def format_enriched_report_plain(report: EnrichedReport) -> str:
    """Formatear reporte enriquecido como texto plano."""
    lines = []
    p = lines.append

    p(f"ENRICHED ANALYSIS — {report.source_name}")
    p(f"Filas: {report.total_rows} | Columnas: {report.total_columns}")
    p(f"Enrichment: {report.enrichment_source}")
    p("")

    p(f"Pass 1: {report.pass1_matched} matched | {report.pass1_inferred} inferred | {report.pass1_unmatched} sin match")
    p(f"Pass 2: {report.final_matched} matched | {report.final_inferred} inferred | {report.final_unmatched} sin match")
    p("")

    p("COLUMNAS:")
    for ec in report.columns:
        p1 = "OK" if ec.pass1_status == "matched" else ("~" if ec.pass1_status == "inferred" else "?")
        p2 = "OK" if ec.final_status == "matched" else ("~" if ec.final_status == "inferred" else "?")
        concept = ec.final_concept or "?"
        p(f"  [{p1}->{p2}] {ec.name:30s} -> {concept} ({ec.enrichment_source})")
    p("")

    if report.moa_analysis:
        p("ANALISIS MoA:")
        p(report.moa_analysis)
        p("")

    return "\n".join(lines)


def _persist_to_graph(report: EnrichedReport, pass1: RapidAssessmentReport):
    """Persistir resultados del enriched analysis al knowledge graph.

    Crea/actualiza:
    - SourceNode para el dataset
    - FieldNode para cada columna
    - ConceptNode para conceptos nuevos inferidos por LLM/humano
    - Aristas IMPLEMENTA (Field -> Concept) y PROVIENE_DE (Field -> Source)
    """
    try:
        g = load_graph_cached()
    except Exception as e:
        log.warning("No se pudo cargar el grafo para persistir: %s", e)
        return

    source_id = f"src:{report.source_name}"
    today = datetime.now().strftime("%Y-%m-%d")

    # SourceNode
    if source_id not in g.graph:
        g.add_source(SourceNode(
            id=source_id,
            name=report.source_name,
            description=f"Dataset analizado via enriched analysis ({report.total_rows} filas, {report.total_columns} columnas)",
            connection=report.source_file,
            last_verified=today,
            review_status=ReviewStatus.APPROVED.value,
        ))
    else:
        g.graph.nodes[source_id]["last_verified"] = today

    # Build pass1 column lookup for quality/PII data
    pass1_lookup = {c.name: c for c in pass1.columns}

    new_concepts = 0
    new_fields = 0

    for ec in report.columns:
        col_clean = clean_column_name(ec.name)
        field_id = f"fld:{report.source_name}:{col_clean}"

        p1 = pass1_lookup.get(ec.name, None)

        # FieldNode
        field_kwargs = dict(
            id=field_id,
            source_db=report.source_name,
            table=report.source_name,
            column=ec.name,
            data_type=ec.data_type,
            sample_values=ec.sample_values[:10],
            last_verified=today,
            review_status=ReviewStatus.APPROVED.value,
        )

        # Quality metrics from Pass 1
        if p1:
            if p1.is_pii:
                cls = DataClassification.PII.value
            elif p1.is_sensitive:
                cls = DataClassification.SENSIBLE.value
            else:
                cls = DataClassification.PUBLICO.value

            uniqueness = p1.unique_count / p1.total_count if p1.total_count > 0 else 0.0

            field_kwargs.update(
                data_classification=cls,
                quality_score=p1.quality_score / 100.0,
                completeness=p1.completeness,
                uniqueness=uniqueness,
                consistency=p1.consistency,
                validity=p1.validity,
            )

        if field_id not in g.graph:
            g.add_field(FieldNode(**field_kwargs))
            new_fields += 1
        else:
            # Update quality metrics on existing field
            for k, v in field_kwargs.items():
                if k != "id" and k != "type":
                    g.graph.nodes[field_id][k] = v
            g._db_upsert_node(field_id, dict(g.graph.nodes[field_id]))

        # Link field -> source
        if not g.graph.has_edge(field_id, source_id):
            g.link_fuente(field_id, source_id)

        # ConceptNode for new matches (LLM-inferred or human-provided)
        if ec.final_concept and ec.enrichment_source in ("llm", "human"):
            concept_id = f"cnc:{ec.final_concept}"

            if concept_id not in g.graph:
                review_status = ReviewStatus.PROPOSED.value if ec.enrichment_source == "llm" else ReviewStatus.APPROVED.value
                g.add_concept(ConceptNode(
                    id=concept_id,
                    name=ec.final_concept,
                    definition=ec.enriched_description or "",
                    standard=ec.enriched_standard,
                    data_classification=ec.enriched_type or "publico",
                    review_status=review_status,
                    proposed_by="enriched:llm" if ec.enrichment_source == "llm" else "enriched:human",
                ))
                new_concepts += 1

            # Link field -> concept (IMPLEMENTA)
            if not g.graph.has_edge(field_id, concept_id):
                g.link_implementa(field_id, concept_id)

    # Save (write-through to PostgreSQL + JSON backup)
    g.bump_version("patch", f"Enriched analysis: {report.source_name} (+{new_concepts} conceptos, +{new_fields} campos)")
    g.save(NOMENCLADOR_PATH)
    clear_graph_cache()

    log.info("Grafo persistido: +%d conceptos, +%d campos para %s", new_concepts, new_fields, report.source_name)


def run_enriched_analysis(
    csv_path: str,
    metadata_path: str = "",
    run_moa: bool = True,
) -> EnrichedReport:
    """Ejecutar enriched analysis completo.

    1. Run Pass 1 (rapid assessment)
    2. Load metadata (human or LLM-inferred)
    3. Re-match columns
    4. Run MoA analysis
    5. Generate report
    """
    log.info("Iniciando enriched analysis para %s", csv_path)

    # Pass 1
    pass1 = assess_csv(csv_path)

    # Load metadata
    metadata = None
    enrichment_source = "llm"
    if metadata_path and os.path.exists(metadata_path):
        metadata = _load_metadata(metadata_path)
        enrichment_source = "human"
        log.info("Metadata humana cargada desde %s", metadata_path)

    # Enrich columns
    enriched_cols = _enrich_columns(pass1, metadata)

    # Build report
    report = EnrichedReport(
        source_file=pass1.source_file,
        source_name=pass1.source_name,
        generated_at=datetime.now().isoformat(),
        total_rows=pass1.total_rows,
        total_columns=pass1.total_columns,
        pass1_grade=pass1.global_grade,
        pass1_matched=pass1.matched_count,
        pass1_inferred=pass1.inferred_count,
        pass1_unmatched=pass1.unmatched_count,
        enrichment_source=enrichment_source,
        columns=enriched_cols,
        avg_quality_score=pass1.avg_quality_score,
        pii_count=pass1.pii_count,
        sensitive_count=pass1.sensitive_count,
        issues_count=pass1.issues_count,
        interop_candidates=pass1.interop_candidates,
        pass1_report=pass1,
    )

    # Count final statuses
    report.final_matched = sum(1 for ec in enriched_cols if ec.final_status == "matched")
    report.final_inferred = sum(1 for ec in enriched_cols if ec.final_status == "inferred")
    report.final_unmatched = sum(1 for ec in enriched_cols if ec.final_status == "unmatched")
    report.enriched_count = report.final_inferred - pass1.inferred_count
    report.still_unmatched = report.final_unmatched

    # MoA analysis
    if run_moa and report.final_unmatched < report.total_columns:
        log.info("Ejecutando MoA para analisis del dataset...")
        moa_query = _build_moa_query(report, pass1)
        moa_result = _run_moa_analysis(moa_query)
        report.moa_analysis = moa_result["analysis"]
        report.moa_juridico = moa_result["juridico"]
        report.moa_tecnico = moa_result["tecnico"]
        report.moa_estadistico = moa_result["estadistico"]

    # Persistir al knowledge graph (Supabase + JSON backup)
    _persist_to_graph(report, pass1)

    return report
