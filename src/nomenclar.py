"""
Nomenclador: Flujo de dos rondas para descubrir y completar variables.

Ronda 1 - DESCUBRIMIENTO:
  - Perfila la fuente (CSV, SQL, JSON)
  - Detecta estandares para cada columna
  - Identifica que columnas mapean a conceptos existentes
  - Identifica que columnas son NUEVAS (sin mapeo)
  - Lista gaps: definiciones faltantes, estandares no detectados, custodios sin asignar
  - NO espera respuesta humana — registra lo que encuentra y lo que falta

Ronda 2 - COMPLETADO:
  - Toma los gaps de la ronda 1
  - Usa LLM (Groq) para proponer definiciones para conceptos nuevos
  - Usa LLM para sugerir estandares cuando las reglas no detectaron
  - Busca respaldo normativo automatico (RAG documental)
  - Integra todo al nomenclador
  - Reporta que se completo y que aun requiere revision humana

Comando CLI: nomenclar <file> [--auto]
"""

import json
import logging
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from .rag_factory import (
    extract_from_csv, extract_from_sql_ddl, extract_from_json_schema,
    clean_column_name, detect_issues, match_to_canonical, match_with_llm,
    _standard_to_concept, RawColumn, IngestionPlan,
)
from .standards import detect_standard, STANDARDS, list_standards
from .graph.catalog import NomencladorGraph, load_graph_cached, clear_graph_cache
from .graph.schema import ConceptNode, FieldNode, SourceNode, EdgeType
from .llm_client import call_groq
from .inference import infer_semantic_type

logger = logging.getLogger(__name__)


NOMENCLADOR_PATH = Path(__file__).parent.parent / "nomenclador" / "nomenclador.json"

# Gap A: Palabras clave para deteccion automatica de PII / datos sensibles
_PII_KEYWORDS = {
    "nombre", "apellido", "identificacion", "documento", "dni", "cedula",
    "direccion", "telefono", "email", "correo", "fecha_nacimiento",
    "numero_seguro", "numero_afiliacion", "pasaporte", "rut", "curp",
}
_SENSIBLE_KEYWORDS = {
    "diagnostico", "diagnostic", "enfermedad", "salud_mental", "vih",
    "discapacidad", "genetico", "genetica", "sexual", "reproductivo",
    "adiccion", "tratamiento", "medicamento", "internamiento",
    "fecha_ingreso", "alta_hospitalaria", "procedimiento",
}


def _infer_context(source_name: str) -> dict:
    """Inferir contexto de captura desde el nombre de la fuente.

    Delegado a context_rules para mantener reglas configurables y centralizadas.
    """
    from .context_rules import infer_context
    return infer_context(source_name)


def _detect_data_classification(column_name: str, data_type: str, sample_values: list) -> str:
    """Auto-detectar nivel de clasificacion de datos (Gap A).

    Returns: publico | interno | pii | sensible
    """
    name_lower = column_name.lower()

    for kw in _SENSIBLE_KEYWORDS:
        if kw in name_lower:
            return "sensible"

    for kw in _PII_KEYWORDS:
        if kw in name_lower:
            return "pii"

    return "publico"


@dataclass
class DiscoveryReport:
    """Reporte de la ronda 1: descubrimiento."""
    source_name: str
    total_columns: int = 0
    mapped: int = 0
    unmapped: int = 0
    new_concepts: list[dict] = field(default_factory=list)
    existing_mappings: list[dict] = field(default_factory=list)
    gaps: list[dict] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    cleanup: list[str] = field(default_factory=list)


@dataclass
class CompletionReport:
    """Reporte de la ronda 2: completado."""
    concepts_created: list[str] = field(default_factory=list)
    definitions_proposed: list[dict] = field(default_factory=list)
    standards_suggested: list[dict] = field(default_factory=list)
    normative_found: list[dict] = field(default_factory=list)
    still_gaps: list[dict] = field(default_factory=list)
    version_before: str = ""
    version_after: str = ""
    # Cobertura por método de resolución
    resolved_by_standard: int = 0
    resolved_by_inference_high: int = 0
    resolved_by_inference_medium: int = 0
    resolved_by_llm: int = 0
    unresolved: int = 0
    total_columns: int = 0


# === RONDA 1: DESCUBRIMIENTO ===

def discover(file_path: str, source_type: str = "csv") -> DiscoveryReport:
    """
    Ronda 1: Descubrir variables en una fuente, detectar gaps.
    NO espera respuesta humana. Solo registra lo que encuentra y lo que falta.
    """
    source_name = Path(file_path).stem

    # Extraer columnas
    if source_type == "sql":
        raw_columns = extract_from_sql_ddl(file_path)
    elif source_type == "json":
        raw_columns = extract_from_json_schema(file_path)
    else:
        raw_columns = extract_from_csv(file_path)

    if not raw_columns:
        return DiscoveryReport(source_name=source_name, issues=["No se pudieron extraer columnas"])

    # Cargar nomenclador existente (usar cache para reuso de conexion PostgreSQL)
    g = load_graph_cached()
    existing_concepts = g.list_concepts()

    report = DiscoveryReport(source_name=source_name, total_columns=len(raw_columns))

    # Limpiar + detectar issues + match
    for col in raw_columns:
        col.clean_name = clean_column_name(col.raw_name)
        if col.clean_name != col.raw_name:
            report.cleanup.append(f"'{col.raw_name}' -> '{col.clean_name}'")

        issues = detect_issues(col)
        report.issues.extend(issues)

        mapping = match_to_canonical(col, existing_concepts, graph=g)

        if mapping.get("proposed_concept"):
            report.mapped += 1
            report.existing_mappings.append({
                "column": col.clean_name,
                "concept": mapping["proposed_concept"],
                "standard": mapping.get("standard"),
                "confidence": mapping.get("confidence"),
                "method": mapping.get("method"),
            })
        else:
            report.unmapped += 1
            # Es un concepto potencialmente nuevo
            concept_name = col.clean_name

            # Detectar estandar aunque no tenga concepto
            std_candidates = detect_standard(col.clean_name, col.sample_values)
            detected_std = std_candidates[0]["standard"] if std_candidates else None

            # Inferencia semántica: patrones, listas de referencia, huella de valores
            inf_result = infer_semantic_type(
                col.clean_name, col.sample_values, col.unique_count, existing_concepts,
            )
            if inf_result.suggested_concept_name:
                concept_name = inf_result.suggested_concept_name
            if not detected_std and inf_result.suggested_standard_id:
                detected_std = inf_result.suggested_standard_id

            gap = {
                "column": col.clean_name,
                "raw_name": col.raw_name,
                "data_type": col.data_type,
                "sample_values": col.sample_values[:5],
                "suggested_concept": concept_name,
                "detected_standard": detected_std,
                "has_definition": False,
                "has_custodian": False,
                "has_normative": False,
                "confidence": inf_result.confidence if inf_result.confidence != "low" else mapping.get("confidence", "low"),
                "inference_reason": inf_result.reason,
            }
            report.gaps.append(gap)
            report.new_concepts.append(gap)

    return report


def format_discovery(report: DiscoveryReport) -> str:
    """Formatear el reporte de descubrimiento para mostrar."""
    lines = [
        f"\n=== RONDA 1: DESCUBRIMIENTO ===",
        f"Fuente: {report.source_name}",
        f"Columnas: {report.total_columns} | Mapeadas: {report.mapped} | Sin mapear: {report.unmapped}",
    ]

    if report.cleanup:
        lines.append(f"\nLimpieza ({len(report.cleanup)}):")
        for c in report.cleanup:
            lines.append(f"  {c}")

    if report.existing_mappings:
        lines.append(f"\nMapeos existentes ({len(report.existing_mappings)}):")
        for m in report.existing_mappings:
            lines.append(f"  OK {m['column']} -> {m['concept']} ({m['standard'] or '-'}) [{m['confidence']}]")

    if report.gaps:
        lines.append(f"\nGaps - conceptos nuevos/potenciales ({len(report.gaps)}):")
        for g in report.gaps:
            std = g.get("detected_standard") or "-"
            lines.append(f"  ?? {g['column']} (tipo: {g['data_type']}, std: {std})")
            lines.append(f"     valores: {', '.join(g['sample_values'][:3])}")

    if report.issues:
        lines.append(f"\nIssues ({len(report.issues)}):")
        for i in report.issues[:10]:
            lines.append(f"  ! {i}")

    lines.append(f"\n-> Ronda 2 completara {len(report.gaps)} gaps")
    return "\n".join(lines)


# === RONDA 2: COMPLETADO ===

def complete(report: DiscoveryReport, auto_confirm: bool = False) -> CompletionReport:
    """
    Ronda 2: Completar los gaps identificados en la ronda 1.
    Usa LLM para proponer definiciones, busca respaldo normativo,
    e integra todo al nomenclador.
    """
    cr = CompletionReport()

    g = load_graph_cached()
    cr.version_before = g.version

    # Registrar fuente
    source_id = f"source:{report.source_name}"
    if source_id not in g.graph:
        g.add_source(SourceNode(
            id=source_id,
            name=report.source_name,
            description=f"Descubierto via Nomenclador (2 rondas)",
        ))

    # Inferir contexto de la fuente para guardrails
    ctx = _infer_context(report.source_name)

    # 1. Mapeos existentes: registrar fields
    for m in report.existing_mappings:
        concept_id = f"concept:{m['concept']}"
        field_id = f"field:{report.source_name}.{m['column']}"
        if field_id not in g.graph:
            g.add_field(FieldNode(
                id=field_id,
                name=m["column"],
                source_db=report.source_name,
                column=m["column"],
                data_type="text",
                sample_values=[],
                population=ctx["population"],
                capture_method=ctx["capture_method"],
                context_label=ctx["context_label"],
            ))
            g.link_implementa(field_id, concept_id)

    # 2. Gaps: crear conceptos nuevos con definicion LLM
    gaps_to_complete = report.gaps

    if not gaps_to_complete:
        # Todo ya estaba mapeado
        g.bump_version("patch", f"Nomenclador: {report.source_name} (sin conceptos nuevos)")
        g.save(str(NOMENCLADOR_PATH))
        cr.version_after = g.version
        return cr

    # Separar gaps resueltos por inferencia (high) de los que necesitan LLM
    inference_resolved = [
        gap for gap in gaps_to_complete
        if gap.get("confidence") == "high" and gap.get("inference_reason")
    ]
    gaps_for_llm = [
        gap for gap in gaps_to_complete
        if not (gap.get("confidence") == "high" and gap.get("inference_reason"))
    ]

    if inference_resolved:
        logger.info("Inferencia high resolvió %d gaps sin LLM", len(inference_resolved))

    # Preparar batch para LLM: pedir definiciones solo para gaps no resueltos por inferencia
    llm_proposals = {}
    if gaps_for_llm:
        gap_descriptions = []
        for gap in gaps_for_llm:
            gap_descriptions.append(
                f"- {gap['suggested_concept']} (tipo: {gap['data_type']}, "
                f"valores: {', '.join(gap['sample_values'][:3])}, "
                f"estandar detectado: {gap.get('detected_standard') or 'ninguno'})"
            )

        # Construir lista de estandares registrados para contexto del LLM
        registered = list_standards()
        std_list = ", ".join(f"{s['id']}({s['name']})" for s in registered) if registered else "ninguno registrado"

        batch_prompt = (
            f"Para cada variable, propone una definicion breve (1-2 lineas) "
            f"y el estandar internacional mas probable.\n"
            f"Estandares registrados en el sistema: {std_list}\n"
            f"Si ninguno aplica, propone null en standard.\n\n"
            f"Variables:\n" + "\n".join(gap_descriptions) + "\n\n"
            f"Responde SOLO en JSON:\n"
            f'{{"variables": [{{"name": "nombre", "definition": "definicion", '
            f'"standard": "ID_estandar_o_null", "confidence": "low|medium|high"}}]}}'
        )

        # Intento 1: batch (todos los gaps en una sola llamada)
        try:
            response = call_groq(
                [{"role": "system", "content": "Eres un experto en estandares de datos e interoperabilidad semantica. El sistema es agnostico al dominio."},
                 {"role": "user", "content": batch_prompt}],
                temperature=0.2,
                max_tokens=4000,
                json_mode=True,
            )
            parsed = json.loads(response)
            for v in parsed.get("variables", []):
                llm_proposals[v["name"]] = v
        except Exception as e:
            logger.warning("LLM batch fallo, intentando individual: %s", e)
            # Intento 2: individual (un gap a la vez, prompt mas corto)
            for gap in gaps_for_llm:
                try:
                    single_prompt = (
                        f"Variable: {gap['suggested_concept']} "
                        f"(tipo: {gap['data_type']}, valores: {', '.join(gap['sample_values'][:3])}, "
                        f"estandar: {gap.get('detected_standard') or 'ninguno'})\n"
                        f"Estandares registrados: {std_list}\n"
                        f"Propone definicion breve (1 linea) y estandar.\n"
                        f'Reponde JSON: {{"name":"{gap["suggested_concept"]}","definition":"def","standard":"ID_o_null","confidence":"low|medium|high"}}'
                    )
                    response = call_groq(
                        [{"role": "system", "content": "Eres experto en estandares de datos e interoperabilidad semantica. El sistema es agnostico al dominio."},
                         {"role": "user", "content": single_prompt}],
                        temperature=0.2,
                        max_tokens=2000,
                        json_mode=True,
                    )
                    parsed = json.loads(response)
                    if "name" in parsed:
                        llm_proposals[parsed["name"]] = parsed
                    elif "variables" in parsed and parsed["variables"]:
                        llm_proposals[parsed["variables"][0]["name"]] = parsed["variables"][0]
                except Exception as e:
                    logger.warning("LLM individual fallo para %r: %s", gap['suggested_concept'], e)

    # 3. Crear conceptos nuevos + registrar fields
    for gap in gaps_to_complete:
        concept_name = gap["suggested_concept"]
        concept_id = f"concept:{concept_name}"

        # Propuesta del LLM o inferencia
        proposal = llm_proposals.get(concept_name, {})
        definition = proposal.get("definition", "")
        standard = proposal.get("standard") or gap.get("detected_standard")
        confidence = proposal.get("confidence", "low")

        # Si no hubo propuesta LLM pero la inferencia resolvió este gap
        if not proposal and gap.get("confidence") == "high" and gap.get("inference_reason"):
            definition = gap["inference_reason"]
            confidence = "high"

        # Gap A: auto-detectar PII/sensible
        data_cls = _detect_data_classification(concept_name, gap['data_type'], gap['sample_values'])

        # Crear concepto si no existe
        if concept_id not in g.graph:
            g.add_concept(ConceptNode(
                id=concept_id,
                name=concept_name,
                definition=definition or f"Variable descubierta de {report.source_name}",
                standard=standard or "",
                data_classification=data_cls,
                # Gap C: conceptos creados por IA quedan en proposed
                review_status="proposed",
                proposed_by="agent:nomenclar",
            ))
            # Persistir confianza de inferencia para batch-approve --confidence
            g.graph.nodes[concept_id]["confidence"] = gap.get("confidence", "low")
            if gap.get("inference_reason"):
                g.graph.nodes[concept_id]["inference_reason"] = gap["inference_reason"]
            cr.concepts_created.append(concept_name)

            if definition:
                cr.definitions_proposed.append({
                    "concept": concept_name,
                    "definition": definition,
                    "source": "llm",
                })

            if standard and standard != gap.get("detected_standard"):
                cr.standards_suggested.append({
                    "concept": concept_name,
                    "standard": standard,
                    "source": "llm",
                })

            # Lifecycle log
            try:
                from .lifecycle import log_event
                log_event(concept_id, "created", actor="agent",
                          reason=f"Descubierta via Nomenclador desde {report.source_name}",
                          details=f"standard={standard}, confidence={confidence}, method=nomenclar_r2")
            except Exception as e:
                logger.warning("Lifecycle log_event fallo: %s", e)

        # Registrar field
        field_id = f"field:{report.source_name}.{gap['column']}"
        if field_id not in g.graph:
            g.add_field(FieldNode(
                id=field_id,
                name=gap["column"],
                source_db=report.source_name,
                column=gap["column"],
                data_type=gap["data_type"],
                sample_values=gap["sample_values"],
                data_classification=data_cls,
                population=ctx["population"],
                capture_method=ctx["capture_method"],
                context_label=ctx["context_label"],
                review_status="proposed",
            ))
            g.link_implementa(field_id, concept_id)

        # Verificar si aun falta info
        still_missing = []
        if not definition:
            still_missing.append("definicion")
        if not standard:
            still_missing.append("estandar")

        # Buscar respaldo normativo
        try:
            from .normative_rag import NormativeRAG
            rag = NormativeRAG()
            if rag.corpus:
                results = rag.search(concept_name, top_k=1)
                if results and results[0]["score"] >= 0.65:
                    best = results[0]
                    g.link_normative(concept_id, best)
                    cr.normative_found.append({
                        "concept": concept_name,
                        "source": best.get("source", ""),
                        "score": best["score"],
                    })
                    # Lifecycle log
                    try:
                        from .lifecycle import log_event
                        log_event(concept_id, "normative_attached", actor="agent",
                                  reason=f"Respaldo normativo auto: {best.get('source', '')}",
                                  details=f"score={best['score']:.3f}")
                    except Exception as e:
                        logger.warning("Lifecycle log_event fallo: %s", e)
                else:
                    still_missing.append("normativa")
            else:
                still_missing.append("normativa")
        except Exception:
            still_missing.append("normativa")

        if still_missing:
            cr.still_gaps.append({
                "concept": concept_name,
                "missing": still_missing,
            })

    # Bump version + guardar
    if cr.concepts_created:
        g.bump_version("minor", f"Nomenclador: {report.source_name} ({len(cr.concepts_created)} conceptos nuevos)")
    else:
        g.bump_version("patch", f"Nomenclador: {report.source_name} (sin conceptos nuevos)")
    g.save(str(NOMENCLADOR_PATH))
    cr.version_after = g.version

    # Calcular cobertura por método de resolución
    cr.total_columns = report.total_columns
    # Mapeos en Ronda 1: distinguir por método
    inference_mapped = sum(
        1 for m in report.existing_mappings
        if (m.get("method") or "").startswith("inference_")
    )
    standard_mapped = report.mapped - inference_mapped
    cr.resolved_by_standard = standard_mapped
    cr.resolved_by_inference_high = inference_mapped + len(inference_resolved)
    cr.resolved_by_inference_medium = sum(
        1 for gap in gaps_for_llm
        if gap.get("confidence") == "medium" and gap.get("inference_reason")
    )
    cr.resolved_by_llm = sum(1 for gap in gaps_for_llm if llm_proposals.get(gap.get("suggested_concept", "")))
    cr.unresolved = len(cr.still_gaps)

    return cr


def format_completion(cr: CompletionReport) -> str:
    """Formatear el reporte de completado."""
    lines = [
        f"\n=== RONDA 2: COMPLETADO ===",
        f"Version: {cr.version_before} -> {cr.version_after}",
    ]

    if cr.concepts_created:
        lines.append(f"\nConceptos nuevos creados ({len(cr.concepts_created)}):")
        for c in cr.concepts_created:
            lines.append(f"  + {c}")

    if cr.definitions_proposed:
        lines.append(f"\nDefiniciones propuestas por LLM ({len(cr.definitions_proposed)}):")
        for d in cr.definitions_proposed:
            lines.append(f"  {d['concept']}: {d['definition'][:80]}")

    if cr.standards_suggested:
        lines.append(f"\nEstandares sugeridos por LLM ({len(cr.standards_suggested)}):")
        for s in cr.standards_suggested:
            lines.append(f"  {s['concept']} -> {s['standard']}")

    if cr.normative_found:
        lines.append(f"\nRespaldo normativo encontrado ({len(cr.normative_found)}):")
        for n in cr.normative_found:
            lines.append(f"  {n['concept']} <- {n['source']} (score: {n['score']:.3f})")

    if cr.still_gaps:
        lines.append(f"\nAun requiere atencion humana ({len(cr.still_gaps)}):")
        for g in cr.still_gaps:
            lines.append(f"  ?? {g['concept']}: falta {', '.join(g['missing'])}")
    else:
        lines.append(f"\n[green]Todo completado. Sin gaps restantes.[/green]")

    # Resumen de cobertura
    if cr.total_columns > 0:
        lines.append(f"\n[bold cyan]Resumen de cobertura:[/bold cyan]")
        total = cr.total_columns
        def _pct(n):
            return f"{n/total*100:.0f}%" if total else "0%"
        lines.append(f"  Estandares ISO detectados:  {cr.resolved_by_standard:>3}  ({_pct(cr.resolved_by_standard)})")
        lines.append(f"  Inferencia semantica high:  {cr.resolved_by_inference_high:>3}  ({_pct(cr.resolved_by_inference_high)})")
        lines.append(f"  Inferencia semantica med:   {cr.resolved_by_inference_medium:>3}  ({_pct(cr.resolved_by_inference_medium)})")
        lines.append(f"  LLM (Ronda 2):              {cr.resolved_by_llm:>3}  ({_pct(cr.resolved_by_llm)})")
        lines.append(f"  Sin mapear:                 {cr.unresolved:>3}  ({_pct(cr.unresolved)})")
        auto_pct = _pct(cr.resolved_by_standard + cr.resolved_by_inference_high)
        lines.append(f"  [green]Auto-resueltos (sin LLM): {auto_pct}[/green]")

    return "\n".join(lines)


# === PUNTO DE ENTRADA ===

def run_nomenclar(file_path: str, source_type: str = "csv", auto_confirm: bool = False) -> str:
    """
    Ejecutar el flujo completo de dos rondas:
    1. Descubrimiento: perfilar, detectar, identificar gaps
    2. Completado: LLM + normative RAG + integrar al nomenclador
    """
    # Ronda 1
    discovery = discover(file_path, source_type)
    discovery_text = format_discovery(discovery)

    # Ronda 2
    completion = complete(discovery, auto_confirm=auto_confirm)
    completion_text = format_completion(completion)

    return f"{discovery_text}\n\n{completion_text}"
