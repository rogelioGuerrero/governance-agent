"""
CLI interactivo del agente de governance.

Human-in-the-loop: el agente propone, el humano confirma.

Uso:
    uv run python -m src.cli profile <archivo.csv> [--auto]
    uv run python -m src.cli catalog
    uv run python -m src.cli search <variable>
    uv run python -m src.cli interop <db1> <db2>
"""

import os
import sys
import json
import logging
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.table import Table as RichTable
from rich.panel import Panel
from rich.prompt import Prompt, Confirm

from .graph.catalog import NomencladorGraph
from .graph.schema import (
    ConceptNode, FieldNode, ClassifierNode,
    ContextNode, SourceNode, EdgeType,
)
from .profiler import profile_csv, detect_standards_for_columns
from .standards import STANDARDS, register_standard, list_standards, import_catalog, unregister_standard
from .guardrails import validate_interoperability, CheckpointStatus
from .transformer import generate_transformation, artifact_to_dict
from .rag_factory import create_ingestion_plan, execute_ingestion_plan, plan_to_dict
from .rag_factory import detect_issues, clean_column_name, RawColumn
from .inference import infer_semantic_type
from .graph.catalog import clear_graph_cache
from .health import check_health, fix_orphan_nodes, retry_stuck_proposals, format_health_report, log_health_run

logger = logging.getLogger(__name__)
console = Console()
NOMENCLADOR_PATH = Path(__file__).parent.parent / "nomenclador" / "nomenclador.json"


def load_graph() -> NomencladorGraph:
    g = NomencladorGraph()
    if NOMENCLADOR_PATH.exists():
        g.load(str(NOMENCLADOR_PATH))
    return g


def save_graph(g: NomencladorGraph):
    g.save(str(NOMENCLADOR_PATH))
    clear_graph_cache()
    console.print(f"[green]Nomenclador guardado en {NOMENCLADOR_PATH}[/green]")


def cmd_profile(csv_path: str, auto: bool = False):
    """Perfilar un CSV y construir entradas del nomenclador con human-in-the-loop."""
    console.print(f"\n[bold cyan]Perfilando: {csv_path}[/bold cyan]\n")

    tables = profile_csv(csv_path)
    g = load_graph()

    # Registrar fuente
    source_name = Path(csv_path).stem
    source_id = f"source:{source_name}"
    g.add_source(SourceNode(id=source_id, name=source_name, description=f"CSV: {csv_path}"))

    for table in tables:
        # === SECCIÓN 1: OVERVIEW + ALERTAS ===
        console.print(f"[bold]Tabla: {table.name}[/bold] ({table.row_count} filas, {len(table.columns)} columnas)\n")

        detect_standards_for_columns(table)

        # Recopilar alertas de todas las columnas
        all_alerts = []
        col_inferences = {}
        for col in table.columns:
            raw_col = RawColumn(
                raw_name=col.column,
                clean_name=clean_column_name(col.column),
                data_type=col.data_type,
                null_count=col.null_count,
                total_count=col.total_count,
                unique_count=col.unique_count,
                sample_values=col.sample_values[:20],
            )
            issues = detect_issues(raw_col)
            for issue in issues:
                all_alerts.append((col.column, issue))

            # Inferencia semántica
            inf = infer_semantic_type(
                raw_col.clean_name, raw_col.sample_values, raw_col.unique_count, g.list_concepts(),
            )
            col_inferences[col.column] = inf

        if all_alerts:
            console.print(f"[bold red]Alertas ({len(all_alerts)}):[/bold red]")
            for col_name, alert in all_alerts:
                console.print(f"  [red]![/red] [yellow]{col_name}[/yellow]: {alert}")
            console.print()

        # === SECCIÓN 2: TABLA DE VARIABLES ===
        if table.columns:
            var_table = RichTable(title="Variables detectadas", show_lines=False)
            var_table.add_column("Columna", style="cyan")
            var_table.add_column("Tipo CSV", style="dim")
            var_table.add_column("Tipo inferido", style="green")
            var_table.add_column("Nulos", style="yellow")
            var_table.add_column("Únicos", style="white")
            var_table.add_column("Estándar", style="magenta")

            for col in table.columns:
                inf = col_inferences.get(col.column)
                inferred_type = inf.semantic_type or inf.reference_match or "-"
                if inf.confidence == "high":
                    inferred_type = f"[green]{inferred_type}[/green]"
                elif inf.confidence == "medium":
                    inferred_type = f"[yellow]{inferred_type}[/yellow]"

                std = col.inferred_standard
                std_str = std["name"] if std else ("-" if not inf.suggested_standard_id else inf.suggested_standard_id)

                null_pct = f"{col.null_count}/{col.total_count}" if col.total_count else "-"
                var_table.add_row(
                    col.column,
                    col.data_type,
                    inferred_type,
                    null_pct,
                    str(col.unique_count),
                    std_str,
                )
            console.print(var_table)
            console.print()

        # === SECCIÓN 3: MAPEO POR COLUMNA ===
        for col in table.columns:
            inf = col_inferences.get(col.column)
            console.print(f"  [yellow]Columna:[/yellow] {col.column}")
            console.print(f"    Tipo CSV: {col.data_type} | Tipo inferido: {inf.semantic_type or inf.reference_match or 'sin inferencia'} ({inf.confidence})")
            if inf.reason and inf.confidence != "low":
                console.print(f"    [dim]Razón inferencia: {inf.reason}[/dim]")
            console.print(f"    Nulos: {col.null_count}/{col.total_count}, Únicos: {col.unique_count}")
            if col.sample_values:
                console.print(f"    Muestras: {', '.join(col.sample_values[:10])}")

            if col.inferred_standard:
                std = col.inferred_standard
                console.print(f"    [green]Estándar detectado:[/green] {std['name']} (confianza: {std['confidence']})")
                console.print(f"    Razón: {std['reason']}")

                if std["confidence"] == "high":
                    if auto or Confirm.ask(f"    ¿Confirmar mapeo a {std['standard']}?", default=True):
                        console.print(f"    [dim]Auto-confirmado[/dim]" if auto else "")
                        _register_field_with_concept(g, col, source_id, source_name, std["standard"])
                    else:
                        _manual_register(g, col, source_id, source_name)
                else:
                    if auto:
                        console.print(f"    [dim]Confianza media, auto-skip (no mapeado)[/dim]")
                    elif Confirm.ask(f"    ¿Mapear a {std['standard']} (confianza media)?", default=False):
                        _register_field_with_concept(g, col, source_id, source_name, std["standard"])
                    else:
                        _manual_register(g, col, source_id, source_name)
            elif inf.matched_concept or inf.suggested_concept_name:
                concept_name = inf.matched_concept or inf.suggested_concept_name
                conf_color = "green" if inf.confidence == "high" else "yellow"
                console.print(f"    [{conf_color}]Inferencia semántica:[/{conf_color}] {concept_name} (confianza: {inf.confidence})")
                if auto and inf.confidence == "high":
                    console.print(f"    [dim]Auto-confirmado por inferencia high[/dim]")
                    _register_field_with_concept(g, col, source_id, source_name, inf.suggested_standard_id or "")
                elif auto:
                    console.print(f"    [dim]Confianza {inf.confidence}, auto-skip[/dim]")
                    _auto_register_no_concept(g, col, source_id, source_name)
                elif Confirm.ask(f"    ¿Mapear a '{concept_name}' (inferencia {inf.confidence})?", default=inf.confidence == "high"):
                    _register_field_with_concept(g, col, source_id, source_name, inf.suggested_standard_id or "")
                else:
                    _manual_register(g, col, source_id, source_name)
            else:
                console.print(f"    [red]Sin estándar ni inferencia detectada[/red]")
                if auto:
                    console.print(f"    [dim]Auto: registrando sin concepto canónico[/dim]")
                    _auto_register_no_concept(g, col, source_id, source_name)
                else:
                    _manual_register(g, col, source_id, source_name)

            console.print()

    save_graph(g)
    _show_stats(g)


def _infer_context(source_name: str) -> dict:
    """Inferir contexto de captura basándose en el nombre de la fuente.

    Delegado a context_rules para mantener reglas configurables y centralizadas.
    """
    from .context_rules import infer_context
    return infer_context(source_name)


def _register_field_with_concept(g: NomencladorGraph, col, source_id: str, source_name: str, standard_id: str):
    """Registrar campo con concepto canonico basado en estandar."""
    std = STANDARDS.get(standard_id, {})

    # Determinar el concepto canonico dinamicamente desde name_hints
    std_type = std.get("standard_type", "classifier")
    if std_type == "format":
        # Estandares de formato (ej: ISO_8601) no mapean a un concepto unico
        # usar el nombre de la columna como concepto
        canonical_name = col.column.lower().strip()
    else:
        # Para clasificadores, usar el primer name_hint como concepto canonico
        hints = std.get("name_hints", [])
        if hints:
            canonical_name = hints[0].lower().strip().replace(" ", "_")
        else:
            canonical_name = col.column.lower().strip()

    concept_id = f"concept:{canonical_name}"

    # Si el concepto no existe, crearlo
    if concept_id not in g.graph:
        ctx = _infer_context(source_name)
        g.add_concept(ConceptNode(
            id=concept_id,
            name=canonical_name,
            definition=f"Variable canonica con estandar {std.get('name', standard_id)}",
            standard=standard_id,
            population=ctx["population"],
            capture_method=ctx["capture_method"],
        ))

        # Si el estándar tiene valores, crear clasificador
        if std.get("values"):
            classifier_id = f"classifier:{standard_id.lower()}"
            if classifier_id not in g.graph:
                g.add_classifier(ClassifierNode(
                    id=classifier_id,
                    name=std["name"],
                    standard=standard_id,
                    values=std["values"],
                ))
            g.link_clasificador(concept_id, classifier_id)

    # Registrar campo fisico
    ctx = _infer_context(source_name)
    field_id = f"field:{source_name}.{col.column}"
    g.add_field(FieldNode(
        id=field_id,
        source_db=source_name,
        table=col.table,
        column=col.column,
        data_type=col.data_type,
        nullable=col.nullable,
        unique_count=col.unique_count,
        null_count=col.null_count,
        total_count=col.total_count,
        sample_values=col.sample_values,
        inferred_standard=standard_id,
        confidence="high",
        population=ctx["population"],
        capture_method=ctx["capture_method"],
        context_label=ctx["context_label"],
    ))
    g.link_implementa(field_id, concept_id)
    g.link_fuente(field_id, source_id)


def _auto_register_no_concept(g: NomencladorGraph, col, source_id: str, source_name: str):
    """Registro automatico sin concepto canonico (para modo --auto)."""
    ctx = _infer_context(source_name)
    field_id = f"field:{source_name}.{col.column}"
    g.add_field(FieldNode(
        id=field_id,
        source_db=source_name,
        table=col.table,
        column=col.column,
        data_type=col.data_type,
        nullable=col.nullable,
        unique_count=col.unique_count,
        null_count=col.null_count,
        total_count=col.total_count,
        sample_values=col.sample_values,
        population=ctx["population"],
        capture_method=ctx["capture_method"],
        context_label=ctx["context_label"],
    ))
    g.link_fuente(field_id, source_id)


def _manual_register(g: NomencladorGraph, col, source_id: str, source_name: str):
    """Registro manual: el humano define el concepto."""
    concept_name = Prompt.ask(f"    Nombre canónico para '{col.column}'", default=col.column.lower().strip())
    concept_id = f"concept:{concept_name}"

    if concept_id not in g.graph:
        definition = Prompt.ask(f"    Definición de '{concept_name}'", default="")
        why = Prompt.ask(f"    Por qué existe (opcional)", default="")
        what_for = Prompt.ask(f"    Para qué sirve (opcional)", default="")
        custodian = Prompt.ask(f"    Custodio/responsable (opcional)", default="")
        department = Prompt.ask(f"    Departamento/dirección (opcional)", default="")

        g.add_concept(ConceptNode(
            id=concept_id,
            name=concept_name,
            definition=definition,
            why=why,
            what_for=what_for,
            custodian=custodian,
            custodian_department=department,
        ))

    # Registrar campo físico
    field_id = f"field:{source_name}.{col.column}"
    g.add_field(FieldNode(
        id=field_id,
        source_db=source_name,
        table=col.table,
        column=col.column,
        data_type=col.data_type,
        nullable=col.nullable,
        unique_count=col.unique_count,
        null_count=col.null_count,
        total_count=col.total_count,
        sample_values=col.sample_values,
    ))
    g.link_implementa(field_id, concept_id)
    g.link_fuente(field_id, source_id)


def cmd_catalog():
    """Mostrar el catálogo (nomenclador)."""
    g = load_graph()
    concepts = g.list_concepts()

    if not concepts:
        console.print("[yellow]Nomenclador vacío. Usa 'profile' para empezar.[/yellow]")
        return

    table = RichTable(title="Nomenclador - Conceptos Canónicos")
    table.add_column("Variable", style="cyan")
    table.add_column("Estándar", style="green")
    table.add_column("Definición", style="white")
    table.add_column("Custodio", style="magenta")
    table.add_column("Depto", style="blue")
    table.add_column("Estado", style="yellow")
    table.add_column("Fuentes", style="yellow")
    table.add_column("Normativa", style="dim")

    for c in concepts:
        fields = g.find_fields_of_concept(c["id"])
        sources = ", ".join(set(f.get("source_db", "?") for f in fields))
        normatives = g.find_normative_of_concept(c["id"])
        norm_ref = "; ".join(n.get("citation", "")[:60] for n in normatives) if normatives else "-"
        status = c.get("status", "activo")
        status_style = "green" if status == "activo" else "yellow" if status == "deprecado" else "red"
        table.add_row(
            c.get("name", ""),
            c.get("standard", "-") or "-",
            c.get("definition", "-") or "-",
            c.get("custodian", "-") or "-",
            c.get("custodian_department", "-") or "-",
            f"[{status_style}]{status}[/{status_style}]",
            sources or "-",
            norm_ref,
        )

    console.print(table)
    _show_stats(g)


def cmd_search(variable: str):
    """Buscar una variable en el nomenclador."""
    g = load_graph()
    concept = g.find_concept(variable)

    if not concept:
        console.print(f"[red]Variable '{variable}' no encontrada en el nomenclador.[/red]")
        return

    panel_content = []
    panel_content.append(f"[bold]Variable:[/bold] {concept.get('name', '')}")
    panel_content.append(f"[bold]Estándar:[/bold] {concept.get('standard', '-') or '-'}")
    panel_content.append(f"[bold]Definición:[/bold] {concept.get('definition', '-') or '-'}")
    panel_content.append(f"[bold]Por qué:[/bold] {concept.get('why', '-') or '-'}")
    panel_content.append(f"[bold]Para qué:[/bold] {concept.get('what_for', '-') or '-'}")
    panel_content.append(f"[bold]Normativa:[/bold] {concept.get('normative', '-') or '-'}")

    fields = g.find_fields_of_concept(concept["id"])
    if fields:
        panel_content.append("\n[bold]Fuentes donde se encuentra:[/bold]")
        for f in fields:
            panel_content.append(f"  - {f.get('source_db', '?')}.{f.get('table', '?')}.{f.get('column', '?')}")

    console.print(Panel("\n".join(panel_content), title=f"Catálogo: {variable}"))


def cmd_interop(db1: str, db2: str):
    """Verificar interoperabilidad entre dos fuentes con guardrails."""
    g = load_graph()
    results = g.find_interoperability_path(db1, db2)

    if not results:
        console.print(f"[red]No se encontraron caminos de interoperabilidad entre {db1} y {db2}.[/red]")
        return

    console.print(f"\n[bold green]Interoperabilidad: {db1} <-> {db2}[/bold green]")
    console.print(f"[dim]{len(results)} camino(s) encontrado(s)[/dim]\n")

    for i, result in enumerate(results, 1):
        field_a = result["field_a"]
        field_b = result["field_b"]
        concept = result["concept"]
        classifier = result.get("classifier")

        console.print(f"[bold cyan]Camino {i}: {concept.get('name', '?')}[/bold cyan]")
        console.print(f"  {field_a.get('source_db', '')}.{field_a.get('column', '')} <-> {field_b.get('source_db', '')}.{field_b.get('column', '')}")

        # === GUARDRAILS ===
        validation = validate_interoperability(field_a, field_b, concept, classifier)

        # Mostrar checkpoints
        for cp in validation.checkpoints:
            if cp.status == CheckpointStatus.MATCH:
                icon = "OK"
                color = "green"
            elif cp.status == CheckpointStatus.MISMATCH:
                icon = "!!"
                color = "red"
            else:
                icon = "??"
                color = "yellow"
            console.print(f"  [{color}]{icon}[/{color}] {cp.name}: {cp.detail}")

        # Mostrar recomendación
        if validation.is_safe:
            console.print(f"  [green]=> {validation.recommendation}[/green]")
        else:
            console.print(f"  [red]=> {validation.recommendation}[/red]")
            for w in validation.warnings:
                console.print(f"  [red]  {w}[/red]")

        console.print()

        # Guardar para uso posterior
        result["_validation"] = validation


def _show_stats(g: NomencladorGraph):
    stats = g.stats()
    console.print(f"\n[dim]Nodos: {stats['total_nodes']} | Aristas: {stats['total_edges']} | Versión: {stats['version']}[/dim]")
    for t, count in stats["by_type"].items():
        console.print(f"  [dim]{t}: {count}[/dim]")


def cmd_transform(db1: str, db2: str, output_dir: str = "transforms"):
    """Generar artefactos de transformación entre dos fuentes."""
    g = load_graph()
    results = g.find_interoperability_path(db1, db2)

    if not results:
        console.print(f"[red]No se encontraron caminos de interoperabilidad entre {db1} y {db2}.[/red]")
        return

    os.makedirs(output_dir, exist_ok=True)

    console.print(f"\n[bold cyan]Generando artefactos de transformación: {db1} -> {db2}[/bold cyan]\n")

    artifacts = []
    for result in results:
        field_a = result["field_a"]
        field_b = result["field_b"]
        concept = result["concept"]
        classifier = result.get("classifier")

        # Validar con guardrails
        validation = validate_interoperability(field_a, field_b, concept, classifier)

        # Generar artefacto
        artifact = generate_transformation(field_a, field_b, concept, classifier, validation)
        artifact_dict = artifact_to_dict(artifact)
        artifacts.append(artifact_dict)

        concept_name = concept.get("name", "unknown")
        console.print(f"[bold]Variable: {concept_name}[/bold] ({artifact.standard})")
        console.print(f"  {field_a.get('source_db', '')}.{field_a.get('column', '')} -> {field_b.get('source_db', '')}.{field_b.get('column', '')}")

        if validation.warnings:
            for w in validation.warnings:
                console.print(f"  [red]  {w}[/red]")

        # SQL
        sql_file = os.path.join(output_dir, f"{concept_name}_transform.sql")
        with open(sql_file, "w", encoding="utf-8") as f:
            f.write(artifact.sql_transform + "\n")
        console.print(f"  [green]SQL:[/green] {sql_file}")

        # JSON Schema
        schema_file = os.path.join(output_dir, f"{concept_name}_schema.json")
        with open(schema_file, "w", encoding="utf-8") as f:
            json.dump(artifact.json_schema, f, ensure_ascii=False, indent=2)
        console.print(f"  [green]Schema:[/green] {schema_file}")

        console.print()

    # Artefacto combinado
    combined_file = os.path.join(output_dir, f"{db1}_to_{db2}_full.json")
    with open(combined_file, "w", encoding="utf-8") as f:
        json.dump({
            "source_db": db1,
            "target_db": db2,
            "artifacts": artifacts,
        }, f, ensure_ascii=False, indent=2)
    console.print(f"[bold green]Artefacto completo:[/bold green] {combined_file}")


def cmd_ingest(file_path: str, auto: bool = False, use_llm: bool = False):
    """Ingerir un archivo sucio via RAG Factory."""
    from .rag_factory import create_ingestion_plan, execute_ingestion_plan, plan_to_dict

    console.print(f"\n[bold cyan]RAG Factory: Ingesta de '{file_path}'[/bold cyan]")

    # Detectar tipo de archivo
    ext = Path(file_path).suffix.lower()
    source_type = {".csv": "csv", ".sql": "sql", ".json": "json"}.get(ext, "csv")

    # Crear plan de ingesta
    console.print("[dim]Extrayendo y perfilando...[/dim]")
    plan = create_ingestion_plan(file_path, source_type=source_type, use_llm=use_llm)

    # Mostrar resumen
    console.print(f"\n[bold]Plan de Ingesta: {plan.source_name}[/bold]")
    console.print(f"  Tipo: {plan.source_type} | Confianza: {plan.confidence}")
    console.print(f"  Columnas: {len(plan.columns)} | Mapeadas: {plan.summary['mapped'] if hasattr(plan, 'summary') else sum(1 for m in plan.proposed_mappings if m.get('proposed_concept'))} | Sin mapear: {sum(1 for m in plan.proposed_mappings if not m.get('proposed_concept'))}")

    # Mostrar cleanup actions
    if plan.cleanup_actions:
        console.print(f"\n[bold yellow]Limpieza aplicada:[/bold yellow]")
        for action in plan.cleanup_actions[:10]:
            console.print(f"  {action}")

    # Mostrar issues
    if plan.issues_found:
        console.print(f"\n[bold red]Problemas detectados ({len(plan.issues_found)}):[/bold red]")
        for issue in plan.issues_found[:10]:
            console.print(f"  {issue}")

    # Mostrar mapeos propuestos
    console.print(f"\n[bold]Mapeos propuestos:[/bold]")
    table = RichTable(show_header=True, header_style="bold")
    table.add_column("Columna")
    table.add_column("Concepto")
    table.add_column("Estandar")
    table.add_column("Confianza")
    table.add_column("Metodo")

    for m in plan.proposed_mappings:
        conf_color = {"high": "green", "medium": "yellow", "low": "red"}.get(m.get("confidence", "low"), "white")
        table.add_row(
            m.get("column", "?"),
            m.get("proposed_concept") or "[dim]-[/dim]",
            m.get("standard") or "[dim]-[/dim]",
            f"[{conf_color}]{m.get('confidence', '?')}[/{conf_color}]",
            m.get("method", "?"),
        )
    console.print(table)

    # Aprobar e ingerir
    if plan.requires_human_review and not auto:
        if not Confirm.ask("\nAprobar este plan de ingesta?"):
            console.print("[yellow]Ingesta cancelada.[/yellow]")
            return
    elif auto:
        console.print("[dim]Auto-aprobado[/dim]")

    result = execute_ingestion_plan(plan, auto_confirm=auto)
    console.print(f"\n[bold green]{result}[/bold green]")

    # Resumen de cobertura
    total = len(plan.proposed_mappings)
    if total > 0:
        by_method = {}
        for m in plan.proposed_mappings:
            method = m.get("method", "unknown")
            by_method[method] = by_method.get(method, 0) + 1

        mapped = sum(1 for m in plan.proposed_mappings if m.get("proposed_concept"))
        unmapped = total - mapped

        console.print(f"\n[bold cyan]Resumen de cobertura:[/bold cyan]")
        console.print(f"  Total columnas:            {total:>3}")
        console.print(f"  Mapeadas:                  {mapped:>3}  ({mapped/total*100:.0f}%)")
        console.print(f"  Sin mapear:                {unmapped:>3}  ({unmapped/total*100:.0f}%)")
        console.print(f"  [dim]Desglose por método:[/dim]")
        for method, count in sorted(by_method.items(), key=lambda x: -x[1]):
            console.print(f"    {method:<30} {count:>3}  ({count/total*100:.0f}%)")


def cmd_nomenclar(file_path: str, auto: bool = False):
    """Ejecutar flujo de dos rondas: descubrimiento + completado."""
    from .nomenclar import run_nomenclar

    console.print(f"\n[bold cyan]Nomenclador — Flujo de 2 rondas[/bold cyan]")
    console.print(f"[dim]Archivo: {file_path}[/dim]\n")

    result = run_nomenclar(file_path, auto_confirm=auto)

    # Mostrar resultado con colores
    for line in result.split("\n"):
        if line.startswith("==="):
            console.print(f"\n[bold yellow]{line}[/bold yellow]")
        elif line.startswith("  OK"):
            console.print(f"[green]{line}[/green]")
        elif line.startswith("  ??"):
            console.print(f"[yellow]{line}[/yellow]")
        elif line.startswith("  +"):
            console.print(f"[green]{line}[/green]")
        elif line.startswith("  !"):
            console.print(f"[red]{line}[/red]")
        elif line.startswith("->"):
            console.print(f"\n[bold cyan]{line}[/bold cyan]")
        elif "[green]" in line:
            console.print(line.replace("[green]", "[green]").replace("[/green]", "[/green]"))
        else:
            console.print(line)


def cmd_assign(variable: str):
    """Asignar custodio y departamento a una variable existente."""
    g = load_graph()
    concept = g.find_concept(variable)

    if not concept:
        console.print(f"[red]Variable '{variable}' no encontrada en el nomenclador.[/red]")
        all_concepts = g.find_all_concepts()
        if all_concepts:
            console.print("[dim]Variables disponibles: " + ", ".join(c["name"] for c in all_concepts) + "[/dim]")
        return

    console.print(f"\n[bold cyan]Asignar custodio: {concept['name']}[/bold cyan]")
    console.print(f"[dim]Custodio actual: {concept.get('custodian', '-') or '-'}[/dim]")
    console.print(f"[dim]Departamento actual: {concept.get('custodian_department', '-') or '-'}[/dim]\n")

    custodian = Prompt.ask("Custodio/responsable", default=concept.get("custodian", ""))
    department = Prompt.ask("Departamento/direccion", default=concept.get("custodian_department", ""))
    contact = Prompt.ask("Contacto (email/tel, opcional)", default=concept.get("custodian_contact", ""))
    why = Prompt.ask("Por que existe esta variable", default=concept.get("why", ""))
    what_for = Prompt.ask("Para que sirve", default=concept.get("what_for", ""))

    g.graph.nodes[concept["id"]]["custodian"] = custodian
    g.graph.nodes[concept["id"]]["custodian_department"] = department
    g.graph.nodes[concept["id"]]["custodian_contact"] = contact
    g.graph.nodes[concept["id"]]["why"] = why
    g.graph.nodes[concept["id"]]["what_for"] = what_for

    nom_path = Path(__file__).parent.parent / "nomenclador" / "nomenclador.json"
    g.save(str(nom_path))
    clear_graph_cache()

    # Lifecycle log
    try:
        from .lifecycle import log_event
        log_event(concept["id"], "custodian_assigned", actor="human",
                  reason=f"Custodio: {custodian}, Depto: {department}",
                  details=f"why={why[:80]}, what_for={what_for[:80]}")
    except Exception as e:
        logger.warning("Lifecycle log_event fallo: %s", e)

    console.print(f"\n[green]OK[/green] Custodio asignado a '{concept['name']}'")


def cmd_review(variable: str, action: str = ""):
    """Gestionar el workflow de revision de conceptos propuestos por IA (Gap C)."""
    from .lifecycle import log_review_event

    g = load_graph()

    if not action:
        # Listar nodos pendientes de revision
        proposed = g.find_proposed_nodes()
        if not proposed:
            console.print("[green]No hay nodos pendientes de revision.[/green]")
            return
        console.print(f"\n[bold cyan]Nodos propuestos por IA ({len(proposed)}):[/bold cyan]\n")
        for n in proposed:
            name = n.get("name", n["id"])
            status = n.get("review_status", "?")
            proposed_by = n.get("proposed_by", "?")
            cls = n.get("data_classification", "publico")
            console.print(f"  [yellow]{status}[/yellow] {name}  [dim](por: {proposed_by}, cls: {cls})[/dim]")
        console.print(f"\n[dim]Usa: review <variable> approve|reject|start[/dim]")
        return

    concept = g.find_concept(variable)
    if not concept:
        console.print(f"[red]Variable '{variable}' no encontrada.[/red]")
        return

    concept_id = concept["id"]
    current = g.get_review_status(concept_id)
    console.print(f"[dim]Estado actual: {current}[/dim]")

    if action in ("approve", "approved", "ok"):
        g.approve_node(concept_id)
        log_review_event(concept_id, "approved", actor="human", reason="Aprobado por custodio")
        console.print(f"[green]APROBADO[/green] '{variable}' — ahora es activo en el nomenclador")
    elif action in ("reject", "rejected", "no"):
        g.reject_node(concept_id)
        log_review_event(concept_id, "rejected", actor="human", reason="Rechazado por custodio")
        console.print(f"[red]RECHAZADO[/red] '{variable}' — marcado como rejected")
    elif action in ("start", "under_review", "review"):
        g.set_review_status(concept_id, "under_review")
        log_review_event(concept_id, "under_review", actor="human", reason="En revision por custodio")
        console.print(f"[yellow]EN REVISION[/yellow] '{variable}'")
    else:
        console.print(f"[red]Accion invalida: {action}. Usa: approve|reject|start[/red]")
        return

    nom_path = Path(__file__).parent.parent / "nomenclador" / "nomenclador.json"
    g.save(str(nom_path))
    clear_graph_cache()


def cmd_batch_approve(standard_filter: str = "", dry_run: bool = False, confidence_filter: str = ""):
    """Aprobar multiples conceptos propuestos por IA en lote (mitiga review fatigue).
    
    Filtra conceptos proposed que tengan:
    - definicion no vacia
    - estandar asignado
    - opcionalmente filtrar por estandar especifico
    - opcionalmente filtrar por nivel de confianza (high|medium|low)
    
    Uso:
        batch-approve                          # aprobar todos los high-confidence
        batch-approve ISO_5218                 # solo los con estandar ISO_5218
        batch-approve --confidence medium      # solo inferencia medium
        batch-approve --dry-run                # solo listar, no aprobar
    """
    from .lifecycle import log_review_event

    g = load_graph()
    proposed = g.find_proposed_nodes()
    
    # Filtrar solo conceptos (no fields) que tengan definicion y estandar
    candidates = []
    for n in proposed:
        if n.get("type") != "concept":
            continue
        if n.get("review_status") not in ("proposed", "under_review"):
            continue
        if not n.get("definition", "").strip():
            continue
        if not n.get("standard", "").strip():
            continue
        if standard_filter and n.get("standard", "") != standard_filter:
            continue
        if confidence_filter:
            node_conf = n.get("confidence", "low")
            if confidence_filter == "high" and node_conf != "high":
                continue
            if confidence_filter == "medium" and node_conf not in ("high", "medium"):
                continue
            if confidence_filter == "low" and node_conf != "low":
                continue
        candidates.append(n)
    
    if not candidates:
        console.print("[yellow]No hay conceptos candidatos para batch approval.[/yellow]")
        console.print("[dim]Criterios: review_status=proposed/under_review + definicion + estandar[/dim]")
        return
    
    # Mostrar candidatos
    console.print(f"\n[bold cyan]Candidatos para batch approval ({len(candidates)}):[/bold cyan]\n")
    table = RichTable(show_header=True, header_style="bold")
    table.add_column("Variable")
    table.add_column("Estandar")
    table.add_column("Confianza")
    table.add_column("Clasificacion")
    table.add_column("Propuesto por")
    for c in candidates:
        conf = c.get("confidence", "low")
        conf_color = "green" if conf == "high" else "yellow" if conf == "medium" else "red"
        table.add_row(
            c.get("name", c["id"]),
            c.get("standard", "-"),
            f"[{conf_color}]{conf}[/{conf_color}]",
            c.get("data_classification", "publico"),
            c.get("proposed_by", "?"),
        )
    console.print(table)
    
    if dry_run:
        console.print(f"\n[dim]Dry run: {len(candidates)} conceptos serian aprobados. Use sin --dry-run para aprobar.[/dim]")
        return
    
    # Confirmar
    if not Confirm.ask(f"\nAprobar {len(candidates)} conceptos?"):
        console.print("[yellow]Cancelado.[/yellow]")
        return
    
    approved = 0
    for c in candidates:
        g.approve_node(c["id"])
        log_review_event(c["id"], "approved", actor="human", reason="Batch approval - high confidence")
        approved += 1
    
    save_graph(g)
    console.print(f"\n[green]{approved} conceptos aprobados en lote.[/green]")


def cmd_classify(variable: str, classification: str = ""):
    """Clasificar el nivel de sensibilidad de datos de una variable (Gap A)."""
    g = load_graph()
    concept = g.find_concept(variable)

    if not concept:
        console.print(f"[red]Variable '{variable}' no encontrada.[/red]")
        return

    if not classification:
        current = g.get_data_classification(concept["id"])
        console.print(f"[dim]Clasificacion actual: {current}[/dim]")
        console.print("[dim]Opciones: publico | interno | pii | sensible[/dim]")
        classification = Prompt.ask("Nueva clasificacion", default=current)

    valid = {"publico", "interno", "pii", "sensible"}
    if classification not in valid:
        console.print(f"[red]Clasificacion invalida: {classification}[/red]")
        console.print(f"[dim]Validas: {', '.join(valid)}[/dim]")
        return

    g.set_data_classification(concept["id"], classification)
    nom_path = Path(__file__).parent.parent / "nomenclador" / "nomenclador.json"
    g.save(str(nom_path))
    clear_graph_cache()

    # Lifecycle log
    try:
        from .lifecycle import log_event
        log_event(concept["id"], "data_classified", actor="human",
                  reason=f"Clasificacion: {classification}",
                  details=f"variable={variable}")
    except Exception as e:
        logger.warning("Lifecycle log_event fallo: %s", e)

    color = "red" if classification in ("pii", "sensible") else "green"
    console.print(f"[{color}]OK[/{color}] '{variable}' clasificada como [bold]{classification}[/bold]")


def cmd_sensitive():
    """Listar todos los datos PII o sensibles del nomenclador (Gap A)."""
    g = load_graph()
    sensitive = g.find_sensitive_data()

    if not sensitive:
        console.print("[green]No hay datos PII o sensibles registrados.[/green]")
        return

    console.print(f"\n[bold red]Datos sensibles / PII ({len(sensitive)}):[/bold red]\n")
    table = RichTable(title="Datos clasificados", show_lines=True)
    table.add_column("Nodo", style="cyan")
    table.add_column("Tipo", style="white")
    table.add_column("Clasificacion", style="red")
    table.add_column("Estandar", style="dim")

    for s in sensitive:
        node_type = s.get("type", "?")
        name = s.get("name", s["id"])
        cls = s.get("classification", "?")
        std = s.get("standard", "-") or "-"
        table.add_row(name, node_type, cls, std)

    console.print(table)


def cmd_history(variable: str):
    """Mostrar el historial (decision log) de una variable."""
    from .lifecycle import format_history, get_history

    g = load_graph()
    concept = g.find_concept(variable)

    if not concept:
        console.print(f"[red]Variable '{variable}' no encontrada.[/red]")
        return

    concept_id = concept["id"]
    history = get_history(concept_id)

    if not history:
        console.print(f"[yellow]Sin historial para '{concept['name']}'.[/yellow]")
        return

    console.print(f"\n[bold cyan]Decision Log: {concept['name']}[/bold cyan]")
    console.print(f"[dim]Estado actual: {concept.get('status', 'activo')}[/dim]\n")

    for i, e in enumerate(history, 1):
        ts = e["timestamp"][:16].replace("T", " ")
        actor_color = "magenta" if e["actor"] == "human" else "dim"
        console.print(f"  [bold]{i}.[/bold] [{actor_color}]{ts}[/{actor_color}] {e['action']}")
        if e.get("reason"):
            console.print(f"     [dim]Razon: {e['reason']}[/dim]")
        if e.get("details"):
            console.print(f"     [dim]Detalles: {e['details']}[/dim]")

    # Nota explicativa
    from .lifecycle import get_explanatory_note
    note = get_explanatory_note(concept_id)
    if note:
        console.print(f"\n[bold]Nota explicativa:[/bold]")
        console.print(f"[dim]{note}[/dim]")


def cmd_deprecate(variable: str, reason: str = "", replacement: str = ""):
    """Marcar una variable como deprecada."""
    from .lifecycle import change_status

    g = load_graph()
    concept = g.find_concept(variable)

    if not concept:
        console.print(f"[red]Variable '{variable}' no encontrada.[/red]")
        return

    if concept.get("status") == "deprecado":
        console.print(f"[yellow]'{concept['name']}' ya esta deprecada.[/yellow]")
        return

    # Analisis de impacto antes de deprecar
    impact = g.analyze_impact(concept["id"])
    if impact["total_impact"] > 0:
        console.print(f"\n[bold yellow]Analisis de impacto[/bold yellow]")
        console.print(f"  Fields afectados: {len(impact['fields'])}")
        console.print(f"  Rutas de interoperabilidad: {len(impact['interop_paths'])}")
        console.print(f"  Conceptos compuestos: {len(impact['composites'])}")
        console.print(f"  Conceptos derivados: {len(impact['derived_from'])}")
        console.print(f"  Clasificadores: {len(impact['classifiers'])}")
        console.print(f"  Normativas vinculadas: {len(impact['normatives'])}")
        console.print(f"  Operaciones de transformacion: {len(impact['transform_operations'])}")
        console.print(f"  [bold]Impacto total: {impact['total_impact']}[/bold]\n")
        
        if impact["fields"]:
            console.print("[dim]Fields afectados:[/dim]")
            for f in impact["fields"][:5]:
                console.print(f"  - {f.get('source_db', '?')}.{f.get('column', '?')}")
        
        if not Confirm.ask(f"Confirmar deprecacion de '{concept['name']}' (impacto: {impact['total_impact']})?", default=False):
            console.print("[yellow]Deprecacion cancelada.[/yellow]")
            return

    if not reason:
        reason = Prompt.ask("Por que se deprecia?", default="")
    if not replacement:
        replacement = Prompt.ask("Concepto que la reemplaza (opcional)", default="")

    g.graph.nodes[concept["id"]]["status"] = "deprecado"
    nom_path = Path(__file__).parent.parent / "nomenclador" / "nomenclador.json"
    g.save(str(nom_path))
    clear_graph_cache()

    msg = change_status(concept["id"], "deprecado", actor="human",
                        reason=reason, replacement=replacement)
    console.print(f"\n[yellow]DEPRECADA[/yellow] '{concept['name']}'")
    if reason:
        console.print(f"[dim]Razon: {reason}[/dim]")
    if replacement:
        console.print(f"[dim]Reemplazada por: {replacement}[/dim]")


def cmd_reactivate(variable: str, reason: str = ""):
    """Reactivar una variable deprecada o retirada."""
    from .lifecycle import change_status

    g = load_graph()
    concept = g.find_concept(variable)

    if not concept:
        console.print(f"[red]Variable '{variable}' no encontrada.[/red]")
        return

    if concept.get("status", "activo") == "activo":
        console.print(f"[yellow]'{concept['name']}' ya esta activa.[/yellow]")
        return

    if not reason:
        reason = Prompt.ask("Por que se reactiva?", default="")

    g.graph.nodes[concept["id"]]["status"] = "activo"
    nom_path = Path(__file__).parent.parent / "nomenclador" / "nomenclador.json"
    g.save(str(nom_path))
    clear_graph_cache()

    msg = change_status(concept["id"], "activo", actor="human", reason=reason)
    console.print(f"\n[green]REACTIVADA[/green] '{concept['name']}'")
    if reason:
        console.print(f"[dim]Razon: {reason}[/dim]")


def cmd_version(action: str = "info", reason: str = ""):
    """Gestionar versionado del nomenclador."""
    g = load_graph()

    if action == "info":
        info = g.version_info()
        console.print(f"\n[bold cyan]Nomenclador v{info['version']}[/bold cyan]")
        console.print(f"[dim]Cambios totales: {info['total_changes']}[/dim]\n")
        for h in info["history"]:
            console.print(f"  {h['from']} -> {h['to']} ({h['type']}) - {h['reason'] or 'sin razon'}")
            console.print(f"    [dim]{h['nodes']} nodos, {h['edges']} aristas[/dim]")
        if not info["history"]:
            console.print("  [dim]Sin cambios registrados[/dim]")

    elif action in ("major", "minor", "patch"):
        if not reason:
            reason = Prompt.ask("Razon del cambio de version", default="")
        old = g.version
        new = g.bump_version(action, reason)
        nom_path = Path(__file__).parent.parent / "nomenclador" / "nomenclador.json"
        g.save(str(nom_path))
        clear_graph_cache()
        console.print(f"\n[green]OK[/green] {old} -> {new} ({action})")
        if reason:
            console.print(f"[dim]Razon: {reason}[/dim]")

    else:
        console.print(f"[red]Accion no reconocida: {action}. Usar: info, major, minor, patch[/red]")


def cmd_compose(composite_name: str, parts: list[str] | None = None):
    """Crear una variable compuesta a partir de partes."""
    g = load_graph()
    composite_id = f"concept:{composite_name}"

    # Crear concepto compuesto si no existe
    if composite_id not in g.graph:
        definition = Prompt.ask(f"Definicion de '{composite_name}'", default="")
        g.add_concept(ConceptNode(
            id=composite_id,
            name=composite_name,
            definition=definition,
        ))
        console.print(f"[green]OK[/green] Concepto compuesto '{composite_name}' creado")

    # Si no hay partes, pedir interactivamente
    if not parts:
        all_concepts = g.find_all_concepts()
        available = [c["name"] for c in all_concepts if c["id"] != composite_id]
        console.print(f"\n[dim]Conceptos disponibles: {', '.join(available)}[/dim]\n")
        parts_input = Prompt.ask("Partes (separadas por coma)", default="")
        parts = [p.strip() for p in parts_input.split(",") if p.strip()]

    operation = Prompt.ask("Operacion", default="concat",
                           choices=["concat", "sum", "calc", "date_diff"])

    for part_name in parts:
        part_id = f"concept:{part_name}"
        if part_id not in g.graph:
            console.print(f"[yellow]Concepto '{part_name}' no existe. Creandolo...[/yellow]")
            part_def = Prompt.ask(f"Definicion de '{part_name}'", default="")
            g.add_concept(ConceptNode(
                id=part_id,
                name=part_name,
                definition=part_def,
            ))
        g.link_composite(composite_id, part_id, operation=operation)
        console.print(f"  [green]+[/green] {part_name} ({operation})")

    # Lifecycle log
    try:
        from .lifecycle import log_event
        log_event(composite_id, "composite_created", actor="human",
                  reason=f"Compuesta de: {', '.join(parts)} ({operation})")
    except Exception as e:
        logger.warning("Lifecycle log_event fallo: %s", e)

    g.bump_version("minor", f"Variable compuesta: {composite_name}")
    nom_path = Path(__file__).parent.parent / "nomenclador" / "nomenclador.json"
    g.save(str(nom_path))
    clear_graph_cache()
    console.print(f"\n[green]OK[/green] '{composite_name}' = {' + '.join(parts)} ({operation})")


def cmd_context(variable: str, source_db: str = "", meaning: str = ""):
    """Registrar significado contextual de una variable por fuente."""
    g = load_graph()
    concept = g.find_concept(variable)

    if not concept:
        console.print(f"[red]Variable '{variable}' no encontrada.[/red]")
        return

    if not source_db:
        source_db = Prompt.ask("Base de datos / fuente", default="")
    if not meaning:
        meaning = Prompt.ask("Significado en esa fuente", default="")
    context = Prompt.ask("Contexto de negocio (opcional)", default="")

    g.set_context_meaning(concept["id"], source_db, meaning, context)
    g.bump_version("patch", f"Contexto: {variable} en {source_db}")
    nom_path = Path(__file__).parent.parent / "nomenclador" / "nomenclador.json"
    g.save(str(nom_path))
    clear_graph_cache()

    # Lifecycle log
    try:
        from .lifecycle import log_event
        log_event(concept["id"], "context_added", actor="human",
                  reason=f"Contexto {source_db}: {meaning[:80]}")
    except Exception as e:
        logger.warning("Lifecycle log_event fallo: %s", e)

    console.print(f"\n[green]OK[/green] Contexto registrado para '{concept['name']}' en {source_db}")


def cmd_conflicts():
    """Detectar y mostrar conflictos de contexto."""
    g = load_graph()
    conflicts = g.find_context_conflicts()

    if not conflicts:
        console.print("[green]No hay conflictos de contexto detectados.[/green]")
        return

    console.print(f"\n[bold yellow]Conflictos de contexto ({len(conflicts)}):[/bold yellow]\n")
    for c in conflicts:
        console.print(f"  [bold]{c['concept_name']}[/bold]")
        for m in c["meanings"]:
            console.print(f"    [cyan]{m.get('source_db', '?')}[/cyan]: {m.get('description', '')}")
            if m.get("name"):
                console.print(f"    [dim]Contexto: {m['name']}[/dim]")
        console.print()


def cmd_normative(file_path: str, tags: list[str] | None = None):
    """Ingerir documento normativo al corpus RAG documental."""
    from .normative_rag import NormativeRAG

    rag = NormativeRAG()
    console.print(f"\n[bold cyan]RAG Documental[/bold cyan]")
    console.print(f"[dim]Archivo: {file_path}[/dim]")
    if tags:
        console.print(f"[dim]Tags: {', '.join(tags)}[/dim]")

    n = rag.ingest_file(file_path, tags=tags)
    if n > 0:
        console.print(f"[green]OK[/green] {n} chunks ingestados")
        s = rag.stats()
        console.print(f"[dim]Corpus: {s['total_chunks']} chunks | {s['total_sources']} fuentes[/dim]")
    else:
        console.print("[red]Error: no se pudieron extraer chunks[/red]")


def cmd_normative_search(query: str, top_k: int = 5):
    """Buscar en el corpus normativo."""
    from .normative_rag import NormativeRAG

    rag = NormativeRAG()
    if rag.stats()["total_chunks"] == 0:
        console.print("[yellow]Corpus normativo vacio. Usa 'normative <file>' para ingestar documentos.[/yellow]")
        return

    results = rag.search(query, top_k=top_k)
    if not results:
        console.print("[yellow]Sin resultados[/yellow]")
        return

    console.print(f"\n[bold cyan]Busqueda normativa:[/bold cyan] '{query}'\n")
    for i, r in enumerate(results, 1):
        score_color = "green" if r["score"] >= 0.65 else "yellow"
        console.print(f"  [{score_color}]{r['score']:.3f}[/{score_color}] [{r['source']}] {r['text'][:150]}...")


def cmd_agent(query: str):
    """Ejecutar el agente ReAct con Groq."""
    from .agent import run_agent

    console.print(f"\n[bold cyan]Agente ReAct[/bold cyan]")
    console.print(f"[dim]Consulta: {query}[/dim]\n")

    result = run_agent(query)

    # Mostrar scratchpad (razonamiento paso a paso)
    if result.get("scratchpad"):
        console.print("[bold dim]Razonamiento:[/bold dim]")
        for i, entry in enumerate(result["scratchpad"], 1):
            # Extraer THOUGHT, ACTION, OBSERVATION
            lines = entry.split("\n")
            for line in lines:
                if line.startswith("THOUGHT:"):
                    console.print(f"  [dim]Paso {i} - Pensamiento:[/dim] {line[9:][:120]}")
                elif line.startswith("ACTION:"):
                    console.print(f"  [yellow]Accion:[/yellow] {line[7:]}")
                elif line.startswith("OBSERVATION:"):
                    console.print(f"  [green]Observacion:[/green] {line[12:][:150]}")
        console.print()

    # Mostrar respuesta final
    if result.get("final_answer"):
        console.print(Panel(result["final_answer"], title="Respuesta", border_style="cyan"))
    elif result.get("needs_human_input"):
        console.print(f"[yellow]El agente necesita aclaracion:[/yellow] {result['needs_human_input']}")
    else:
        console.print("[yellow]El agente no pudo generar una respuesta final.[/yellow]")
        console.print(f"[dim]Iteraciones: {result.get('iterations', 0)}[/dim]")


def cmd_moa(query: str):
    """Ejecutar MoA multi-agente (juridico + tecnico + estadistico + sintetizador)."""
    from .moa_agent import run_moa

    console.print(f"\n[bold cyan]MoA Multi-Agente[/bold cyan]")
    console.print(f"[dim]Consulta: {query}[/dim]\n")

    result = run_moa(query)

    # Mostrar perspectivas individuales
    if result.get("juridico"):
        console.print(Panel(result["juridico"], title="Juridico", border_style="magenta"))
    if result.get("tecnico"):
        console.print(Panel(result["tecnico"], title="Tecnico", border_style="blue"))
    if result.get("estadistico"):
        console.print(Panel(result["estadistico"], title="Estadistico", border_style="green"))

    # Respuesta sintetizada
    if result.get("final_answer"):
        console.print(Panel(result["final_answer"], title="Sintesis", border_style="cyan"))

    console.print(f"\n[dim]MoA completado: 3 agentes + 1 sintetizador[/dim]")


def cmd_register_standard():
    """Registrar un estandar nuevo en el catalogo dinamico."""
    console.print("[bold]Registrar estandar nuevo[/bold]\n")
    std_id = Prompt.ask("ID del estandar", default="")
    if not std_id:
        console.print("[red]ID obligatorio[/red]")
        return
    name = Prompt.ask("Nombre descriptivo", default=std_id)
    domain = Prompt.ask("Dominio", default="transversal")
    std_type = Prompt.ask("Tipo", choices=["classifier", "format"], default="classifier")
    regex = Prompt.ask("Regex de validacion (opcional)", default="")
    hints_str = Prompt.ask("Name hints separados por coma (opcional)", default="")
    catalog_file = Prompt.ask("Archivo catalogo CSV/JSON (opcional, carga diferida)", default="")

    name_hints = [h.strip() for h in hints_str.split(",") if h.strip()] if hints_str else []

    register_standard(
        standard_id=std_id.upper().replace(" ", "_"),
        name=name,
        domain=domain,
        standard_type=std_type,
        regex=regex or None,
        name_hints=name_hints,
        catalog_file=catalog_file or None,
    )
    console.print(f"[green]Estandar '{std_id}' registrado.[/green]")
    console.print(f"[dim]Valores: usar 'import-catalog {std_id} <archivo>' para cargar catalogo de valores.[/dim]")


def cmd_list_standards():
    """Listar estandares registrados."""
    stds = list_standards()
    if not stds:
        console.print("[yellow]No hay estandares registrados.[/yellow]")
        console.print("[dim]Usa 'register-standard' para registrar uno nuevo.[/dim]")
        return

    table = RichTable(title="Estandares Registrados")
    table.add_column("ID", style="cyan")
    table.add_column("Nombre", style="white")
    table.add_column("Tipo", style="magenta")
    table.add_column("Dominio", style="green")
    table.add_column("Valores", style="yellow")
    table.add_column("Importable", style="dim")

    for s in stds:
        table.add_row(
            s["id"],
            s["name"],
            s.get("standard_type", "classifier"),
            s["domain"],
            str(s["values_count"]),
            "Si" if s["importable"] else "No",
        )
    console.print(table)


def cmd_import_catalog(standard_id: str, file_path: str = ""):
    """Importar catalogo de valores desde archivo CSV/JSON."""
    if not file_path:
        console.print(f"[red]Uso: import-catalog {standard_id} <archivo.csv|json>[/red]")
        return
    loaded = import_catalog(standard_id, file_path)
    if loaded > 0:
        console.print(f"[green]Catalogo cargado: {loaded} valores para {standard_id}[/green]")
    else:
        console.print(f"[red]No se pudo cargar catalogo desde {file_path}[/red]")


def cmd_impact(variable: str):
    """Analizar el impacto de cambiar un concepto."""
    g = load_graph()
    concept = g.find_concept(variable, include_proposed=True)
    if not concept:
        console.print(f"[red]Variable '{variable}' no encontrada.[/red]")
        return

    impact = g.analyze_impact(concept["id"])
    
    console.print(f"\n[bold cyan]Analisis de impacto: '{concept['name']}'[/bold cyan]\n")
    
    if impact.get("error"):
        console.print(f"[red]{impact['error']}[/red]")
        return

    if impact["total_impact"] == 0:
        console.print("[green]Sin dependencias. Este concepto es independiente.[/green]")
        return

    table = RichTable(title="Dependencias del concepto")
    table.add_column("Tipo", style="cyan")
    table.add_column("Cantidad", style="yellow")
    table.add_column("Detalle", style="white")
    
    if impact["fields"]:
        detail = ", ".join(f"{f.get('source_db','?')}.{f.get('column','?')}" for f in impact["fields"][:3])
        if len(impact["fields"]) > 3:
            detail += f" (+{len(impact['fields'])-3} mas)"
        table.add_row("Fields", str(len(impact["fields"])), detail)
    if impact["interop_paths"]:
        detail = ", ".join(f"{p['source_db']}->{p['target_db']}" for p in impact["interop_paths"][:3])
        table.add_row("Rutas interop", str(len(impact["interop_paths"])), detail)
    if impact["composites"]:
        detail = ", ".join(c["name"] for c in impact["composites"][:3])
        table.add_row("Compuestos", str(len(impact["composites"])), detail)
    if impact["derived_from"]:
        detail = ", ".join(d["name"] for d in impact["derived_from"][:3])
        table.add_row("Derivan de este", str(len(impact["derived_from"])), detail)
    if impact["derives_to"]:
        detail = ", ".join(d["name"] for d in impact["derives_to"][:3])
        table.add_row("Deriva de", str(len(impact["derives_to"])), detail)
    if impact["classifiers"]:
        table.add_row("Clasificadores", str(len(impact["classifiers"])), impact["classifiers"][0].get("id", "?"))
    if impact["normatives"]:
        detail = ", ".join(n.get("citation", "?")[:40] for n in impact["normatives"][:2])
        table.add_row("Normativas", str(len(impact["normatives"])), detail)
    if impact["transform_operations"]:
        table.add_row("Transformaciones", str(len(impact["transform_operations"])), "")
    
    console.print(table)
    console.print(f"\n[bold]Impacto total: {impact['total_impact']} dependencias[/bold]")


def cmd_demo_agri_env():
    """Demo: Interoperabilidad MAG <-> MARN con datos agroambientales."""
    import importlib.util
    demo_path = Path(__file__).parent.parent / "demo" / "run_demo.py"
    if not demo_path.exists():
        console.print(f"[red]Demo no encontrada: {demo_path}[/red]")
        return
    spec = importlib.util.spec_from_file_location("run_demo", demo_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.run_demo()


def cmd_health(do_fix: bool = False, do_retry: bool = False, do_heartbeat: bool = False, dry_run: bool = True):
    """Diagnostico y auto-healing del governance-agent."""
    console.print("\n[bold cyan]Governance Agent — Health Check[/bold cyan]\n")

    report = check_health()
    console.print(format_health_report(report))

    if not report["passed"]:
        console.print("\n[red]⚠  Health check FALLÓ — hay violaciones que requieren atención[/red]")
    else:
        console.print("\n[green]✓ Health check PASS[/green]")

    if do_fix:
        console.print("\n[bold cyan]Auto-healing: fix_orphans[/bold cyan]")
        if dry_run:
            console.print("[dim](dry-run — no se aplicarán cambios, usa --fix sin --dry-run para aplicar)[/dim]")
        result = fix_orphan_nodes(dry_run=dry_run)
        if result["fixes_suggested" if dry_run else "fixes_applied"]:
            for f in (result["fixes_suggested"] if dry_run else result["fixes_applied"]):
                console.print(f"  [yellow]{'→' if dry_run else '✓'}[/yellow] {f['node_name']} -> {f['suggested_concept']} (score={f['match_score']})")
        if result["manual_needed"]:
            console.print("\n[dim]Requiere intervención manual:[/dim]")
            for m in result["manual_needed"]:
                console.print(f"  [red]✗[/red] {m['node_name']}: {m['reason']}")
        if not result["fixes_suggested" if dry_run else "fixes_applied"] and not result["manual_needed"]:
            console.print("  [green]No hay nodos huérfanos para fix[/green]")

    if do_retry:
        console.print("\n[bold cyan]Auto-healing: retry_proposals[/bold cyan]")
        if dry_run:
            console.print("[dim](dry-run — no se aplicarán cambios)[/dim]")
        result = retry_stuck_proposals(dry_run=dry_run)
        if result["auto_approved"]:
            for a in result["auto_approved"]:
                console.print(f"  [yellow]{'→' if dry_run else '✓'}[/yellow] auto-approve: {a['node_name']} (qs={a['quality_score']}, {a['days_stale']}d)")
        if result["flagged_manual"]:
            console.print("\n[dim]Flaggeados para revisión manual:[/dim]")
            for f in result["flagged_manual"]:
                console.print(f"  [red]✗[/red] {f['node_name']} (qs={f['quality_score']}): {f['reason']}")
        if result["alerts"]:
            console.print("\n[red]Alertas:[/red]")
            for a in result["alerts"]:
                console.print(f"  ⚠ {a['node_name']}: {a['reason']}")
        if not result["auto_approved"] and not result["flagged_manual"] and not result["alerts"]:
            console.print("  [green]No hay proposals stale[/green]")

    if do_heartbeat:
        console.print("\n[bold cyan]Heartbeat: logueando a PostgreSQL...[/bold cyan]")
        ok = log_health_run(report)
        if ok:
            console.print("  [green]✓ Heartbeat logueado en governance.health_runs[/green]")
        else:
            console.print("  [yellow]⚠ No se pudo loguear heartbeat (PostgreSQL no disponible)[/yellow]")


def main():
    if len(sys.argv) < 2:
        console.print("[bold]Agente de Governance - Nomenclador Institucional[/bold]\n")
        console.print("Comandos:")
        console.print("  profile <csv>          Perfilar un CSV y construir el nomenclador")
        console.print("  ingest <file>          Ingerir archivo sucio via RAG Factory [--auto] [--llm]")
        console.print("  nomenclar <file>       Descubrir + completar variables en 2 rondas [--auto]")
        console.print("  catalog                Mostrar el catálogo completo")
        console.print("  search <var>           Buscar una variable en el nomenclador")
        console.print("  interop <db1> <db2>    Verificar interoperabilidad con guardrails")
        console.print("  transform <db1> <db2>  Generar artefactos SQL + JSON Schema")
        console.print("  normative <file>       Ingerir documento normativo (ley, reglamento) [--tag concepto]")
        console.print("  normative-search \"q\"  Buscar en corpus normativo")
        console.print("  assign <variable>      Asignar custodio/departamento a una variable")
        console.print("  history <variable>     Ver decision log y ciclo de vida de una variable")
        console.print("  deprecate <variable>   Marcar variable como deprecada [--reason ...]")
        console.print("  reactivate <variable>  Reactivar variable deprecada/retirada")
        console.print("  version [info|major|minor|patch]  Versionado del nomenclador")
        console.print("  compose <nombre>       Crear variable compuesta (ej: nombre_completo)")
        console.print("  context <variable>     Registrar significado contextual por fuente")
        console.print("  conflicts              Detectar conflictos de contexto")
        console.print("  review [variable] [approve|reject|start]  Gestionar conceptos propuestos por IA")
        console.print("  batch-approve [estandar] [--confidence high|medium|low] [--dry-run]  Aprobar en lote")
        console.print("  classify <variable> [publico|interno|pii|sensible]  Clasificar sensibilidad del dato")
        console.print("  sensitive              Listar todos los datos PII o sensibles")
        console.print("  agent \"consulta\"      Ejecutar agente ReAct con Groq (razonamiento)")
        console.print("  moa \"consulta\"        MoA: 3 agentes especializados (juridico, tecnico, estadistico)")
        console.print("  register-standard     Registrar un estandar nuevo en el catalogo dinamico")
        console.print("  list-standards        Listar estandares registrados")
        console.print("  import-catalog <id> <archivo>  Cargar valores de un estandar desde CSV/JSON")
        console.print("  impact <variable>     Analizar impacto de cambiar/deprecar un concepto")
        console.print("  demo-agri-env          Demo: interoperabilidad MAG <-> MARN")
        console.print("  health                 Diagnostico del governance-agent [--fix] [--retry] [--heartbeat]")
        return

    cmd = sys.argv[1]

    if cmd == "profile" and len(sys.argv) >= 3:
        auto = "--auto" in sys.argv
        cmd_profile(sys.argv[2], auto=auto)
    elif cmd == "catalog":
        cmd_catalog()
    elif cmd == "search" and len(sys.argv) >= 3:
        cmd_search(sys.argv[2])
    elif cmd == "interop" and len(sys.argv) >= 4:
        cmd_interop(sys.argv[2], sys.argv[3])
    elif cmd == "transform" and len(sys.argv) >= 4:
        cmd_transform(sys.argv[2], sys.argv[3])
    elif cmd == "ingest" and len(sys.argv) >= 3:
        auto = "--auto" in sys.argv
        use_llm = "--llm" in sys.argv
        cmd_ingest(sys.argv[2], auto=auto, use_llm=use_llm)
    elif cmd == "nomenclar" and len(sys.argv) >= 3:
        auto = "--auto" in sys.argv
        cmd_nomenclar(sys.argv[2], auto=auto)
    elif cmd == "agent":
        query = " ".join(sys.argv[2:]) if len(sys.argv) >= 3 else ""
        if not query:
            console.print("[red]Uso: agent \"consulta\"[/red]")
            console.print("[dim]Ej: agent \"¿Puedo cruzar el censo con el hospital?\"[/dim]")
            return
        cmd_agent(query)
    elif cmd == "normative" and len(sys.argv) >= 3:
        tags = []
        args = sys.argv[3:]
        i = 0
        while i < len(args):
            if args[i] == "--tag" and i + 1 < len(args):
                tags.append(args[i + 1])
                i += 2
            else:
                i += 1
        cmd_normative(sys.argv[2], tags=tags if tags else None)
    elif cmd == "normative-search" and len(sys.argv) >= 3:
        query = " ".join(sys.argv[2:])
        cmd_normative_search(query)
    elif cmd == "assign" and len(sys.argv) >= 3:
        cmd_assign(sys.argv[2])
    elif cmd == "history" and len(sys.argv) >= 3:
        cmd_history(sys.argv[2])
    elif cmd == "deprecate" and len(sys.argv) >= 3:
        reason = ""
        replacement = ""
        args = sys.argv[3:]
        i = 0
        while i < len(args):
            if args[i] == "--reason" and i + 1 < len(args):
                reason = args[i + 1]
                i += 2
            elif args[i] == "--replacement" and i + 1 < len(args):
                replacement = args[i + 1]
                i += 2
            else:
                i += 1
        cmd_deprecate(sys.argv[2], reason=reason, replacement=replacement)
    elif cmd == "reactivate" and len(sys.argv) >= 3:
        cmd_reactivate(sys.argv[2])
    elif cmd == "version":
        action = sys.argv[2] if len(sys.argv) >= 3 else "info"
        reason = ""
        if "--reason" in sys.argv:
            idx = sys.argv.index("--reason")
            if idx + 1 < len(sys.argv):
                reason = sys.argv[idx + 1]
        cmd_version(action, reason=reason)
    elif cmd == "compose" and len(sys.argv) >= 3:
        composite_name = sys.argv[2]
        parts = sys.argv[3:] if len(sys.argv) > 3 else None
        cmd_compose(composite_name, parts=parts)
    elif cmd == "context" and len(sys.argv) >= 3:
        variable = sys.argv[2]
        source_db = sys.argv[3] if len(sys.argv) >= 4 else ""
        meaning = sys.argv[4] if len(sys.argv) >= 5 else ""
        cmd_context(variable, source_db=source_db, meaning=meaning)
    elif cmd == "conflicts":
        cmd_conflicts()
    elif cmd == "review":
        variable = sys.argv[2] if len(sys.argv) >= 3 else ""
        action = sys.argv[3] if len(sys.argv) >= 4 else ""
        if variable:
            cmd_review(variable, action)
        else:
            cmd_review("", "")
    elif cmd == "batch-approve":
        dry_run = "--dry-run" in sys.argv
        confidence_filter = ""
        if "--confidence" in sys.argv:
            idx = sys.argv.index("--confidence")
            if idx + 1 < len(sys.argv):
                confidence_filter = sys.argv[idx + 1]
        args = [a for a in sys.argv[2:] if not a.startswith("--") and a != confidence_filter]
        standard_filter = args[0] if args else ""
        cmd_batch_approve(standard_filter=standard_filter, dry_run=dry_run, confidence_filter=confidence_filter)
    elif cmd == "classify" and len(sys.argv) >= 3:
        variable = sys.argv[2]
        classification = sys.argv[3] if len(sys.argv) >= 4 else ""
        cmd_classify(variable, classification)
    elif cmd == "sensitive":
        cmd_sensitive()
    elif cmd == "moa":
        query = " ".join(sys.argv[2:]) if len(sys.argv) >= 3 else ""
        if not query:
            console.print("[red]Uso: moa \"consulta\"[/red]")
            console.print("[dim]Ej: moa \"¿Puedo cruzar fecha_ingreso del hospital con el seguro?\"[/dim]")
            return
        cmd_moa(query)
    elif cmd == "register-standard":
        cmd_register_standard()
    elif cmd == "list-standards":
        cmd_list_standards()
    elif cmd == "import-catalog" and len(sys.argv) >= 4:
        cmd_import_catalog(sys.argv[2], sys.argv[3])
    elif cmd == "impact" and len(sys.argv) >= 3:
        cmd_impact(sys.argv[2])
    elif cmd == "health":
        do_fix = "--fix" in sys.argv
        do_retry = "--retry" in sys.argv
        do_heartbeat = "--heartbeat" in sys.argv
        dry_run = "--dry-run" in sys.argv
        cmd_health(do_fix=do_fix, do_retry=do_retry, do_heartbeat=do_heartbeat, dry_run=dry_run)
    else:
        console.print(f"[red]Comando no reconocido: {cmd}[/red]")


if __name__ == "__main__":
    main()
