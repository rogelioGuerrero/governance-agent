"""
Servidor MCP del Nomenclador.

Expone el knowledge graph del nomenclador como tools MCP para que
IDEs (Cursor, Windsurf, VS Code) puedan consultar el nomenclador
en tiempo real mientras el desarrollador escribe codigo.

Tools expuestas:
- list_concepts: listar todos los conceptos canonicos
- search_variable: buscar una variable por nombre
- get_concept: obtener detalle completo de un concepto
- check_interoperability: validar interoperabilidad entre dos fuentes con guardrails
- get_transform: generar artefacto de transformacion SQL + JSON Schema
- validate_field: validar si un campo cumple con el estandar canonico
- get_classifier: obtener los valores validos de un clasificador
"""

import json

from mcp.server.fastmcp import FastMCP

from .graph.catalog import load_graph_cached
from .graph.schema import EdgeType
from .guardrails import validate_interoperability, CheckpointStatus
from .transformer import generate_transformation, artifact_to_dict
from .standards import STANDARDS, get_standard_values, detect_standard

mcp = FastMCP(
    "nomenclador",
    instructions=(
        "Servidor MCP del Nomenclador Institucional. "
        "Permite consultar el knowledge graph de variables canonicas, "
        "validar interoperabilidad entre fuentes con guardrails semanticos, "
        "y generar transformaciones SQL automaticas."
    ),
)


@mcp.tool()
def list_concepts() -> str:
    """
    Listar todos los conceptos canonicos del nomenclador.
    Retorna una tabla con: variable, estandar, definicion, fuentes.
    """
    g = load_graph_cached()
    concepts = g.list_concepts()
    if not concepts:
        return "Nomenclador vacio. Usa el CLI para perfilar fuentes primero."

    lines = []
    for c in concepts:
        fields = g.find_fields_of_concept(c["id"])
        sources = ", ".join(set(f.get("source_db", "?") for f in fields))
        lines.append(
            f"- {c.get('name', '?')} | "
            f"estandar: {c.get('standard', '-') or '-'} | "
            f"fuentes: {sources or '-'} | "
            f"def: {c.get('definition', '-') or '-'}"
        )
    return "\n".join(lines)


@mcp.tool()
def search_variable(name: str) -> str:
    """
    Buscar una variable en el nomenclador por nombre.
    Retorna el concepto canonico + todas las fuentes donde se encuentra.
    """
    g = load_graph_cached()
    concept = g.find_concept(name)
    if not concept:
        return f"Variable '{name}' no encontrada en el nomenclador."

    lines = []
    lines.append(f"Variable: {concept.get('name', '')}")
    lines.append(f"Estandar: {concept.get('standard', '-') or '-'}")
    lines.append(f"Definicion: {concept.get('definition', '-') or '-'}")
    lines.append(f"Poblacion: {concept.get('population', '-') or '-'}")
    lines.append(f"Captura: {concept.get('capture_method', '-') or '-'}")

    fields = g.find_fields_of_concept(concept["id"])
    if fields:
        lines.append("\nFuentes:")
        for f in fields:
            lines.append(
                f"  - {f.get('source_db', '?')}.{f.get('table', '?')}.{f.get('column', '?')} "
                f"| tipo: {f.get('data_type', '?')} | "
                f"valores: {', '.join(f.get('sample_values', [])[:5])}"
            )
    return "\n".join(lines)


@mcp.tool()
def get_concept(name: str) -> str:
    """
    Obtener el detalle completo de un concepto canonico incluyendo clasificador.
    """
    g = load_graph_cached()
    concept = g.find_concept(name)
    if not concept:
        return f"Concepto '{name}' no encontrado."

    lines = []
    lines.append(f"Concepto: {concept.get('name', '')}")
    lines.append(f"Estandar: {concept.get('standard', '-') or '-'}")
    lines.append(f"Definicion: {concept.get('definition', '-') or '-'}")
    lines.append(f"Poblacion: {concept.get('population', '-') or '-'}")
    lines.append(f"Metodo de captura: {concept.get('capture_method', '-') or '-'}")
    lines.append(f"Version: {concept.get('version', '1.0')}")

    # Buscar clasificador
    for cls_id in g.graph.successors(concept["id"]):
        edge = g.graph.get_edge_data(concept["id"], cls_id)
        if edge and edge.get("type") == EdgeType.USA_CLASIFICADOR.value:
            cls = g.graph.nodes[cls_id]
            lines.append(f"\nClasificador: {cls.get('name', '')}")
            values = cls.get("values", {})
            if values:
                lines.append("Valores validos:")
                for code, label in values.items():
                    lines.append(f"  {code} = {label}")
            break

    return "\n".join(lines)


@mcp.tool()
def check_interoperability(source_db: str, target_db: str) -> str:
    """
    Verificar interoperabilidad entre dos fuentes con guardrails de validacion.
    Ejecuta 4 checkpoints: Poblacion, Metodologia, Clasificador, Distribucion de datos.
    Retorna warnings de asimetria semantica si los checkpoints no coinciden.
    """
    g = load_graph_cached()
    results = g.find_interoperability_path(source_db, target_db)

    if not results:
        return f"No se encontraron caminos de interoperabilidad entre {source_db} y {target_db}."

    lines = [f"Interoperabilidad: {source_db} <-> {target_db}"]
    lines.append(f"{len(results)} camino(s) encontrado(s)\n")

    for i, result in enumerate(results, 1):
        field_a = result["field_a"]
        field_b = result["field_b"]
        concept = result["concept"]
        classifier = result.get("classifier")

        lines.append(f"Camino {i}: {concept.get('name', '?')}")
        lines.append(
            f"  {field_a.get('source_db', '')}.{field_a.get('column', '')} <-> "
            f"{field_b.get('source_db', '')}.{field_b.get('column', '')}"
        )

        validation = validate_interoperability(field_a, field_b, concept, classifier)

        for cp in validation.checkpoints:
            status_icon = {"match": "OK", "mismatch": "!!", "unknown": "??"}.get(cp.status.value, "??")
            lines.append(f"  [{status_icon}] {cp.name}: {cp.detail}")

        lines.append(f"  => {validation.recommendation}")
        if validation.warnings:
            for w in validation.warnings:
                lines.append(f"  WARNING: {w}")
        lines.append("")

    return "\n".join(lines)


@mcp.tool()
def get_transform(source_db: str, target_db: str) -> str:
    """
    Generar artefactos de transformacion (SQL CASE WHEN + JSON Schema)
    para conectar dos fuentes. El desarrollador solo revisa y aprueba.
    """
    g = load_graph_cached()
    results = g.find_interoperability_path(source_db, target_db)

    if not results:
        return f"No se encontraron caminos entre {source_db} y {target_db}."

    lines = [f"Transformaciones: {source_db} -> {target_db}\n"]

    for result in results:
        field_a = result["field_a"]
        field_b = result["field_b"]
        concept = result["concept"]
        classifier = result.get("classifier")

        validation = validate_interoperability(field_a, field_b, concept, classifier)
        artifact = generate_transformation(field_a, field_b, concept, classifier, validation)

        lines.append(f"=== {artifact.concept_name} ({artifact.standard}) ===")
        lines.append(f"  {field_a.get('source_db', '')}.{field_a.get('column', '')} -> {field_b.get('source_db', '')}.{field_b.get('column', '')}")
        lines.append(f"\nSQL:")
        lines.append(artifact.sql_transform)
        lines.append(f"\nJSON Schema:")
        lines.append(json.dumps(artifact.json_schema, ensure_ascii=False, indent=2))

        if validation.warnings:
            lines.append("\nWarnings:")
            for w in validation.warnings:
                lines.append(f"  {w}")
        lines.append("")

    return "\n".join(lines)


@mcp.tool()
def validate_field(column_name: str, sample_values: list[str]) -> str:
    """
    Validar si un campo cumple con el estandar canonico del nomenclador.
    Útil para cuando un desarrollador esta creando un nuevo campo y quiere
    saber si los valores que planea usar son canonicos.

    Retorna:
    - El estandar detectado
    - Si los valores son canonicos o necesitan transformacion
    - Los valores validos del estandar
    """
    candidates = detect_standard(column_name, sample_values)

    if not candidates:
        return f"No se detecto ningun estandar para '{column_name}' con valores {sample_values}."

    lines = [f"Validacion de campo: {column_name}"]
    lines.append(f"Valores de muestra: {sample_values}\n")

    for cand in candidates:
        std_id = cand["standard"]
        std = STANDARDS.get(std_id, {})
        canonical_values = get_standard_values(std_id)
        canonical_codes = set(canonical_values.keys())

        lines.append(f"Estandar detectado: {std_id} ({cand['name']})")
        lines.append(f"Confianza: {cand['confidence']}")
        lines.append(f"Razon: {cand['reason']}")

        if canonical_codes:
            samples_upper = set(str(v).strip().upper() for v in sample_values if v)
            non_canonical = samples_upper - canonical_codes
            if non_canonical:
                lines.append(f"\nVALORES NO CANONICOS: {non_canonical}")
                lines.append("Necesita transformacion. Valores validos:")
                for code, label in canonical_values.items():
                    lines.append(f"  {code} = {label}")
            else:
                lines.append("\nTodos los valores son canonicos. OK.")
        else:
            lines.append(f"\nEl estandar {std_id} no tiene valores enumerados (formato-based).")
        lines.append("")

    return "\n".join(lines)


@mcp.tool()
def get_classifier(standard_id: str) -> str:
    """
    Obtener los valores validos de un clasificador/estandar.
    Ejemplo: get_classifier("ISO_5218") retorna 0=desconocido, 1=masculino, etc.
    """
    std = STANDARDS.get(standard_id)
    if not std:
        return f"Estandar '{standard_id}' no encontrado. Estandares disponibles: {', '.join(STANDARDS.keys())}"

    lines = [f"Estandar: {std['name']}"]
    lines.append(f"Dominio: {std.get('domain', '-')}\n")

    values = std.get("values", {})
    if values:
        lines.append("Valores validos:")
        for code, label in values.items():
            lines.append(f"  {code} = {label}")
    else:
        lines.append("Este estandar no tiene valores enumerados (es de formato).")
        if "regex" in std:
            lines.append(f"Patron: {std['regex']}")

    return "\n".join(lines)


def main():
    mcp.run()


if __name__ == "__main__":
    main()
