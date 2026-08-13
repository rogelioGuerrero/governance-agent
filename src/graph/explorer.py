"""
Adapter para visualizar el nomenclador en Semantica Knowledge Explorer.

Exporta el NomencladorGraph (NetworkX) al formato JSON que el Explorer espera:
{
    "entities": [{"id": ..., "type": ..., "name": ..., ...props}],
    "relationships": [{"source": ..., "target": ..., "type": ..., ...props}]
}

Uso:
    python -m src.graph.explorer --output nomenclador_explorer.json
    semantica-explorer --graph nomenclador_explorer.json
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .catalog import NomencladorGraph, load_graph_cached

NOMENCLADOR_PATH = Path(__file__).parent.parent.parent / "nomenclador" / "nomenclador.json"

NODE_TYPE_LABELS = {
    "concept": "Concept",
    "field": "Field",
    "classifier": "Classifier",
    "operation": "Operation",
    "context": "Context",
    "source": "Source",
    "normative": "Normative",
    "anonymization": "Anonymization",
    "quality_issue": "QualityIssue",
    "insight": "Insight",
}

EDGE_TYPE_LABELS = {
    "implementa": "implementa",
    "usa_clasificador": "usa_clasificador",
    "transforma_a": "transforma_a",
    "pertenece_a": "pertenece_a",
    "proviene_de": "proviene_de",
    "compone": "compone",
    "deriva_de": "deriva_de",
    "respaldado_por": "respaldado_por",
    "aplica_anonimizacion": "aplica_anonimizacion",
    "equivalente_a": "equivalente_a",
    "subconcepto_de": "subconcepto_de",
    "tiene_issue": "tiene_issue",
    "tiene_contexto": "tiene_contexto",
    "generates_insight": "generates_insight",
}


def graph_to_explorer_json(graph: NomencladorGraph) -> dict:
    """Convertir NomencladorGraph al formato JSON de Semantica Explorer."""
    entities = []
    for node_id, data in graph.graph.nodes(data=True):
        raw_type = data.get("type", "concept")
        entity = {
            "id": node_id,
            "type": NODE_TYPE_LABELS.get(raw_type, raw_type),
            "name": data.get("name", node_id),
        }
        for key, value in data.items():
            if key not in ("id", "type", "name") and value is not None:
                if isinstance(value, (str, int, float, bool)):
                    entity[key] = value
                elif isinstance(value, list) and all(isinstance(v, str) for v in value):
                    entity[key] = ", ".join(value[:10])
        entities.append(entity)

    relationships = []
    for source, target, data in graph.graph.edges(data=True):
        raw_type = data.get("type", "relates_to")
        rel = {
            "source": source,
            "target": target,
            "type": EDGE_TYPE_LABELS.get(raw_type, raw_type),
        }
        for key, value in data.items():
            if key not in ("source", "target", "type") and value is not None:
                if isinstance(value, (str, int, float, bool)):
                    rel[key] = value
        relationships.append(rel)

    return {"entities": entities, "relationships": relationships}


def export_graph(output_path: str | None = None) -> Path:
    """Exportar el nomenclador a JSON para el Explorer.

    Returns: path al archivo generado.
    """
    graph = load_graph_cached()
    if graph.graph.number_of_nodes() == 0:
        if NOMENCLADOR_PATH.exists():
            graph.load(str(NOMENCLADOR_PATH))
        else:
            print(f"No se encontro nomenclador en {NOMENCLADOR_PATH}", file=sys.stderr)
            sys.exit(1)

    explorer_data = graph_to_explorer_json(graph)
    out = Path(output_path) if output_path else NOMENCLADOR_PATH.parent / "nomenclador_explorer.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(explorer_data, f, ensure_ascii=False, indent=2)

    n_nodes = len(explorer_data["entities"])
    n_edges = len(explorer_data["relationships"])
    print(f"Exportado: {out} ({n_nodes} nodos, {n_edges} aristas)")
    return out


def launch_explorer(port: int = 8000, host: str = "127.0.0.1", no_browser: bool = False):
    """Exportar el grafo y lanzar Nomenclador Explorer."""
    graph_path = export_graph()
    cmd = [
        sys.executable, "-m", "semantica.explorer",
        "--graph", str(graph_path),
        "--port", str(port),
        "--host", host,
    ]
    if no_browser:
        cmd.append("--no-browser")
    env = {**__import__("os").environ, "SEMANTICA_ALLOW_ANONYMOUS": "true"}
    print(f"Lanzando Explorer: {' '.join(cmd)}")
    subprocess.run(cmd, env=env)


def main():
    parser = argparse.ArgumentParser(
        description="Visualizar el nomenclador en Semantica Knowledge Explorer"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Ruta del JSON de salida (default: nomenclador/nomenclador_explorer.json)",
    )
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Lanzar semantica-explorer despues de exportar",
    )
    parser.add_argument("--port", "-p", type=int, default=8000, help="Puerto del Explorer")
    parser.add_argument("--host", default="127.0.0.1", help="Host del Explorer")
    parser.add_argument("--no-browser", action="store_true", help="No abrir browser automaticamente")
    args = parser.parse_args()

    if args.launch:
        launch_explorer(port=args.port, host=args.host, no_browser=args.no_browser)
    else:
        export_graph(args.output)


if __name__ == "__main__":
    main()
