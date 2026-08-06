"""Rebuild the nomenclador graph with ONLY real data (no demo sources)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
from dotenv import load_dotenv
load_dotenv()

from src.graph.catalog import NomencladorGraph

# Load existing graph, remove all nodes, save empty
NOMENCLADOR_PATH = str(Path(__file__).parent.parent / "nomenclador" / "nomenclador.json")

g = NomencladorGraph()
g.load(NOMENCLADOR_PATH)
print(f"Before: {g.graph.number_of_nodes()} nodes, {g.graph.number_of_edges()} edges")

# Clear in-memory graph
g.graph.clear()

# Clear PostgreSQL too
g.clear_db_graph()

g.save(NOMENCLADOR_PATH)
print(f"After clear: {g.graph.number_of_nodes()} nodes, {g.graph.number_of_edges()} edges")
print("Graph is now empty. Run profile on real CSVs to rebuild.")
