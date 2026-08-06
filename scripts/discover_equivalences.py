"""
Discover semantic equivalences between two sources by comparing value sets.

This is the core interoperability test: two sources with different column names
but same underlying data. The script:
1. Samples values from each column of both CSVs
2. Computes overlap coefficient between value sets
3. Persists EQUIVALE_A edges in the graph for high-overlap pairs
4. Runs interop to verify paths are found
"""
import csv
import json
import sys
from pathlib import Path
from collections import Counter

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.graph.catalog import NomencladorGraph
from src.graph.schema import EdgeType

try:
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env")
except ImportError:
    pass

NOMENCLADOR_PATH = project_root / "nomenclador" / "nomenclador.json"


def load_csv_values(filepath: str, max_rows: int = 120) -> dict[str, list[str]]:
    """Load CSV and return dict of column_name -> list of string values."""
    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)[:max_rows]
    return {col: [row.get(col, "") for row in rows] for col in reader.fieldnames}


def overlap_coefficient(set_a: set, set_b: set) -> float:
    """Overlap coefficient = |A ∩ B| / min(|A|, |B|)."""
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    return len(intersection) / min(len(set_a), len(set_b))


def normalize_value(v: str) -> str:
    """Normalize a value for comparison: strip, lowercase, try round float."""
    v = str(v).strip().lower()
    if not v:
        return ""
    # Try to normalize floats (round to 1 decimal to handle rounding differences)
    try:
        return str(round(float(v), 1))
    except ValueError:
        return v


def discover_equivalences(csv_a: str, csv_b: str, threshold: float = 0.7) -> list[dict]:
    """Discover equivalent column pairs between two CSVs.
    
    Returns list of {col_a, col_b, overlap, matched_values, total_a, total_b}.
    """
    vals_a = load_csv_values(csv_a)
    vals_b = load_csv_values(csv_b)
    
    results = []
    for col_a_name, col_a_vals in vals_a.items():
        set_a = set(normalize_value(v) for v in col_a_vals if v)
        set_a.discard("")
        
        best_match = None
        best_score = 0.0
        
        for col_b_name, col_b_vals in vals_b.items():
            set_b = set(normalize_value(v) for v in col_b_vals if v)
            set_b.discard("")
            
            score = overlap_coefficient(set_a, set_b)
            if score > best_score:
                best_score = score
                best_match = col_b_name
        
        if best_match and best_score >= threshold:
            # Find the actual matched values
            vals_b_best = vals_b[best_match]
            set_b = set(normalize_value(v) for v in vals_b_best if v)
            set_b.discard("")
            matched = set_a & set_b
            
            results.append({
                "col_a": col_a_name,
                "col_b": best_match,
                "overlap": round(best_score, 3),
                "matched_count": len(matched),
                "total_a": len(set_a),
                "total_b": len(set_b),
            })
    
    return results


def persist_equivalences(g: NomencladorGraph, source_a: str, source_b: str, equivalences: list[dict]):
    """Persist EQUIVALE_A edges in the graph."""
    persisted = 0
    for eq in equivalences:
        field_a_id = f"field:{source_a}.{eq['col_a']}"
        field_b_id = f"field:{source_b}.{eq['col_b']}"
        
        # Check both fields exist
        if field_a_id not in g.graph or field_b_id not in g.graph:
            print(f"  SKIP: {field_a_id} or {field_b_id} not in graph")
            continue
        
        # Check if EQUIVALE_A edge already exists (in either direction)
        has_equiv = False
        if g.graph.has_edge(field_a_id, field_b_id):
            ed = g.graph.get_edge_data(field_a_id, field_b_id)
            if ed.get("type") == EdgeType.EQUIVALE_A.value:
                has_equiv = True
        if g.graph.has_edge(field_b_id, field_a_id):
            ed = g.graph.get_edge_data(field_b_id, field_a_id)
            if ed.get("type") == EdgeType.EQUIVALE_A.value:
                has_equiv = True
        if has_equiv:
            print(f"  EXISTS: {eq['col_a']} -> {eq['col_b']}")
            continue
        
        # Add EQUIVALE_A edge
        g.graph.add_edge(
            field_a_id, field_b_id,
            type=EdgeType.EQUIVALE_A.value,
            confidence=eq["overlap"],
            method="value_overlap",
            matched_values=eq["matched_count"],
        )
        # Write-through to PostgreSQL
        g._db_upsert_edge(field_a_id, field_b_id, EdgeType.EQUIVALE_A.value, {
            "confidence": eq["overlap"],
            "method": "value_overlap",
            "matched_values": eq["matched_count"],
        })
        
        # Also link the concepts
        # Find concepts for both fields
        concept_a = None
        concept_b = None
        for successor in g.graph.successors(field_a_id):
            edge_data = g.graph.get_edge_data(field_a_id, successor)
            if edge_data.get("type") == EdgeType.IMPLEMENTA.value:
                concept_a = successor
                break
        for successor in g.graph.successors(field_b_id):
            edge_data = g.graph.get_edge_data(field_b_id, successor)
            if edge_data.get("type") == EdgeType.IMPLEMENTA.value:
                concept_b = successor
                break
        
        if concept_a and concept_b and concept_a != concept_b:
            if not g.graph.has_edge(concept_a, concept_b):
                g.graph.add_edge(
                    concept_a, concept_b,
                    type=EdgeType.EQUIVALE_A.value,
                    confidence=eq["overlap"],
                    method="value_overlap_inferred",
                )
                g._db_upsert_edge(concept_a, concept_b, EdgeType.EQUIVALE_A.value, {
                    "confidence": eq["overlap"],
                    "method": "value_overlap_inferred",
                })
                print(f"  CONCEPT LINK: {concept_a} -> {concept_b}")
        
        print(f"  EQUIVALE_A: {eq['col_a']} <-> {eq['col_b']} (overlap={eq['overlap']}, matched={eq['matched_count']})")
        persisted += 1
    
    return persisted


def main():
    csv_a = str(project_root / "data" / "real" / "us_economic_indicators.csv")
    csv_b = str(project_root / "data" / "real" / "ministerio_economia_sv.csv")
    source_a = "us_economic_indicators"
    source_b = "ministerio_economia_sv"
    
    print("=" * 70)
    print("DESCUBRIMIENTO DE EQUIVALENCIAS SEMANTICAS")
    print(f"  Fuente A: {source_a}")
    print(f"  Fuente B: {source_b}")
    print("=" * 70)
    
    # Phase 1: Discover
    print("\n## Fase 1: Comparando valores de columnas...")
    equivalences = discover_equivalences(csv_a, csv_b, threshold=0.7)
    
    print(f"\n{len(equivalences)} equivalencias descubiertas:")
    print(f"{'Columna A':<30} {'Columna B':<30} {'Overlap':<10} {'Matched':<10}")
    print("-" * 80)
    for eq in equivalences:
        print(f"{eq['col_a']:<30} {eq['col_b']:<30} {eq['overlap']:<10} {eq['matched_count']}/{eq['total_a']}")
    
    # Phase 2: Persist
    print(f"\n## Fase 2: Persistiendo en grafo...")
    g = NomencladorGraph()
    g.load(str(NOMENCLADOR_PATH))
    
    persisted = persist_equivalences(g, source_a, source_b, equivalences)
    
    # Debug: verify edges in memory before save
    equiv_before = [(u, v, d) for u, v, d in g.graph.edges(data=True) if d.get("type") == "EQUIVALE_A"]
    print(f"\n  DEBUG: EQUIVALE_A edges in memory before save: {len(equiv_before)}")
    print(f"  DEBUG: _loaded_mtime = {getattr(g, '_loaded_mtime', 'NOT SET')}")
    
    g.save(str(NOMENCLADOR_PATH))
    
    # Debug: verify edges in memory after save
    equiv_after = [(u, v, d) for u, v, d in g.graph.edges(data=True) if d.get("type") == "EQUIVALE_A"]
    print(f"  DEBUG: EQUIVALE_A edges in memory after save: {len(equiv_after)}")
    print(f"\n{persisted} aristas EQUIVALE_A persistidas.")
    
    # Phase 3: Verify interop
    print(f"\n## Fase 3: Verificando interoperabilidad...")
    results = g.find_interoperability_path(source_a, source_b)
    
    if results:
        print(f"\n{len(results)} camino(s) de interoperabilidad encontrados:")
        for i, r in enumerate(results, 1):
            fa = r["field_a"]
            fb = r["field_b"]
            concept = r.get("concept") or {}
            match_type = r.get("match_type", "unknown")
            conf = r.get("confidence", "")
            conf_str = f" (conf={conf})" if conf else ""
            print(f"  Camino {i}: {concept.get('name', '?')} [{match_type}]{conf_str}")
            print(f"    {fa.get('source_db', '')}.{fa.get('column', '')} <-> {fb.get('source_db', '')}.{fb.get('column', '')}")
    else:
        print("\nNo se encontraron caminos. Los conceptos siguen separados.")
        
        # Fallback: check EQUIVALE_A edges directly
        equiv_edges = [
            (u, v, d) for u, v, d in g.graph.edges(data=True)
            if d.get("type") == EdgeType.EQUIVALE_A.value
        ]
        print(f"\nEQUIVALE_A edges en grafo: {len(equiv_edges)}")
        for u, v, d in equiv_edges:
            print(f"  {u} -> {v} (confidence={d.get('confidence', '?')})")
    
    # Summary
    print("\n" + "=" * 70)
    print("REPORTE DE INTEROPERABILIDAD")
    print("=" * 70)
    total_fields_a = len([n for n, d in g.graph.nodes(data=True) if d.get("type") == "field" and d.get("source_db") == source_a])
    total_fields_b = len([n for n, d in g.graph.nodes(data=True) if d.get("type") == "field" and d.get("source_db") == source_b])
    print(f"  Fuente A: {source_a} ({total_fields_a} campos)")
    print(f"  Fuente B: {source_b} ({total_fields_b} campos)")
    print(f"  Equivalencias descubiertas: {len(equivalences)}")
    print(f"  Equivalencias persistidas: {persisted}")
    print(f"  Cobertura de interoperabilidad: {len(equivalences)}/{min(total_fields_a, total_fields_b)} ({round(len(equivalences)/max(min(total_fields_a, total_fields_b), 1)*100, 1)}%)")


if __name__ == "__main__":
    main()
