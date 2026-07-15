"""
Self-monitoring y auto-healing para el governance-agent.

Inspirado en el monitor-agent de BienCuidar pero adaptado a un sistema
local + stateful con Knowledge Graph.

Capacidades:
1. check_health() — diagnóstico completo del estado del governance-agent
2. fix_orphan_nodes() — auto-healing: linkea nodos huérfanos a conceptos
3. retry_stuck_proposals() — re-intenta nodos proposed que llevan días sin revisión
4. log_health_run() — heartbeat a PostgreSQL para detectar si el agente deja de correr

Uso:
    from src.health import check_health, fix_orphan_nodes, retry_stuck_proposals
    report = check_health()  # dict con estado completo
    fixes = fix_orphan_nodes(dry_run=True)  # lista de fixes aplicados/sugeridos
"""

import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from .graph.catalog import NomencladorGraph, load_graph_cached, clear_graph_cache
from .graph.schema import NodeType, EdgeType, ReviewStatus
from .verifier import verify_graph_invariants, verify_all_fields, compute_all_confidence

logger = logging.getLogger(__name__)

NOMENCLADOR_PATH = Path(__file__).parent.parent / "nomenclador" / "nomenclador.json"


def _get_db():
    """Obtener conexión PostgreSQL si está disponible."""
    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        return None
    try:
        import psycopg
        return psycopg.connect(db_url)
    except Exception:
        return None


def check_health() -> dict:
    """Ejecutar diagnóstico completo del governance-agent.

    Returns:
        dict con:
        - timestamp
        - graph_stats: nodos, edges, conceptos, fields, clasificadores
        - connectivity: postgres, groq_api
        - graph_audit: violations, warnings (from verify_graph_invariants)
        - coverage: % approved vs proposed vs rejected
        - quality: avg quality_score, fields con baja calidad
        - stale_proposals: nodos proposed sin revisión por N días
        - orphan_nodes: nodos sin aristas
        - fields_without_concept: fields que no implementan ningún concepto
        - classifier_issues: fields con match_ratio < 1.0
        - passed: bool (True si no hay violations críticas)
    """
    now = datetime.now()
    report = {
        "timestamp": now.isoformat(),
        "graph_stats": {},
        "connectivity": {},
        "graph_audit": {},
        "coverage": {},
        "quality": {},
        "stale_proposals": [],
        "orphan_nodes": [],
        "fields_without_concept": [],
        "classifier_issues": [],
        "passed": True,
    }

    # === 1. GRAPH STATS ===
    g = load_graph_cached()
    nodes = list(g.graph.nodes(data=True))
    edges = list(g.graph.edges(data=True))

    node_types = {}
    for _, data in nodes:
        t = data.get("type", "unknown")
        node_types[t] = node_types.get(t, 0) + 1

    report["graph_stats"] = {
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "by_type": node_types,
        "version": g.version,
    }

    # === 2. CONNECTIVITY ===
    # PostgreSQL
    db = _get_db()
    if db:
        try:
            with db.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
            report["connectivity"]["postgres"] = "OK"
        except Exception as e:
            report["connectivity"]["postgres"] = f"FAIL: {e}"
            report["passed"] = False
        finally:
            try:
                db.close()
            except Exception:
                pass
    else:
        report["connectivity"]["postgres"] = "NOT_CONFIGURED (using JSON fallback)"

    # Groq API key
    groq_key = os.environ.get("GROQ_API_KEY", "")
    report["connectivity"]["groq_api"] = "OK" if groq_key else "MISSING_KEY"

    # Cohere API key (si existe)
    cohere_key = os.environ.get("COHERE_API_KEY", "")
    report["connectivity"]["cohere_api"] = "OK" if cohere_key else "NOT_CONFIGURED"

    # === 3. GRAPH AUDIT (determinista) ===
    audit = verify_graph_invariants(g)
    report["graph_audit"] = {
        "violations": len(audit["violations"]),
        "warnings": len(audit["warnings"]),
        "passed": audit["passed"],
        "violation_details": audit["violations"][:10],
        "warning_details": audit["warnings"][:10],
    }
    if not audit["passed"]:
        report["passed"] = False

    # === 4. COVERAGE (review status) ===
    review_counts = {"approved": 0, "proposed": 0, "under_review": 0, "rejected": 0, "other": 0}
    for _, data in nodes:
        status = data.get("review_status", "approved")
        if status in review_counts:
            review_counts[status] += 1
        else:
            review_counts["other"] += 1

    total = sum(review_counts.values())
    report["coverage"] = {
        "counts": review_counts,
        "pct_approved": round(review_counts["approved"] / total * 100, 1) if total else 0,
        "pct_proposed": round(review_counts["proposed"] / total * 100, 1) if total else 0,
        "pct_pending_review": round(
            (review_counts["proposed"] + review_counts["under_review"]) / total * 100, 1
        ) if total else 0,
    }

    # === 5. QUALITY ===
    quality_scores = []
    low_quality = []
    for node_id, data in nodes:
        if data.get("type") == NodeType.FIELD.value:
            qs = data.get("quality_score", 0.0)
            quality_scores.append(qs)
            if qs < 0.4:
                low_quality.append({
                    "id": node_id,
                    "column": data.get("column", "?"),
                    "source_db": data.get("source_db", "?"),
                    "quality_score": qs,
                })

    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
    report["quality"] = {
        "avg_quality_score": round(avg_quality, 3),
        "total_fields": len(quality_scores),
        "low_quality_count": len(low_quality),
        "low_quality_fields": low_quality[:10],
    }

    # === 6. STALE PROPOSALS (proposed sin revisión por 7+ días) ===
    stale_threshold = now - timedelta(days=7)
    for node_id, data in nodes:
        status = data.get("review_status", "approved")
        if status in ("proposed", "under_review"):
            created = data.get("created_at", "")
            if created:
                try:
                    created_dt = datetime.fromisoformat(created.replace("Z", "+00:00").replace("+00:00", ""))
                    if created_dt < stale_threshold:
                        report["stale_proposals"].append({
                            "id": node_id,
                            "name": data.get("name", data.get("column", "?")),
                            "type": data.get("type", "?"),
                            "status": status,
                            "created_at": created,
                            "days_stale": (now - created_dt).days,
                        })
                except (ValueError, TypeError):
                    pass
            else:
                # Sin created_at, asumir stale
                report["stale_proposals"].append({
                    "id": node_id,
                    "name": data.get("name", data.get("column", "?")),
                    "type": data.get("type", "?"),
                    "status": status,
                    "created_at": "unknown",
                    "days_stale": -1,
                })

    # === 7. ORPHAN NODES (sin aristas) ===
    for node_id, data in nodes:
        if g.graph.degree(node_id) == 0:
            report["orphan_nodes"].append({
                "id": node_id,
                "name": data.get("name", data.get("column", "?")),
                "type": data.get("type", "?"),
            })

    # === 8. FIELDS WITHOUT CONCEPT ===
    for node_id, data in nodes:
        if data.get("type") == NodeType.FIELD.value:
            has_concept = False
            for successor in g.graph.successors(node_id):
                edge = g.graph.get_edge_data(node_id, successor)
                if edge and edge.get("type") == EdgeType.IMPLEMENTA.value:
                    has_concept = True
                    break
            if not has_concept:
                report["fields_without_concept"].append({
                    "id": node_id,
                    "column": data.get("column", "?"),
                    "source_db": data.get("source_db", "?"),
                })

    # === 9. CLASSIFIER ISSUES (fields con match_ratio < 1.0) ===
    try:
        field_results = verify_all_fields(g)
        for r in field_results:
            if "error" not in r and r.get("match_ratio", 1.0) < 1.0:
                report["classifier_issues"].append({
                    "field_id": r["field_id"],
                    "match_ratio": round(r["match_ratio"], 3),
                    "invalid_count": r["invalid_count"],
                    "invalid_values": r["invalid_values"][:5],
                })
    except Exception as e:
        logger.warning("Classifier verification failed: %s", e)

    return report


def fix_orphan_nodes(dry_run: bool = True) -> dict:
    """Auto-healing: intentar linkear nodos huérfanos a conceptos existentes.

    Estrategia:
    - Para fields huérfanos: buscar concepto con nombre similar (fuzzy match simple)
    - Para classifiers huérfanos: buscar concepto que use el mismo standard
    - Para concepts huérfanos: reportar (requiere decisión humana)

    Args:
        dry_run: si True, solo sugiere fixes sin aplicarlos

    Returns:
        dict con: fixes_applied, fixes_suggested, manual_needed
    """
    g = load_graph_cached()
    fixes_applied = []
    fixes_suggested = []
    manual_needed = []

    concepts = g.list_concepts()
    concept_names = {c["id"]: c.get("name", "").lower() for c in concepts}

    for node_id, data in list(g.graph.nodes(data=True)):
        if g.graph.degree(node_id) > 0:
            continue

        node_type = data.get("type", "")
        node_name = data.get("name", data.get("column", "")).lower()

        if node_type == NodeType.FIELD.value:
            # Buscar concepto con nombre similar
            best_match = None
            best_score = 0
            for cid, cname in concept_names.items():
                if not cname:
                    continue
                # Simple fuzzy: palabras en común
                node_words = set(node_name.split("_"))
                concept_words = set(cname.split())
                common = node_words & concept_words
                score = len(common) / max(len(node_words), 1) if node_words else 0
                if score > best_score and score >= 0.5:
                    best_score = score
                    best_match = cid

            if best_match:
                fix = {
                    "node_id": node_id,
                    "node_name": node_name,
                    "suggested_concept": best_match,
                    "match_score": round(best_score, 2),
                    "action": "link_implementa",
                }
                if dry_run:
                    fixes_suggested.append(fix)
                else:
                    g.link_implementa(node_id, best_match)
                    fixes_applied.append(fix)
            else:
                manual_needed.append({
                    "node_id": node_id,
                    "node_name": node_name,
                    "reason": "No se encontró concepto similar (score < 0.5)",
                })

        elif node_type == NodeType.CONCEPT.value:
            manual_needed.append({
                "node_id": node_id,
                "node_name": node_name,
                "reason": "Concepto huérfano — requiere definición humana de relaciones",
            })

        elif node_type == NodeType.CLASSIFIER.value:
            manual_needed.append({
                "node_id": node_id,
                "node_name": node_name,
                "reason": "Clasificador huérfano — requiere asociación manual a concepto",
            })

    if not dry_run and fixes_applied:
        g.save(str(NOMENCLADOR_PATH))
        clear_graph_cache()

    return {
        "fixes_applied": fixes_applied,
        "fixes_suggested": fixes_suggested,
        "manual_needed": manual_needed,
        "dry_run": dry_run,
    }


def retry_stuck_proposals(dry_run: bool = True) -> dict:
    """Auto-healing: re-intentar nodos proposed que llevan días sin revisión.

    Estrategia:
    - Nodos proposed con quality_score >= 0.7 y sin issues → auto-approve
    - Nodos proposed con quality_score < 0.4 → marcar para revisión manual
    - Nodos under_review por 30+ días → alertar

    Args:
        dry_run: si True, solo sugiere sin aplicar

    Returns:
        dict con: auto_approved, flagged_manual, alerts
    """
    g = load_graph_cached()
    now = datetime.now()
    auto_approved = []
    flagged_manual = []
    alerts = []

    for node_id, data in g.graph.nodes(data=True):
        status = data.get("review_status", "approved")
        if status not in ("proposed", "under_review"):
            continue

        created = data.get("created_at", "")
        days_stale = 0
        if created:
            try:
                created_dt = datetime.fromisoformat(created.replace("Z", "").split("+")[0])
                days_stale = (now - created_dt).days
            except (ValueError, TypeError):
                pass

        qs = data.get("quality_score", 0.0)
        node_name = data.get("name", data.get("column", "?"))

        if status == "proposed" and days_stale >= 7:
            if qs >= 0.7:
                fix = {
                    "node_id": node_id,
                    "node_name": node_name,
                    "quality_score": qs,
                    "days_stale": days_stale,
                    "action": "auto_approve",
                    "reason": f"Quality score {qs} >= 0.7 y propuesto hace {days_stale} días",
                }
                if dry_run:
                    auto_approved.append(fix)
                else:
                    g.approve_node(node_id)
                    auto_approved.append(fix)
            elif qs < 0.4:
                flagged_manual.append({
                    "node_id": node_id,
                    "node_name": node_name,
                    "quality_score": qs,
                    "days_stale": days_stale,
                    "reason": f"Quality score {qs} < 0.4 — requiere revisión manual",
                })

        if status == "under_review" and days_stale >= 30:
            alerts.append({
                "node_id": node_id,
                "node_name": node_name,
                "days_stale": days_stale,
                "reason": f"En revisión por {days_stale} días — posible abandono",
            })

    if not dry_run and auto_approved:
        g.save(str(NOMENCLADOR_PATH))
        clear_graph_cache()

    return {
        "auto_approved": auto_approved,
        "flagged_manual": flagged_manual,
        "alerts": alerts,
        "dry_run": dry_run,
    }


def log_health_run(report: dict) -> bool:
    """Heartbeat: loguear el resultado del health check a PostgreSQL.

    Crea la tabla governance.health_runs si no existe.
    Esto permite detectar si el governance-agent deja de correr.

    Returns:
        True si se logueó correctamente, False si falló.
    """
    db = _get_db()
    if not db:
        return False

    try:
        with db.cursor() as cur:
            # Crear tabla si no existe
            cur.execute("""
                CREATE TABLE IF NOT EXISTS governance.health_runs (
                    id SERIAL PRIMARY KEY,
                    run_at TIMESTAMPTZ DEFAULT NOW(),
                    passed BOOLEAN,
                    total_nodes INT,
                    total_edges INT,
                    violations INT,
                    warnings INT,
                    stale_proposals INT,
                    orphan_nodes INT,
                    fields_without_concept INT,
                    classifier_issues INT,
                    report JSONB
                )
            """)
            db.commit()

            # Insertar health run
            cur.execute("""
                INSERT INTO governance.health_runs
                    (passed, total_nodes, total_edges, violations, warnings,
                     stale_proposals, orphan_nodes, fields_without_concept,
                     classifier_issues, report)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                report["passed"],
                report["graph_stats"]["total_nodes"],
                report["graph_stats"]["total_edges"],
                report["graph_audit"]["violations"],
                report["graph_audit"]["warnings"],
                len(report["stale_proposals"]),
                len(report["orphan_nodes"]),
                len(report["fields_without_concept"]),
                len(report["classifier_issues"]),
                json.dumps(report, ensure_ascii=False, default=str),
            ))
            db.commit()
        return True
    except Exception as e:
        logger.warning("log_health_run failed: %s", e)
        try:
            db.rollback()
        except Exception:
            pass
        return False
    finally:
        try:
            db.close()
        except Exception:
            pass


def format_health_report(report: dict) -> str:
    """Formatear el reporte de salud para salida CLI/consola."""
    lines = [
        f"=== GOVERNANCE AGENT HEALTH REPORT ===",
        f"Timestamp: {report['timestamp']}",
        f"Overall: {'PASS' if report['passed'] else 'FAIL'}",
        "",
        "--- GRAPH STATS ---",
        f"  Nodes: {report['graph_stats']['total_nodes']}",
        f"  Edges: {report['graph_stats']['total_edges']}",
        f"  Version: {report['graph_stats']['version']}",
        f"  By type: {json.dumps(report['graph_stats']['by_type'])}",
        "",
        "--- CONNECTIVITY ---",
    ]

    for service, status in report["connectivity"].items():
        icon = "OK" if "OK" in status else "!!"
        lines.append(f"  [{icon}] {service}: {status}")

    lines.extend([
        "",
        "--- GRAPH AUDIT ---",
        f"  Violations: {report['graph_audit']['violations']}",
        f"  Warnings: {report['graph_audit']['warnings']}",
        f"  Audit passed: {report['graph_audit']['passed']}",
    ])

    for v in report["graph_audit"].get("violation_details", []):
        lines.append(f"  VIOLATION: {v['message']}")

    lines.extend([
        "",
        "--- COVERAGE ---",
        f"  Approved: {report['coverage']['counts']['approved']} ({report['coverage']['pct_approved']}%)",
        f"  Proposed: {report['coverage']['counts']['proposed']} ({report['coverage']['pct_proposed']}%)",
        f"  Pending review: {report['coverage']['pct_pending_review']}%",
    ])

    lines.extend([
        "",
        "--- QUALITY ---",
        f"  Avg quality: {report['quality']['avg_quality_score']}",
        f"  Low quality fields: {report['quality']['low_quality_count']}/{report['quality']['total_fields']}",
    ])

    if report["stale_proposals"]:
        lines.append(f"\n--- STALE PROPOSALS ({len(report['stale_proposals'])}) ---")
        for s in report["stale_proposals"][:5]:
            lines.append(f"  - {s['name']} ({s['type']}) — {s['days_stale']} días stale")

    if report["orphan_nodes"]:
        lines.append(f"\n--- ORPHAN NODES ({len(report['orphan_nodes'])}) ---")
        for o in report["orphan_nodes"][:5]:
            lines.append(f"  - {o['name']} ({o['type']})")

    if report["fields_without_concept"]:
        lines.append(f"\n--- FIELDS WITHOUT CONCEPT ({len(report['fields_without_concept'])}) ---")
        for f in report["fields_without_concept"][:5]:
            lines.append(f"  - {f['source_db']}.{f['column']}")

    if report["classifier_issues"]:
        lines.append(f"\n--- CLASSIFIER ISSUES ({len(report['classifier_issues'])}) ---")
        for c in report["classifier_issues"][:5]:
            lines.append(f"  - {c['field_id']}: match={c['match_ratio']:.0%}, invalid={c['invalid_count']}")

    lines.append("\n=== END REPORT ===")
    return "\n".join(lines)
