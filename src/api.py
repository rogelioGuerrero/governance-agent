"""
Governance Agent REST API.

Expone el motor de governance (profiling, schema discovery, rapid assessment,
interoperability, transforms) via FastAPI para que nomenclador-explorer
u otros clientes puedan consumirlo.

Endpoints:
  POST /api/profiler/csv           — Perfila un CSV (metadata only)
  POST /api/profiler/postgresql     — Lee esquema de PostgreSQL (metadata only)
  POST /api/rapid-assessment        — Assessment completo de un CSV
  GET  /api/concepts                — Lista conceptos del nomenclador
  GET  /api/concepts/{name}         — Detalle de un concepto
  GET  /api/health                  — Health check del grafo
  GET  /api/interop/{source}/{target} — Interoperabilidad entre fuentes
  GET  /api/transform/{source}/{target} — Transform SQL entre fuentes
  POST /api/normative/upload          — Subir documento normativo al corpus RAG
  GET  /api/normative/search          — Buscar en corpus normativo
  GET  /api/normative/corpus          — Listar corpus normativo

Uso:
    governance-api --port 8001
    python -m src.api --port 8001
"""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

logger = logging.getLogger("governance.api")

NOMENCLADOR_PATH = Path(__file__).parent.parent / "nomenclador" / "nomenclador.json"


def create_app() -> FastAPI:
    app = FastAPI(
        title="Governance Agent API",
        description="Motor de governance para interoperabilidad semantica — profiling, schema discovery, quality, matching.",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # Profiling
    # ------------------------------------------------------------------

    @app.post("/api/profiler/csv")
    async def profile_csv_endpoint(file: UploadFile = File(...)):
        """Perfila un archivo CSV y retorna metadata del esquema.

        No persiste datos. Solo lee estructura: tipos, nulos, unicos, muestras, min/max.
        """
        from .profiler import profile_csv, detect_standards_for_columns

        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Empty file")

        # Escribir a temp file para que profile_csv lo lea
        with tempfile.NamedTemporaryFile(
            suffix=f"_{file.filename}", delete=False, mode="wb"
        ) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name

        try:
            tables = profile_csv(tmp_path)
            if not tables:
                raise HTTPException(status_code=400, detail="No se pudo perfilar el archivo")

            # Detectar estandares para cada tabla
            for table in tables:
                detect_standards_for_columns(table)

            return {
                "filename": file.filename,
                "table_count": len(tables),
                "tables": [
                    {
                        "name": t.name,
                        "row_count": t.row_count,
                        "columns": [
                            {
                                "name": c.column,
                                "data_type": c.data_type,
                                "nullable": c.nullable,
                                "total_count": c.total_count,
                                "null_count": c.null_count,
                                "unique_count": c.unique_count,
                                "sample_values": c.sample_values[:10],
                                "min_value": c.min_value,
                                "max_value": c.max_value,
                                "inferred_standard": c.inferred_standard,
                            }
                            for c in t.columns
                        ],
                    }
                    for t in tables
                ],
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("profile_csv_endpoint error")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    class PostgresProfileRequest(BaseModel):
        conn_string: str
        schema: str = "public"

    @app.post("/api/profiler/postgresql")
    async def profile_postgresql_endpoint(req: PostgresProfileRequest):
        """Lee el esquema de una base de datos PostgreSQL.

        Se conecta via psycopg, lee information_schema, cuenta filas,
        saca muestras y min/max. No transfiere datos.
        """
        from .profiler import profile_postgresql, detect_standards_for_columns

        try:
            tables = profile_postgresql(req.conn_string, req.schema)
            if not tables:
                return {"tables": [], "table_count": 0}

            for table in tables:
                detect_standards_for_columns(table)

            return {
                "schema": req.schema,
                "table_count": len(tables),
                "tables": [
                    {
                        "name": t.name,
                        "row_count": t.row_count,
                        "columns": [
                            {
                                "name": c.column,
                                "data_type": c.data_type,
                                "nullable": c.nullable,
                                "total_count": c.total_count,
                                "null_count": c.null_count,
                                "unique_count": c.unique_count,
                                "sample_values": c.sample_values[:10],
                                "min_value": c.min_value,
                                "max_value": c.max_value,
                                "inferred_standard": c.inferred_standard,
                            }
                            for c in t.columns
                        ],
                    }
                    for t in tables
                ],
            }
        except ImportError:
            raise HTTPException(
                status_code=503,
                detail="psycopg no instalado. Instala con: pip install psycopg[binary]",
            )
        except Exception as e:
            logger.exception("profile_postgresql_endpoint error")
            raise HTTPException(status_code=500, detail=str(e))

    # ------------------------------------------------------------------
    # Rapid Assessment
    # ------------------------------------------------------------------

    @app.post("/api/rapid-assessment")
    async def rapid_assessment_endpoint(file: UploadFile = File(...)):
        """Ejecuta rapid assessment completo sobre un CSV.

        Profiling + quality score (A-F) + anomalias + inferencia semantica +
        deteccion PII + matching contra conceptos + candidatos interop.
        Sin LLM, sin contexto humano.
        """
        from .rapid_assessment import assess_csv

        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Empty file")

        with tempfile.NamedTemporaryFile(
            suffix=f"_{file.filename}", delete=False, mode="wb"
        ) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name

        try:
            report = assess_csv(tmp_path)
            return {
                "source_file": report.source_file,
                "source_name": report.source_name,
                "generated_at": report.generated_at,
                "total_rows": report.total_rows,
                "total_columns": report.total_columns,
                "avg_quality_score": report.avg_quality_score,
                "global_grade": report.global_grade,
                "matched_count": report.matched_count,
                "inferred_count": report.inferred_count,
                "unmatched_count": report.unmatched_count,
                "pii_count": report.pii_count,
                "sensitive_count": report.sensitive_count,
                "issues_count": report.issues_count,
                "interop_candidates": report.interop_candidates,
                "columns": [
                    {
                        "name": c.name,
                        "clean_name": c.clean_name,
                        "data_type": c.data_type,
                        "total_count": c.total_count,
                        "null_count": c.null_count,
                        "unique_count": c.unique_count,
                        "sample_values": c.sample_values,
                        "min_value": c.min_value,
                        "max_value": c.max_value,
                        "quality_score": c.quality_score,
                        "quality_grade": c.quality_grade,
                        "completeness": c.completeness,
                        "consistency": c.consistency,
                        "validity": c.validity,
                        "issues": c.issues,
                        "anomaly_count": c.anomaly_count,
                        "anomaly_ratio": c.anomaly_ratio,
                        "inferred_type": c.inferred_type,
                        "inferred_confidence": c.inferred_confidence,
                        "inferred_reason": c.inferred_reason,
                        "suggested_concept": c.suggested_concept,
                        "suggested_standard": c.suggested_standard,
                        "matched_concept_id": c.matched_concept_id,
                        "is_pii": c.is_pii,
                        "is_sensitive": c.is_sensitive,
                        "sensitivity_reason": c.sensitivity_reason,
                        "match_status": c.match_status,
                        "match_detail": c.match_detail,
                    }
                    for c in report.columns
                ],
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("rapid_assessment_endpoint error")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Nomenclador queries
    # ------------------------------------------------------------------

    @app.get("/api/concepts")
    async def list_concepts():
        """Lista todos los conceptos canonicos del nomenclador."""
        from .graph.catalog import load_graph_cached

        g = load_graph_cached()
        concepts = g.list_concepts()
        return {"concepts": concepts, "count": len(concepts)}

    @app.get("/api/concepts/{name}")
    async def get_concept(name: str):
        """Detalle completo de un concepto incluyendo fields y clasificador."""
        from .graph.catalog import load_graph_cached

        g = load_graph_cached()
        concept = g.find_concept(name)
        if not concept:
            raise HTTPException(status_code=404, detail=f"Concepto '{name}' no encontrado")

        ctx = g.build_concept_context(concept["id"])
        return ctx

    # ------------------------------------------------------------------
    # Interoperability
    # ------------------------------------------------------------------

    @app.get("/api/interop/{source_db}/{target_db}")
    async def check_interop(source_db: str, target_db: str):
        """Verifica interoperabilidad entre dos fuentes con guardrails."""
        from .graph.catalog import load_graph_cached
        from .guardrails import validate_interoperability

        g = load_graph_cached()
        results = g.find_interoperability_path(source_db, target_db)
        if not results:
            return {"source": source_db, "target": target_db, "paths": [], "count": 0}

        paths = []
        for result in results:
            field_a = result["field_a"]
            field_b = result["field_b"]
            concept = result["concept"]
            classifier = result.get("classifier")

            validation = validate_interoperability(field_a, field_b, concept, classifier)

            paths.append({
                "concept": concept,
                "field_a": field_a,
                "field_b": field_b,
                "checkpoints": [
                    {
                        "name": cp.name,
                        "status": cp.status.value,
                        "detail": cp.detail,
                    }
                    for cp in validation.checkpoints
                ],
                "recommendation": validation.recommendation,
                "warnings": validation.warnings,
            })

        return {"source": source_db, "target": target_db, "paths": paths, "count": len(paths)}

    @app.get("/api/transform/{source_db}/{target_db}")
    async def get_transform(source_db: str, target_db: str):
        """Genera transformaciones SQL + JSON Schema entre dos fuentes."""
        from .graph.catalog import load_graph_cached
        from .guardrails import validate_interoperability
        from .transformer import generate_transformation, artifact_to_dict

        g = load_graph_cached()
        results = g.find_interoperability_path(source_db, target_db)
        if not results:
            return {"source": source_db, "target": target_db, "transforms": []}

        transforms = []
        for result in results:
            field_a = result["field_a"]
            field_b = result["field_b"]
            concept = result["concept"]
            classifier = result.get("classifier")

            validation = validate_interoperability(field_a, field_b, concept, classifier)
            artifact = generate_transformation(field_a, field_b, concept, classifier, validation)

            transforms.append({
                "concept_name": artifact.concept_name,
                "standard": artifact.standard,
                "sql_transform": artifact.sql_transform,
                "json_schema": artifact.json_schema,
                "quality_assessment": artifact.quality_assessment,
                "warnings": validation.warnings,
            })

        return {"source": source_db, "target": target_db, "transforms": transforms}

    # ------------------------------------------------------------------
    # Normative RAG — corpus documental de respaldo
    # ------------------------------------------------------------------

    @app.post("/api/normative/upload")
    async def normative_upload(
        file: UploadFile = File(...),
        tags: str = Query("", description="Comma-separated tags (concept names)"),
    ):
        """Sube un documento normativo al corpus RAG.

        Chunking + embeddings (Cohere) + almacenamiento en normative_corpus.json.
        El agent usara este corpus para buscar respaldo de variables.
        """
        from .normative_rag import NormativeRAG

        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Empty file")

        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None
        source_name = Path(file.filename or "upload").stem

        with tempfile.NamedTemporaryFile(
            suffix=f"_{file.filename}", delete=False, mode="wb"
        ) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name

        try:
            rag = NormativeRAG()
            n_chunks = rag.ingest_file(tmp_path, source=source_name, tags=tag_list)
            if n_chunks == 0:
                raise HTTPException(
                    status_code=400,
                    detail="No se pudieron extraer chunks del documento. Requiere COHERE_API_KEY para embeddings.",
                )
            stats = rag.stats()
            return {
                "source": source_name,
                "filename": file.filename,
                "tags": tag_list or [],
                "chunks_ingested": n_chunks,
                "corpus_total_chunks": stats["total_chunks"],
                "corpus_total_sources": stats["total_sources"],
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("normative_upload error")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    @app.get("/api/normative/search")
    async def normative_search(
        q: str = Query(..., description="Search query (variable name or keyword)"),
        top_k: int = Query(5, description="Max results"),
    ):
        """Busca en el corpus normativo (hybrid: BM25 + cosine similarity)."""
        from .normative_rag import NormativeRAG

        try:
            rag = NormativeRAG()
            if rag.stats()["total_chunks"] == 0:
                return {"results": [], "count": 0, "total_chunks": 0}
            results = rag.search(q, top_k=top_k)
            return {
                "query": q,
                "results": results,
                "count": len(results),
                "total_chunks": rag.stats()["total_chunks"],
            }
        except Exception as e:
            logger.exception("normative_search error")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/normative/corpus")
    async def normative_corpus():
        """Lista metadata del corpus normativo (sin embeddings)."""
        from .normative_rag import NormativeRAG

        try:
            rag = NormativeRAG()
            stats = rag.stats()
            chunks = [
                {
                    "id": c.id,
                    "source": c.source,
                    "source_type": c.source_type,
                    "chunk_index": c.chunk_index,
                    "text": c.text[:200] + "..." if len(c.text) > 200 else c.text,
                    "tags": c.tags,
                }
                for c in rag.chunks
            ]
            return {
                "total_chunks": stats["total_chunks"],
                "total_sources": stats["total_sources"],
                "chunks": chunks,
            }
        except Exception as e:
            logger.exception("normative_corpus error")
            raise HTTPException(status_code=500, detail=str(e))

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    @app.get("/api/health")
    async def health():
        """Health check del governance agent y su grafo."""
        from .graph.catalog import load_graph_cached

        try:
            g = load_graph_cached()
            node_count = g.graph.number_of_nodes()
            edge_count = g.graph.number_of_edges()
            concepts = g.list_concepts()
            return {
                "status": "ok",
                "graph": {
                    "nodes": node_count,
                    "edges": edge_count,
                    "concepts": len(concepts),
                },
            }
        except Exception as e:
            return {"status": "error", "detail": str(e)}

    @app.get("/")
    async def root():
        return {
            "service": "governance-agent",
            "version": "0.1.0",
            "docs": "/docs",
            "endpoints": [
                "POST /api/profiler/csv",
                "POST /api/profiler/postgresql",
                "POST /api/rapid-assessment",
                "GET  /api/concepts",
                "GET  /api/concepts/{name}",
                "GET  /api/interop/{source}/{target}",
                "GET  /api/transform/{source}/{target}",
                "POST /api/normative/upload",
                "GET  /api/normative/search",
                "GET  /api/normative/corpus",
                "GET  /api/health",
            ],
        }

    return app


def main(argv=None):
    """CLI entry point for the Governance Agent API server."""
    parser = argparse.ArgumentParser(
        prog="governance-api",
        description="Governance Agent REST API — profiling, schema discovery, interoperability.",
    )
    parser.add_argument(
        "--port", "-p", type=int, default=8001,
        help="Port to bind (default: 8001)",
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="Host to bind (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--reload", action="store_true",
        help="Enable auto-reload for development",
    )
    args = parser.parse_args(argv)

    import uvicorn

    logger.info("Starting Governance Agent API on %s:%d", args.host, args.port)
    uvicorn.run(
        "src.api:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
