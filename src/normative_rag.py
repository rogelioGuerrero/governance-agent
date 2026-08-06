"""
RAG Documental: vector store local de leyes/normativas que respaldan
cada variable canonica del nomenclador.

Stack:
- Cohere embed-multilingual-v3.0 (1024d) para embeddings
- JSON local para almacenamiento (nomenclador/normative_corpus.json)
- Hybrid search: BM25 (keyword) + cosine similarity (semantic) en Python puro
- Groq gpt-oss-120b para resumir chunks largos en citas breves

Uso:
  from src.normative_rag import NormativeRAG
  rag = NormativeRAG()
  rag.ingest_text("Ley General de Salud Art. 47: ...", source="ley_general_salud", tag="sexo")
  results = rag.search("sexo ISO 5218 norma legal", top_k=3)
  backing = rag.find_backing("sexo", "ISO_5218")
"""

import json
import math
import os
import re
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from .log_config import get_logger

log = get_logger("normative_rag")

NORMATIVE_CORPUS_PATH = Path(__file__).parent.parent / "nomenclador" / "normative_corpus.json"

COHERE_EMBED_URL = "https://api.cohere.com/v2/embed"

load_dotenv(Path(__file__).parent.parent / ".env")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY", "")
EMBEDDING_MODEL = "embed-multilingual-v3.0"
EMBEDDING_DIMENSIONS = 1024
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
SIMILARITY_THRESHOLD = 0.65
BM25_WEIGHT = 0.4
COSINE_WEIGHT = 0.6
BM25_K1 = 1.5
BM25_B = 0.75


@dataclass
class NormativeChunk:
    id: str
    source: str
    source_type: str
    chunk_index: int
    text: str
    embedding: list[float] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


_RE_TOKENIZE = re.compile(r"\W+")


def _tokenize(text: str) -> list[str]:
    """Tokenizacion simple: lowercase, split por espacios y puntuacion."""
    return [t for t in _RE_TOKENIZE.split(text.lower()) if len(t) > 1]


class _BM25Index:
    """Indice BM25 en memoria para keyword search sobre chunks de normativa.

    BM25 (Best Match 25) es el algoritmo estandar de ranking para busqueda
    por palabras clave. Combina TF (term frequency) con IDF (inverse document
    frequency) y normalizacion por longitud de documento.

    Se usa junto a cosine similarity para hybrid search: BM25 captura matches
    exactos de terminos legales (ej: "Art. 47", "ISO 5218") que cosine puede
    perder cuando el embedding semantico no alinea perfectamente.
    """

    def __init__(self, k1: float = BM25_K1, b: float = BM25_B):
        self.k1 = k1
        self.b = b
        self.doc_freqs: list[dict[str, int]] = []
        self.doc_lens: list[int] = []
        self.avgdl: float = 0.0
        self.idf: dict[str, float] = {}
        self.n_docs: int = 0

    def build(self, texts: list[str]):
        """Construir el indice desde una lista de documentos."""
        self.n_docs = len(texts)
        self.doc_freqs = []
        self.doc_lens = []
        total_len = 0
        df: dict[str, int] = {}

        for text in texts:
            tokens = _tokenize(text)
            self.doc_lens.append(len(tokens))
            total_len += len(tokens)
            tf: dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self.doc_freqs.append(tf)
            for term in tf:
                df[term] = df.get(term, 0) + 1

        self.avgdl = total_len / self.n_docs if self.n_docs > 0 else 0.0
        self.idf = {}
        for term, freq in df.items():
            self.idf[term] = math.log(1 + (self.n_docs - freq + 0.5) / (freq + 0.5))

    def score(self, query: str, doc_idx: int) -> float:
        """Puntuar un documento contra un query con BM25."""
        if doc_idx >= self.n_docs or not self.idf:
            return 0.0
        query_tokens = _tokenize(query)
        tf = self.doc_freqs[doc_idx]
        dl = self.doc_lens[doc_idx]
        score = 0.0
        for term in query_tokens:
            if term not in self.idf or term not in tf:
                continue
            idf = self.idf[term]
            f = tf[term]
            denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
            score += idf * (f * (self.k1 + 1)) / denom
        return score


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) > chunk_size:
            if current:
                chunks.append(current.strip())
                current = ""
            sentences = para.split(". ")
            sent_chunk = ""
            for s in sentences:
                if len(sent_chunk) + len(s) > chunk_size:
                    if sent_chunk:
                        chunks.append(sent_chunk.strip())
                    sent_chunk = s
                else:
                    sent_chunk = f"{sent_chunk}. {s}" if sent_chunk else s
            if sent_chunk:
                chunks.append(sent_chunk.strip())
        else:
            if len(current) + len(para) > chunk_size:
                if current:
                    chunks.append(current.strip())
                    tail = current[-overlap:] if overlap > 0 else ""
                    current = f"{tail}\n\n{para}"
                else:
                    current = para
            else:
                current = f"{current}\n\n{para}" if current else para
    if current:
        chunks.append(current.strip())
    return [c for c in chunks if len(c) > 30]


def _cohere_embed(texts: list[str]) -> list[list[float]]:
    if not COHERE_API_KEY:
        raise RuntimeError("COHERE_API_KEY no configurada")
    all_embeddings: list[list[float]] = []
    batch_size = 96
    max_retries = 2
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        payload = json.dumps({
            "model": EMBEDDING_MODEL,
            "texts": batch,
            "input_type": "search_document",
            "embedding_types": ["float"],
        }).encode("utf-8")
        req = urllib.request.Request(
            COHERE_EMBED_URL,
            data=payload,
            headers={
                "Authorization": f"Bearer {COHERE_API_KEY}",
                "Content-Type": "application/json",
            },
        )
        for retry in range(max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=45) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                all_embeddings.extend(data["embeddings"]["float"])
                break
            except urllib.error.HTTPError as e:
                if e.code == 429 and retry < max_retries:
                    import time
                    wait = 2 ** (retry + 1)
                    log.warning("Cohere 429 rate limit, esperando %ss...", wait)
                    time.sleep(wait)
                    continue
                raise
            except urllib.error.URLError as e:
                if retry < max_retries:
                    import time
                    wait = 2 ** (retry + 1)
                    log.warning("Cohere URLError: %s, reintentando en %ss...", e, wait)
                    time.sleep(wait)
                    continue
                raise
    return all_embeddings


def _cohere_embed_query(query: str) -> list[float]:
    if not COHERE_API_KEY:
        raise RuntimeError("COHERE_API_KEY no configurada")
    payload = json.dumps({
        "model": EMBEDDING_MODEL,
        "texts": [query],
        "input_type": "search_query",
        "embedding_types": ["float"],
    }).encode("utf-8")
    req = urllib.request.Request(
        COHERE_EMBED_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {COHERE_API_KEY}",
            "Content-Type": "application/json",
        },
    )
    max_retries = 2
    for retry in range(max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["embeddings"]["float"][0]
        except urllib.error.HTTPError as e:
            if e.code == 429 and retry < max_retries:
                import time
                wait = 2 ** (retry + 1)
                log.warning("Cohere query 429, esperando %ss...", wait)
                time.sleep(wait)
                continue
            raise
        except urllib.error.URLError as e:
            if retry < max_retries:
                import time
                wait = 2 ** (retry + 1)
                log.warning("Cohere query URLError: %s, reintentando en %ss...", e, wait)
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("Cohere embed query: max retries exceeded")


class NormativeRAG:
    """Vector store local de normativas con hybrid search (BM25 + cosine)."""

    def __init__(self, corpus_path: Path | None = None):
        self.corpus_path = corpus_path or NORMATIVE_CORPUS_PATH
        self.chunks: list[NormativeChunk] = []
        self._bm25 = _BM25Index()
        self._load()

    def _load(self):
        if self.corpus_path.exists():
            raw = json.loads(self.corpus_path.read_text(encoding="utf-8"))
            self.chunks = [NormativeChunk(**c) for c in raw.get("chunks", [])]
        self._rebuild_bm25()

    def _rebuild_bm25(self):
        """Reconstruir el indice BM25 despues de cambios en chunks."""
        if self.chunks:
            self._bm25.build([c.text for c in self.chunks])

    def _save(self):
        self.corpus_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": "1.0",
            "embedding_model": EMBEDDING_MODEL,
            "dimensions": EMBEDDING_DIMENSIONS,
            "chunks": [
                {
                    "id": c.id,
                    "source": c.source,
                    "source_type": c.source_type,
                    "chunk_index": c.chunk_index,
                    "text": c.text,
                    "embedding": c.embedding,
                    "tags": c.tags,
                    "metadata": c.metadata,
                }
                for c in self.chunks
            ],
        }
        self.corpus_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        self._rebuild_bm25()

    def ingest_text(
        self,
        text: str,
        source: str,
        source_type: str = "text",
        tags: list[str] | None = None,
        metadata: dict | None = None,
    ) -> int:
        """Ingerir texto: chunking + embedding + storage. Retorna num chunks."""
        if not text or len(text) < 50:
            return 0
        chunks_text = _chunk_text(text)
        if not chunks_text:
            return 0
        try:
            embeddings = _cohere_embed(chunks_text)
        except Exception as e:
            log.error("Error embedding: %s", e)
            return 0
        existing = [c for c in self.chunks if c.source == source]
        if existing:
            self.chunks = [c for c in self.chunks if c.source != source]
        base_idx = len(self.chunks)
        for i, (ct, emb) in enumerate(zip(chunks_text, embeddings)):
            chunk_id = f"norm:{source}:{i}"
            self.chunks.append(NormativeChunk(
                id=chunk_id,
                source=source,
                source_type=source_type,
                chunk_index=i,
                text=ct,
                embedding=emb,
                tags=tags or [],
                metadata=metadata or {},
            ))
        self._save()
        return len(chunks_text)

    def ingest_file(
        self,
        file_path: str,
        source: str | None = None,
        tags: list[str] | None = None,
    ) -> int:
        """Ingerir archivo de texto o PDF simplificado."""
        p = Path(file_path)
        if not p.exists():
            log.error("Archivo no encontrado: %s", file_path)
            return 0
        source = source or p.stem
        text = p.read_text(encoding="utf-8", errors="replace")
        return self.ingest_text(text, source=source, source_type="file", tags=tags)

    def search(
        self,
        query: str,
        top_k: int = 5,
        tag_filter: str | None = None,
    ) -> list[dict]:
        """Buscar chunks con hybrid search: BM25 (keyword) + cosine (semantic).

        Combina dos senales:
        - BM25: captura matches exactos de terminos (ej: "Art. 47", "ISO 5218")
        - Cosine: captura similitud semantica (ej: "genero" ~= "sexo")

        Los scores se normalizan a [0, 1] antes de combinar con pesos
        BM25_WEIGHT y COSINE_WEIGHT. Retorna lista de dicts con score hibrido.
        """
        if not self.chunks:
            return []
        try:
            query_emb = _cohere_embed_query(query)
        except Exception as e:
            log.error("Error embedding query: %s", e)
            return []
        candidates = self.chunks
        if tag_filter:
            candidates = [c for c in candidates if tag_filter in c.tags]
        if not candidates:
            return []

        # Calcular scores de ambos metodos
        # Mapear chunks a indices globales una sola vez (evita O(n²) con .index())
        chunk_to_idx = {id(c): i for i, c in enumerate(self.chunks)}
        cosine_scores = []
        bm25_scores = []
        for chunk in candidates:
            cs = _cosine_similarity(query_emb, chunk.embedding)
            cosine_scores.append(cs)
            global_idx = chunk_to_idx[id(chunk)]
            bs = self._bm25.score(query, global_idx)
            bm25_scores.append(bs)

        # Normalizar BM25 a [0, 1] (puede ser > 1, dividir por max)
        max_bm25 = max(bm25_scores) if bm25_scores else 1.0
        if max_bm25 == 0:
            max_bm25 = 1.0

        scored = []
        for i, chunk in enumerate(candidates):
            norm_bm25 = bm25_scores[i] / max_bm25
            norm_cosine = max(0.0, cosine_scores[i])
            hybrid = norm_bm25 * BM25_WEIGHT + norm_cosine * COSINE_WEIGHT
            scored.append({
                "id": chunk.id,
                "source": chunk.source,
                "text": chunk.text,
                "score": hybrid,
                "cosine_score": round(norm_cosine, 3),
                "bm25_score": round(norm_bm25, 3),
                "tags": chunk.tags,
            })
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def find_backing(
        self,
        concept_name: str,
        standard_id: str | None = None,
        threshold: float = SIMILARITY_THRESHOLD,
    ) -> dict:
        """
        Buscar respaldo normativo para un concepto canonico.
        Retorna dict con: found (bool), references (list), summary (str).
        """
        query = f"{concept_name} {standard_id or ''} norma legal definicion"
        # Filtrar por tag solo si algun chunk tiene el concept_name como tag
        use_tag = any(concept_name in c.tags for c in self.chunks)
        results = self.search(query, top_k=3, tag_filter=concept_name if use_tag else None)
        above = [r for r in results if r["score"] >= threshold]
        refs = []
        for r in above:
            cite = r["text"][:200].replace("\n", " ")
            refs.append({
                "source": r["source"],
                "score": round(r["score"], 3),
                "cite": cite,
            })
        return {
            "concept": concept_name,
            "standard": standard_id,
            "found": len(above) > 0,
            "references": refs,
            "best_score": round(above[0]["score"], 3) if above else 0.0,
        }

    def batch_backing(
        self,
        concepts: list[dict],
        threshold: float = SIMILARITY_THRESHOLD,
    ) -> list[dict]:
        """
        Batch: dado una lista de {name, standard}, buscar respaldo normativo
        para todos. Retorna lista de resultados.
        """
        results = []
        for c in concepts:
            backing = self.find_backing(c["name"], c.get("standard"), threshold)
            results.append(backing)
        return results

    def stats(self) -> dict:
        sources = set(c.source for c in self.chunks)
        return {
            "total_chunks": len(self.chunks),
            "total_sources": len(sources),
            "sources": list(sources),
        }
