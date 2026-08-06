"""
Evaluacion del RAG documental (normative_rag.py) con metricas estilo RAGAS.

No instala dependencias nuevas. No modifica normative_rag.py.
Usa Groq gpt-oss-120b como LLM-as-judge para 4 metricas:

1. Faithfulness: cada claim en la respuesta esta respaldada por el contexto recuperado?
2. Answer Relevancy: la respuesta aborda la pregunta?
3. Context Precision: los chunks relevantes estan en los primeros resultados?
4. Context Recall: el contexto recuperado contiene info necesaria para responder?

Uso:
    cd governance-agent
    uv run python tests/eval_rag.py [--top-k 5] [--output tests/eval_results.json]
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Asegurar que src/ es importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.groq_client import call_groq
from src.normative_rag import NormativeRAG


# ─── Metricas LLM-as-judge ───────────────────────────────────────────

def score_faithfulness(answer: str, contexts: list[str]) -> float:
    """Faithfulness: que fraccion de claims en la respuesta estan respaldadas por el contexto."""
    if not answer.strip():
        return 0.0
    context_text = "\n---\n".join(contexts)
    prompt = f"""Eres un evaluador de sistemas RAG. Tu trabajo es medir "faithfulness":
que porcentaje de las afirmaciones (claims) en la respuesta estan respaldadas
por el contexto recuperado.

Pregunta implicita: La respuesta solo dice lo que el contexto dice?

Contexto recuperado:
{context_text}

Respuesta a evaluar:
{answer}

Instrucciones:
1. Extrae cada afirmacion independiente en la respuesta.
2. Para cada una, decide si esta respaldada por el contexto (SI/NO).
3. Calcula: claims_respaldadas / total_claims.
4. Retorna SOLO un numero entre 0.0 y 1.0.

Numero:"""
    try:
        result = call_groq(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2000,
            json_mode=False,
        )
        # Extraer primer numero de la respuesta
        for line in result.strip().split("\n"):
            line = line.strip()
            if line and line[0].isdigit():
                score = float(line.split()[0])
                return max(0.0, min(1.0, score))
        return 0.0
    except Exception:
        return 0.0


def score_answer_relevancy(question: str, answer: str) -> float:
    """Answer Relevancy: la respuesta aborda directamente la pregunta?"""
    if not answer.strip():
        return 0.0
    prompt = f"""Eres un evaluador de sistemas RAG. Mide "answer relevancy":
que tan relevante es la respuesta respecto a la pregunta.

Pregunta: {question}
Respuesta: {answer}

Criterios:
- 1.0: La respuesta aborda la pregunta directamente y completamente.
- 0.5: La respuesta aborda parcialmente la pregunta.
- 0.0: La respuesta no tiene relacion con la pregunta.

Retorna SOLO un numero entre 0.0 y 1.0.

Numero:"""
    try:
        result = call_groq(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2000,
            json_mode=False,
        )
        for line in result.strip().split("\n"):
            line = line.strip()
            if line and line[0].isdigit():
                score = float(line.split()[0])
                return max(0.0, min(1.0, score))
        return 0.0
    except Exception:
        return 0.0


def score_context_precision(question: str, contexts: list[str], ground_truth: str) -> float:
    """Context Precision: los chunks relevantes estan en los primeros resultados?"""
    if not contexts:
        return 0.0
    context_list = "\n".join(f"[{i+1}] {c[:300]}" for i, c in enumerate(contexts))
    prompt = f"""Eres un evaluador de sistemas RAG. Mide "context precision":
que tan relevantes son los chunks recuperados para responder la pregunta.

Pregunta: {question}
Respuesta esperada (ground truth): {ground_truth}

Chunks recuperados (ordenados por score, 1 = mas relevante):
{context_list}

Instrucciones:
1. Para cada chunk, decide si contiene informacion relevante para la respuesta esperada (SI/NO).
2. Calcula precision: chunks_relevantes / total_chunks.
3. Penaliza si los chunks irrelevantes estan en los primeros puestos.

Retorna SOLO un numero entre 0.0 y 1.0.

Numero:"""
    try:
        result = call_groq(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2000,
            json_mode=False,
        )
        for line in result.strip().split("\n"):
            line = line.strip()
            if line and line[0].isdigit():
                score = float(line.split()[0])
                return max(0.0, min(1.0, score))
        return 0.0
    except Exception:
        return 0.0


def score_context_recall(question: str, ground_truth: str, contexts: list[str]) -> float:
    """Context Recall: el contexto recuperado contiene la info necesaria para la respuesta esperada?"""
    if not contexts:
        return 0.0
    context_text = "\n---\n".join(contexts)
    prompt = f"""Eres un evaluador de sistemas RAG. Mide "context recall":
el contexto recuperado contiene la informacion necesaria para responder
la pregunta de forma completa?

Pregunta: {question}
Respuesta esperada (ground truth): {ground_truth}

Contexto recuperado:
{context_text}

Instrucciones:
1. Extrae cada hecho clave de la respuesta esperada.
2. Para cada uno, verifica si esta presente en el contexto (SI/NO).
3. Calcula: hechos_presentes / total_hechos.
4. Retorna SOLO un numero entre 0.0 y 1.0.

Numero:"""
    try:
        result = call_groq(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2000,
            json_mode=False,
        )
        for line in result.strip().split("\n"):
            line = line.strip()
            if line and line[0].isdigit():
                score = float(line.split()[0])
                return max(0.0, min(1.0, score))
        return 0.0
    except Exception:
        return 0.0


# ─── Generacion de respuesta ─────────────────────────────────────────

def generate_answer(question: str, contexts: list[str]) -> str:
    """Generar respuesta usando el contexto recuperado (simula el uso real del RAG)."""
    context_text = "\n---\n".join(contexts)
    prompt = f"""Responde la pregunta usando SOLO el contexto proporcionado.
Si el contexto no contiene la respuesta, di "No tengo informacion suficiente".

Contexto:
{context_text}

Pregunta: {question}

Respuesta:"""
    try:
        return call_groq(
            [{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=4000,
        ).strip()
    except Exception as e:
        return f"[ERROR generando respuesta: {e}]"


# ─── Pipeline de evaluacion ──────────────────────────────────────────

def run_evaluation(top_k: int = 5) -> dict:
    """Ejecutar evaluacion completa del RAG contra el golden set."""
    golden_path = Path(__file__).parent / "golden_set.json"
    golden = json.loads(golden_path.read_text(encoding="utf-8"))

    print(f"Cargando RAG...")
    rag = NormativeRAG()
    stats = rag.stats()
    print(f"  Corpus: {stats['total_chunks']} chunks, {stats['total_sources']} fuentes")
    print(f"  Evaluando {len(golden)} preguntas con top_k={top_k}")
    print()

    results = []
    for i, item in enumerate(golden):
        qid = item["id"]
        question = item["question"]
        ground_truth = item["ground_truth"]

        print(f"  [{i+1}/{len(golden)}] {qid}: {question[:60]}...")

        # 1. Recuperar chunks
        retrieved = rag.search(question, top_k=top_k)
        contexts = [r["text"] for r in retrieved]
        scores = [r["score"] for r in retrieved]
        sources = [r["source"] for r in retrieved]

        if not contexts:
            print(f"    SIN RESULTADOS")
            results.append({
                "id": qid,
                "question": question,
                "ground_truth": ground_truth,
                "retrieved_scores": [],
                "retrieved_sources": [],
                "answer": "",
                "metrics": {
                    "faithfulness": 0.0,
                    "answer_relevancy": 0.0,
                    "context_precision": 0.0,
                    "context_recall": 0.0,
                },
            })
            continue

        # 2. Generar respuesta con los chunks recuperados
        answer = generate_answer(question, contexts)
        print(f"    Respuesta: {answer[:80]}...")

        # 3. Evaluar con 4 metricas (sleep 3s entre calls para evitar rate limit)
        time.sleep(3)
        faith = score_faithfulness(answer, contexts)
        time.sleep(3)
        rel = score_answer_relevancy(question, answer)
        time.sleep(3)
        cprec = score_context_precision(question, contexts, ground_truth)
        time.sleep(3)
        crec = score_context_recall(question, ground_truth, contexts)

        print(f"    Faithfulness={faith:.2f} Relevancy={rel:.2f} C.Precision={cprec:.2f} C.Recall={crec:.2f}")

        results.append({
            "id": qid,
            "question": question,
            "ground_truth": ground_truth,
            "retrieved_scores": [round(s, 3) for s in scores],
            "retrieved_sources": sources,
            "answer": answer,
            "metrics": {
                "faithfulness": round(faith, 3),
                "answer_relevancy": round(rel, 3),
                "context_precision": round(cprec, 3),
                "context_recall": round(crec, 3),
            },
        })

    # Agregados
    n = len(results)
    agg = {
        "faithfulness": round(sum(r["metrics"]["faithfulness"] for r in results) / n, 3) if n else 0,
        "answer_relevancy": round(sum(r["metrics"]["answer_relevancy"] for r in results) / n, 3) if n else 0,
        "context_precision": round(sum(r["metrics"]["context_precision"] for r in results) / n, 3) if n else 0,
        "context_recall": round(sum(r["metrics"]["context_recall"] for r in results) / n, 3) if n else 0,
    }

    report = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "top_k": top_k,
            "corpus_chunks": stats["total_chunks"],
            "corpus_sources": stats["total_sources"],
            "golden_set_size": len(golden),
            "model": "gpt-oss-120b",
        },
        "aggregate": agg,
        "per_question": results,
    }

    return report


def print_report(report: dict):
    """Imprimir reporte resumido en consola."""
    agg = report["aggregate"]
    print()
    print("=" * 60)
    print("REPORTE DE EVALUACION RAG")
    print("=" * 60)
    print(f"  Timestamp: {report['timestamp']}")
    print(f"  Corpus: {report['config']['corpus_chunks']} chunks, {report['config']['corpus_sources']} fuentes")
    print(f"  Golden set: {report['config']['golden_set_size']} preguntas")
    print(f"  Top-K: {report['config']['top_k']}")
    print()
    print("METRICAS AGREGADAS (promedio):")
    print(f"  Faithfulness:       {agg['faithfulness']:.3f}  (claims respaldadas por contexto)")
    print(f"  Answer Relevancy:   {agg['answer_relevancy']:.3f}  (respuesta aborda la pregunta)")
    print(f"  Context Precision:  {agg['context_precision']:.3f}  (chunks relevantes bien rankeados)")
    print(f"  Context Recall:     {agg['context_recall']:.3f}  (contexto tiene info necesaria)")
    print()
    print("DETALLE POR PREGUNTA:")
    print(f"  {'ID':<6} {'Faith':>7} {'Relev':>7} {'C.Prec':>7} {'C.Rec':>7}  Pregunta")
    print(f"  {'-'*6} {'-'*7} {'-'*7} {'-'*7} {'-'*7}  {'-'*40}")
    for r in report["per_question"]:
        m = r["metrics"]
        q = r["question"][:40]
        print(f"  {r['id']:<6} {m['faithfulness']:>7.2f} {m['answer_relevancy']:>7.2f} {m['context_precision']:>7.2f} {m['context_recall']:>7.2f}  {q}")
    print()
    print("INTERPRETACION:")
    print("  >0.8 = bueno | 0.5-0.8 = mejorable | <0.5 = problema")
    print("  Si Context Recall < 0.7: el retriever no encuentra los chunks correctos")
    print("  Si Context Precision < 0.7: hay ruido (chunks irrelevantes en top results)")
    print("  Si Faithfulness < 0.8: la respuesta inventa cosas no en el contexto")
    print("  Si Answer Relevancy < 0.7: la respuesta no aborda la pregunta")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluar RAG documental con metricas estilo RAGAS")
    parser.add_argument("--top-k", type=int, default=5, help="Numero de chunks a recuperar (default: 5)")
    parser.add_argument("--output", type=str, default="tests/eval_results.json", help="Archivo de salida JSON")
    args = parser.parse_args()

    report = run_evaluation(top_k=args.top_k)
    print_report(report)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResultados guardados en: {output_path}")


if __name__ == "__main__":
    main()
