"""Similitud determinista entre columnas y textos. Sin LLM, sin numpy."""

import math
import re
from collections import Counter


def _value_freq(values: list[str]) -> Counter:
    return Counter(str(v).strip().upper() for v in values if v and str(v).strip())


def cosine_similarity(a: list[str], b: list[str]) -> float:
    """Cosine similarity entre dos columnas basada en frecuencia de valores.

    Trata cada columna como un vector de frecuencias de valores.
    Retorna score 0.0-1.0.
    """
    if not a or not b:
        return 0.0
    fa = _value_freq(a)
    fb = _value_freq(b)
    all_keys = set(fa.keys()) | set(fb.keys())
    if not all_keys:
        return 0.0
    dot = sum(fa.get(k, 0) * fb.get(k, 0) for k in all_keys)
    norm_a = math.sqrt(sum(v * v for v in fa.values()))
    norm_b = math.sqrt(sum(v * v for v in fb.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def jaccard_similarity(a: list[str], b: list[str]) -> float:
    """Jaccard index: |A ∩ B| / |A ∪ B| sobre conjuntos de valores."""
    if not a or not b:
        return 0.0
    sa = set(str(v).strip().upper() for v in a if v and str(v).strip())
    sb = set(str(v).strip().upper() for v in b if v and str(v).strip())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def overlap_coefficient(a: list[str], b: list[str]) -> float:
    """Overlap coefficient: |A ∩ B| / min(|A|, |B|)."""
    if not a or not b:
        return 0.0
    sa = set(str(v).strip().upper() for v in a if v and str(v).strip())
    sb = set(str(v).strip().upper() for v in b if v and str(v).strip())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / min(len(sa), len(sb))


_TF_TOKENIZE = re.compile(r"\W+")


def _tokenize(text: str) -> list[str]:
    return [t for t in _TF_TOKENIZE.split(text.lower()) if len(t) > 1]


def _tf(tokens: list[str]) -> dict[str, float]:
    counts = Counter(tokens)
    total = len(tokens) or 1
    return {t: c / total for t, c in counts.items()}


def _idf(documents: list[list[str]]) -> dict[str, float]:
    n = len(documents) or 1
    df: dict[str, int] = {}
    for doc in documents:
        for t in set(doc):
            df[t] = df.get(t, 0) + 1
    # sklearn-style smoothing: idf = log((n+1)/(df+1)) + 1, always >= 1
    return {t: math.log((n + 1) / (d + 1)) + 1 for t, d in df.items()}


def tfidf_similarity(text_a: str, text_b: str) -> float:
    """TF-IDF cosine similarity entre dos textos."""
    if not text_a or not text_b:
        return 0.0
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)
    if not tokens_a or not tokens_b:
        return 0.0
    idf = _idf([tokens_a, tokens_b])
    tf_a = _tf(tokens_a)
    tf_b = _tf(tokens_b)
    vec_a = {t: tf_a.get(t, 0) * idf.get(t, 0) for t in set(tokens_a) | set(tokens_b)}
    vec_b = {t: tf_b.get(t, 0) * idf.get(t, 0) for t in set(tokens_a) | set(tokens_b)}
    dot = sum(vec_a.get(k, 0) * vec_b.get(k, 0) for k in vec_a)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def tfidf_similarity_batch(query: str, documents: list[str]) -> list[dict]:
    """Batch TF-IDF: rankear documentos contra un query."""
    if not query or not documents:
        return []
    q_tokens = _tokenize(query)
    doc_tokens = [_tokenize(d) for d in documents]
    idf = _idf([q_tokens] + doc_tokens)
    tf_q = _tf(q_tokens)
    vec_q = {t: tf_q.get(t, 0) * idf.get(t, 0) for t in set(q_tokens)}
    norm_q = math.sqrt(sum(v * v for v in vec_q.values())) or 1.0
    results = []
    for i, dt in enumerate(doc_tokens):
        tf_d = _tf(dt)
        vec_d = {t: tf_d.get(t, 0) * idf.get(t, 0) for t in set(dt) | set(q_tokens)}
        norm_d = math.sqrt(sum(v * v for v in vec_d.values()))
        if norm_d == 0:
            results.append({"index": i, "score": 0.0})
            continue
        dot = sum(vec_q.get(k, 0) * vec_d.get(k, 0) for k in vec_q)
        results.append({"index": i, "score": dot / (norm_q * norm_d)})
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def composite_similarity(
    values_a: list[str],
    values_b: list[str],
    text_a: str = "",
    text_b: str = "",
) -> dict:
    """Similitud compuesta: combina valores (cosine, jaccard, overlap) + texto (tfidf)."""
    cos = cosine_similarity(values_a, values_b)
    jac = jaccard_similarity(values_a, values_b)
    ovl = overlap_coefficient(values_a, values_b)
    tfidf = tfidf_similarity(text_a, text_b) if text_a and text_b else 0.0

    value_score = (cos + jac + ovl) / 3.0
    composite = value_score * 0.7 + tfidf * 0.3

    if composite >= 0.7:
        verdict = "high"
    elif composite >= 0.4:
        verdict = "medium"
    else:
        verdict = "low"

    return {
        "cosine": round(cos, 4),
        "jaccard": round(jac, 4),
        "overlap": round(ovl, 4),
        "tfidf": round(tfidf, 4),
        "composite": round(composite, 4),
        "verdict": verdict,
    }
