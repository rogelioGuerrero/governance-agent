"""Clustering de columnas por perfil estructural. K-means en Python puro."""

import math
from collections import Counter
from .statistics import _to_float_list


def column_profile(values: list) -> dict:
    """Perfilar una columna y extraer features para clustering.

    Retorna dict con 'features' (list[float]) y 'metadata' (dict).
    Features: [numeric_ratio, cardinality_ratio, missing_ratio, has_dates, avg_length]
    """
    total = len(values) if values else 0
    if total == 0:
        return {"features": [0.0, 0.0, 0.0, 0.0, 0.0], "metadata": {"type": "empty"}}

    non_null = [v for v in values if v is not None and str(v).strip() and str(v).strip().lower() not in ("", "na", "n/a", "null", "none")]
    missing_count = total - len(non_null)
    missing_ratio = missing_count / total

    unique = set(str(v).strip().upper() for v in non_null)
    cardinality_ratio = len(unique) / total if total > 0 else 0.0

    floats = _to_float_list(non_null)
    numeric_ratio = len(floats) / total if total > 0 else 0.0

    import re
    date_count = sum(1 for v in non_null if re.match(r"\d{4}-\d{2}-\d{2}", str(v)) or re.match(r"\d{2}/\d{2}/\d{4}", str(v)))
    has_dates = date_count / total if total > 0 else 0.0

    avg_length = sum(len(str(v)) for v in non_null) / len(non_null) if non_null else 0.0
    avg_length_norm = min(avg_length / 50.0, 1.0)

    features = [numeric_ratio, cardinality_ratio, missing_ratio, has_dates, avg_length_norm]

    col_type = "numeric" if numeric_ratio > 0.7 else ("date" if has_dates > 0.5 else "categorical")

    return {
        "features": features,
        "metadata": {
            "type": col_type,
            "unique": len(unique),
            "missing_ratio": missing_ratio,
            "numeric_ratio": numeric_ratio,
        },
    }


def _euclidean(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _kmeans(data: list[list[float]], k: int, max_iter: int = 50) -> dict:
    """K-means clustering en Python puro.

    Retorna dict con clusters, centroids, iterations, inertia.
    """
    n = len(data)
    if n == 0 or k <= 0:
        return {"clusters": [], "centroids": [], "iterations": 0, "inertia": 0.0}
    k = min(k, n)

    # Inicializar centroides con k-means++ simplificado
    centroids = [data[0][:]]
    for _ in range(1, k):
        dists = [min(_euclidean(d, c) for c in centroids) for d in data]
        max_idx = dists.index(max(dists))
        centroids.append(data[max_idx][:])

    assignments = [0] * n
    iterations = 0

    for iteration in range(max_iter):
        iterations = iteration + 1
        changed = False
        for i, point in enumerate(data):
            best_k = 0
            best_dist = float("inf")
            for j, c in enumerate(centroids):
                d = _euclidean(point, c)
                if d < best_dist:
                    best_dist = d
                    best_k = j
            if assignments[i] != best_k:
                assignments[i] = best_k
                changed = True

        if not changed and iteration > 0:
            break

        # Recalcular centroides
        for j in range(k):
            members = [data[i] for i in range(n) if assignments[i] == j]
            if members:
                dim = len(centroids[j])
                centroids[j] = [sum(m[d] for m in members) / len(members) for d in range(dim)]

    # Calcular inertia
    inertia = sum(_euclidean(data[i], centroids[assignments[i]]) ** 2 for i in range(n))

    return {
        "assignments": assignments,
        "centroids": centroids,
        "iterations": iterations,
        "inertia": inertia,
    }


def cluster_columns(columns: dict, k: int = 2) -> dict:
    """Agrupar columnas por perfil estructural usando k-means.

    Args:
        columns: dict de {column_name: list_of_values}
        k: numero de clusters

    Retorna dict con clusters, iterations, inertia.
    """
    names = list(columns.keys())
    profiles = [column_profile(columns[name]) for name in names]
    feature_vectors = [p["features"] for p in profiles]

    if len(names) <= 1:
        return {
            "clusters": [{"cluster_id": 0, "members": names, "size": len(names), "centroid": profiles[0]["metadata"] if profiles else {}}],
            "iterations": 0,
            "inertia": 0.0,
        }

    k = min(k, len(names))
    result = _kmeans(feature_vectors, k)

    clusters = []
    for j in range(k):
        members = [names[i] for i in range(len(names)) if result["assignments"][i] == j]
        centroid_meta = {}
        if members:
            member_profiles = [profiles[i]["metadata"] for i in range(len(names)) if result["assignments"][i] == j]
            centroid_meta = {
                "numeric_ratio": sum(p.get("numeric_ratio", 0) for p in member_profiles) / len(member_profiles),
                "cardinality_ratio": sum(p.get("unique", 0) for p in member_profiles) / (len(member_profiles) * max(1, max(p.get("unique", 1) for p in member_profiles))),
                "missing_ratio": sum(p.get("missing_ratio", 0) for p in member_profiles) / len(member_profiles),
                "has_dates": sum(1 for p in member_profiles if p.get("type") == "date") / len(member_profiles),
            }
        clusters.append({
            "cluster_id": j,
            "members": members,
            "size": len(members),
            "centroid": centroid_meta,
        })

    return {
        "clusters": clusters,
        "iterations": result["iterations"],
        "inertia": result["inertia"],
    }


def optimal_k(feature_vectors: list[list[float]], k_max: int = 6) -> dict:
    """Encontrar k optimo via elbow method.

    Retorna dict con optimal_k, inertias.
    """
    if len(feature_vectors) <= 2 or k_max <= 2:
        return {"optimal_k": 1, "inertias": []}

    inertias = []
    ks = range(1, min(k_max, len(feature_vectors)) + 1)
    for k in ks:
        result = _kmeans(feature_vectors, k)
        inertias.append(result["inertia"])

    # Elbow: punto donde la reduccion de inertia se vuelve marginal
    if len(inertias) <= 2:
        return {"optimal_k": 1, "inertias": inertias}

    diffs = [inertias[i] - inertias[i + 1] for i in range(len(inertias) - 1)]
    if not diffs:
        return {"optimal_k": 1, "inertias": inertias}

    max_diff_idx = diffs.index(max(diffs))
    optimal_k = max_diff_idx + 2  # +2 porque el diff[i] corresponde a k=i+1 -> k=i+2

    return {"optimal_k": optimal_k, "inertias": inertias}
