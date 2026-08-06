"""Pruebas estadisticas deterministicas. Usa scipy si esta disponible, fallback a stdlib."""

import math
import statistics
from collections import Counter

try:
    from scipy import stats as _scipy_stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def _to_float_list(values: list) -> list:
    """Convertir lista de strings a lista de floats.

    Retorna solo los valores que se pueden convertir (filtra None/no-numeric).
    """
    result = []
    for v in values:
        if v is None:
            continue
        try:
            result.append(float(str(v).strip()))
        except (ValueError, TypeError):
            continue
    return result


def _to_float_list_padded(values: list, length: int) -> list:
    """Como _to_float_list pero rellena con None hasta length."""
    floats = _to_float_list(values)
    while len(floats) < length:
        floats.append(None)
    return floats[:length]


def _betacf(a: float, b: float, x: float, max_iter: int = 200, eps: float = 1e-12) -> float:
    """Continued fraction de Lentz para la funcion beta incompleta.

    Implementacion basada en Numerical Recipes.
    """
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < eps:
        d = eps
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < eps:
            d = eps
        c = 1.0 + aa / c
        if abs(c) < eps:
            c = eps
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < eps:
            d = eps
        c = 1.0 + aa / c
        if abs(c) < eps:
            c = eps
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _regularized_beta(x: float, a: float, b: float) -> float:
    """Funcion beta incompleta regularizada I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(a * math.log(x) + b * math.log(1.0 - x) + lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    else:
        return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _t_distribution_cdf(t: float, df: int) -> float:
    """CDF de la distribucion t de Student: P(T <= t).

    Usa la funcion beta incompleta regularizada.
    """
    if df <= 0:
        return 0.5
    if abs(t) < 1e-10:
        return 0.5
    if df > 200:
        return 0.5 * (1 + math.erf(t / math.sqrt(2)))
    x = df / (df + t * t)
    ibeta = _regularized_beta(x, df / 2.0, 0.5)
    if t > 0:
        return 1.0 - 0.5 * ibeta
    else:
        return 0.5 * ibeta


def welch_t_test(a: list[float], b: list[float]) -> dict:
    """Welch's t-test (varianzas desiguales).

    Retorna dict con mean_a, mean_b, std_a, std_b, t_stat, p_value, significant.
    Usa scipy.stats.ttest_ind si esta disponible, fallback a stdlib.
    """
    if len(a) < 2 or len(b) < 2:
        return {
            "mean_a": 0, "mean_b": 0, "std_a": 0, "std_b": 0,
            "t_stat": 0, "p_value": 1.0, "significant": False,
        }
    mean_a = statistics.mean(a)
    mean_b = statistics.mean(b)
    var_a = statistics.variance(a)
    var_b = statistics.variance(b)
    n_a = len(a)
    n_b = len(b)

    if _HAS_SCIPY:
        result = _scipy_stats.ttest_ind(a, b, equal_var=False)
        t_stat = float(result.statistic)
        p_value = float(result.pvalue)
        if math.isnan(p_value):
            p_value = 1.0
        return {
            "mean_a": mean_a,
            "mean_b": mean_b,
            "std_a": math.sqrt(var_a),
            "std_b": math.sqrt(var_b),
            "t_stat": t_stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
        }

    # Fallback stdlib
    se = math.sqrt(var_a / n_a + var_b / n_b)
    if se == 0:
        return {
            "mean_a": mean_a, "mean_b": mean_b,
            "std_a": math.sqrt(var_a), "std_b": math.sqrt(var_b),
            "t_stat": 0, "p_value": 1.0, "significant": False,
        }
    t_stat = (mean_a - mean_b) / se
    # Grados de libertad Welch-Satterthwaite
    num = (var_a / n_a + var_b / n_b) ** 2
    den = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    df = num / den if den > 0 else n_a + n_b - 2
    # p-value (two-tailed)
    cdf = _t_distribution_cdf(abs(t_stat), int(df))
    p_value = 2 * (1 - cdf)
    p_value = max(0.0, min(1.0, p_value))
    return {
        "mean_a": mean_a,
        "mean_b": mean_b,
        "std_a": math.sqrt(var_a),
        "std_b": math.sqrt(var_b),
        "t_stat": t_stat,
        "p_value": p_value,
        "significant": p_value < 0.05,
    }


def chi_square_test(a: list[str], b: list[str]) -> dict:
    """Chi-square test de independencia entre dos distribuciones categoricas.

    Retorna dict con chi2, p_value, significant.
    Usa scipy.stats.chi2.sf si esta disponible, fallback a Wilson-Hilferty.
    """
    ca = Counter(str(v).strip().upper() for v in a if v and str(v).strip())
    cb = Counter(str(v).strip().upper() for v in b if v and str(v).strip())
    all_cats = set(ca.keys()) | set(cb.keys())
    if len(all_cats) < 2:
        return {"chi2": 0.0, "p_value": 1.0, "significant": False}
    na = sum(ca.values())
    nb = sum(cb.values())
    n = na + nb
    if n == 0:
        return {"chi2": 0.0, "p_value": 1.0, "significant": False}
    chi2 = 0.0
    for cat in all_cats:
        observed_a = ca.get(cat, 0)
        observed_b = cb.get(cat, 0)
        expected_a = (observed_a + observed_b) * na / n
        expected_b = (observed_a + observed_b) * nb / n
        if expected_a > 0:
            chi2 += (observed_a - expected_a) ** 2 / expected_a
        if expected_b > 0:
            chi2 += (observed_b - expected_b) ** 2 / expected_b
    df = (len(all_cats) - 1) * 1  # 2 muestras -> 1 fila libre

    if _HAS_SCIPY:
        p_value = float(_scipy_stats.chi2.sf(chi2, df))
        if math.isnan(p_value):
            p_value = 1.0
    elif df > 0:
        # Aproximacion Wilson-Hilferty
        h = 2 / (9 * df)
        z = ((chi2 / df) ** (1/3) - (1 - h)) / math.sqrt(h) if h > 0 else 0
        p_value = 1 - 0.5 * (1 + math.erf(z / math.sqrt(2)))
    else:
        p_value = 1.0
    p_value = max(0.0, min(1.0, p_value))
    return {
        "chi2": chi2,
        "p_value": p_value,
        "significant": p_value < 0.05,
    }


def pearson_correlation(a: list[float], b: list[float]) -> dict:
    """Coeficiente de correlacion de Pearson."""
    n = min(len(a), len(b))
    if n < 3:
        return {"r": 0.0, "verdict": "insuficientes datos"}
    pairs = [(a[i], b[i]) for i in range(n) if a[i] is not None and b[i] is not None]
    if len(pairs) < 3:
        return {"r": 0.0, "verdict": "insuficientes datos"}
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in pairs)
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return {"r": 0.0, "verdict": "sin varianza"}
    r = num / (den_x * den_y)
    r = max(-1.0, min(1.0, r))
    abs_r = abs(r)
    if abs_r >= 0.7:
        verdict = "correlacion fuerte"
    elif abs_r >= 0.4:
        verdict = "correlacion moderada"
    else:
        verdict = "correlacion debil"
    return {"r": r, "verdict": verdict}


def spearman_correlation(a: list[float], b: list[float]) -> dict:
    """Coeficiente de correlacion de Spearman (rank-based)."""
    n = min(len(a), len(b))
    if n < 3:
        return {"r": 0.0, "verdict": "insuficientes datos"}
    pairs = [(a[i], b[i]) for i in range(n) if a[i] is not None and b[i] is not None]
    if len(pairs) < 3:
        return {"r": 0.0, "verdict": "insuficientes datos"}
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    rank_a = _rank(xs)
    rank_b = _rank(ys)
    return pearson_correlation(rank_a, rank_b)


def _rank(values: list) -> list:
    """Calcular rangos de una lista (1-based, promedio para ties)."""
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) - 1 and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def distribution_summary(values: list[str]) -> dict:
    """Resumen de distribucion de una columna.

    Retorna dict con count, unique, missing, type (numeric/categorical),
    y estadisticas adicionales segun el tipo.
    """
    total = len(values)
    non_null = [v for v in values if v and str(v).strip() and str(v).strip().lower() not in ("", "na", "n/a", "null", "none")]
    missing = total - len(non_null)
    unique_count = len(set(str(v).strip().upper() for v in non_null))

    floats = _to_float_list(non_null)
    is_numeric = len(floats) > len(non_null) * 0.8 if non_null else False

    result = {
        "count": total,
        "unique": unique_count,
        "missing": missing,
        "type": "numeric" if is_numeric else "categorical",
    }

    if is_numeric and len(floats) >= 2:
        floats_sorted = sorted(floats)
        n = len(floats_sorted)
        result.update({
            "mean": statistics.mean(floats),
            "median": statistics.median(floats),
            "std": statistics.stdev(floats) if n > 1 else 0.0,
            "min": floats_sorted[0],
            "max": floats_sorted[-1],
            "q1": _percentile(floats_sorted, 25),
            "q3": _percentile(floats_sorted, 75),
        })
    else:
        counts = Counter(non_null)
        result["top_5"] = dict(counts.most_common(5))

    return result


def _percentile(sorted_vals: list, p: float) -> float:
    """Calcular percentil de una lista ordenada."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    k = (n - 1) * p / 100.0
    f = int(k)
    c = k - f
    if f + 1 < n:
        return sorted_vals[f] + c * (sorted_vals[f + 1] - sorted_vals[f])
    return sorted_vals[f]
