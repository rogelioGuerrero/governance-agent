"""Tests para semantic_tools: similarity, statistics, quality, clustering, profiling, schema_match.

Todos deterministicos, sin LLM, sin scipy. Usa datos sinteticos.
"""

import pytest
from pathlib import Path

from semantic_tools.similarity import (
    cosine_similarity, jaccard_similarity, overlap_coefficient,
    tfidf_similarity, tfidf_similarity_batch, composite_similarity,
)
from semantic_tools.statistics import (
    welch_t_test, chi_square_test, pearson_correlation, spearman_correlation,
    distribution_summary, _to_float_list,
)
from semantic_tools.quality import (
    column_quality_score, detect_numeric_anomalies,
    detect_categorical_anomalies, auto_clean,
)
from semantic_tools.clustering import (
    column_profile, cluster_columns, optimal_k,
)
from semantic_tools.profiling import profile_csv, format_profile_summary
from semantic_tools.schema_match import auto_match, format_match_result


# === SIMILARITY ===

class TestCosineSimilarity:
    def test_identical_columns(self):
        vals = ["M", "F", "M", "F", "M"]
        assert cosine_similarity(vals, vals) == pytest.approx(1.0, abs=0.01)

    def test_disjoint_columns(self):
        a = ["M", "M", "M"]
        b = ["F", "F", "F"]
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=0.01)

    def test_empty_columns(self):
        assert cosine_similarity([], ["a"]) == 0.0
        assert cosine_similarity(["a"], []) == 0.0

    def test_case_insensitive(self):
        a = ["M", "F", "M"]
        b = ["m", "f", "m"]
        assert cosine_similarity(a, b) == pytest.approx(1.0, abs=0.01)


class TestJaccardSimilarity:
    def test_identical_sets(self):
        assert jaccard_similarity(["a", "b", "c"], ["a", "b", "c"]) == pytest.approx(1.0)

    def test_disjoint_sets(self):
        assert jaccard_similarity(["a", "b"], ["c", "d"]) == pytest.approx(0.0)

    def test_partial_overlap(self):
        # intersection = {a, b} = 2, union = {a, b, c, d} = 4
        assert jaccard_similarity(["a", "b", "c"], ["a", "b", "d"]) == pytest.approx(0.5)


class TestOverlapCoefficient:
    def test_subset(self):
        # |A ∩ B| / min(|A|, |B|) = 2 / 2 = 1.0
        assert overlap_coefficient(["a", "b"], ["a", "b", "c", "d"]) == pytest.approx(1.0)

    def test_empty(self):
        assert overlap_coefficient([], ["a"]) == 0.0


class TestTfidfSimilarity:
    def test_identical_texts(self):
        text = "variable canonica de sexo en salud"
        assert tfidf_similarity(text, text) == pytest.approx(1.0, abs=0.05)

    def test_unrelated_texts(self):
        a = "variable de sexo"
        b = "indicador economico de desempleo"
        assert tfidf_similarity(a, b) < 0.3

    def test_empty(self):
        assert tfidf_similarity("", "hola") == 0.0


class TestTfidfBatch:
    def test_ranking(self):
        query = "sexo"
        docs = ["variable de sexo", "indicador economico", "sexo y genero"]
        results = tfidf_similarity_batch(query, docs)
        assert len(results) == 3
        assert results[0]["score"] >= results[1]["score"]
        # "sexo y genero" o "variable de sexo" deben ser los primeros
        assert results[0]["index"] in (0, 2)


class TestCompositeSimilarity:
    def test_high_similarity(self):
        result = composite_similarity(
            values_a=["M", "F", "M", "F"],
            values_b=["M", "F", "M", "F"],
            text_a="variable de sexo",
            text_b="variable de sexo",
        )
        assert result["verdict"] == "high"
        assert result["composite"] >= 0.7

    def test_low_similarity(self):
        result = composite_similarity(
            values_a=["M", "F", "M"],
            values_b=["123", "456", "789"],
            text_a="variable de sexo",
            text_b="codigo postal",
        )
        assert result["verdict"] == "low"

    def test_returns_all_keys(self):
        result = composite_similarity(["a"], ["b"])
        for key in ("cosine", "jaccard", "overlap", "tfidf", "composite", "verdict"):
            assert key in result


# === STATISTICS ===

class TestToFloatList:
    def test_numeric_strings(self):
        assert _to_float_list(["1", "2", "3"]) == [1.0, 2.0, 3.0]

    def test_mixed(self):
        result = _to_float_list(["1", "abc", "3"])
        assert result == [1.0, 3.0]

    def test_empty(self):
        assert _to_float_list([]) == []


class TestWelchTTest:
    def test_same_distribution(self):
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = welch_t_test(a, b)
        assert result["significant"] == False
        assert result["t_stat"] == pytest.approx(0.0, abs=0.01)

    def test_different_distributions(self):
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [100.0, 200.0, 300.0, 400.0, 500.0]
        result = welch_t_test(a, b)
        assert result["significant"] == True

    def test_insufficient_data(self):
        result = welch_t_test([1.0], [2.0])
        assert result["significant"] == False


class TestChiSquareTest:
    def test_same_distribution(self):
        a = ["M", "F", "M", "F", "M", "F"]
        b = ["M", "F", "M", "F", "M", "F"]
        result = chi_square_test(a, b)
        assert result["significant"] == False

    def test_different_distribution(self):
        a = ["M"] * 100
        b = ["F"] * 100
        result = chi_square_test(a, b)
        assert result["significant"] == True

    def test_single_category(self):
        result = chi_square_test(["M", "M"], ["M", "M"])
        assert result["chi2"] == pytest.approx(0.0)


class TestPearsonCorrelation:
    def test_perfect_positive(self):
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [2.0, 4.0, 6.0, 8.0, 10.0]
        result = pearson_correlation(a, b)
        assert result["r"] == pytest.approx(1.0, abs=0.01)

    def test_perfect_negative(self):
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [5.0, 4.0, 3.0, 2.0, 1.0]
        result = pearson_correlation(a, b)
        assert result["r"] == pytest.approx(-1.0, abs=0.01)

    def test_no_correlation(self):
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [3.0, 1.0, 5.0, 2.0, 4.0]
        result = pearson_correlation(a, b)
        assert abs(result["r"]) < 0.5

    def test_insufficient_data(self):
        result = pearson_correlation([1.0], [2.0])
        assert "insuficientes" in result["verdict"]


class TestSpearmanCorrelation:
    def test_monotonic(self):
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [10.0, 20.0, 30.0, 40.0, 50.0]
        result = spearman_correlation(a, b)
        assert result["r"] == pytest.approx(1.0, abs=0.01)


class TestDistributionSummary:
    def test_numeric(self):
        vals = ["1", "2", "3", "4", "5", "5"]
        result = distribution_summary(vals)
        assert result["type"] == "numeric"
        assert result["count"] == 6
        assert result["unique"] == 5
        assert result["missing"] == 0
        assert "mean" in result
        assert "median" in result

    def test_categorical(self):
        vals = ["M", "F", "M", "F", "H"]
        result = distribution_summary(vals)
        assert result["type"] == "categorical"
        assert result["unique"] == 3
        assert "top_5" in result

    def test_with_missing(self):
        vals = ["M", "", None, "F", "N/A"]
        result = distribution_summary(vals)
        assert result["missing"] == 3
        assert result["count"] == 5


# === QUALITY ===

class TestColumnQualityScore:
    def test_perfect_column(self):
        vals = ["M", "F", "M", "F", "M", "F", "M", "F", "M", "F"]
        result = column_quality_score(vals)
        assert result["score"] >= 80
        assert result["grade"] in ("A", "B")

    def test_empty_column(self):
        result = column_quality_score([])
        assert result["score"] == 0
        assert result["grade"] == "F"

    def test_high_nullity(self):
        vals = ["M", None, None, None, None, None, None, None, None, None]
        result = column_quality_score(vals)
        assert result["completeness"] < 0.2
        assert "alta nulidad" in " ".join(result["issues"])


class TestDetectNumericAnomalies:
    def test_no_anomalies(self):
        vals = ["1", "2", "3", "4", "5", "6", "7", "8"]
        result = detect_numeric_anomalies(vals, method="iqr")
        assert result["anomaly_count"] == 0

    def test_with_outlier(self):
        vals = ["1", "2", "3", "4", "5", "6", "7", "100"]
        result = detect_numeric_anomalies(vals, method="iqr")
        assert result["anomaly_count"] >= 1
        assert result["anomalies"][0]["value"] == 100.0

    def test_insufficient_data(self):
        result = detect_numeric_anomalies(["1", "2"])
        assert result["anomaly_count"] == 0


class TestDetectCategoricalAnomalies:
    def test_normal_column(self):
        vals = ["M", "F", "M", "F", "M", "F", "M", "F", "M", "F"]
        result = detect_categorical_anomalies(vals)
        assert result["high_cardinality"] == False

    def test_high_cardinality(self):
        vals = [str(i) for i in range(100)]
        result = detect_categorical_anomalies(vals)
        assert result["high_cardinality"] == True
        assert result["cardinality"] == 100


class TestAutoClean:
    def test_clean_column(self):
        vals = ["  M  ", "F", "M", "F"]
        result = auto_clean(vals)
        assert result["total_changes"] >= 1
        assert "whitespace" in result["fixes_applied"]

    def test_encoding_fix(self):
        vals = ["a~o", "M", "F"]
        result = auto_clean(vals)
        assert result["total_changes"] >= 1
        assert "encoding" in result["fixes_applied"]

    def test_no_changes_needed(self):
        vals = ["M", "F", "M", "F"]
        result = auto_clean(vals)
        assert result["total_changes"] == 0


# === CLUSTERING ===

class TestColumnProfile:
    def test_numeric_column(self):
        vals = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
        result = column_profile(vals)
        assert result["metadata"]["type"] == "numeric"
        assert len(result["features"]) == 5

    def test_categorical_column(self):
        vals = ["M", "F", "M", "F", "M", "F", "M", "F", "M", "F"]
        result = column_profile(vals)
        assert result["metadata"]["type"] == "categorical"

    def test_empty_column(self):
        result = column_profile([])
        assert result["metadata"]["type"] == "empty"


class TestClusterColumns:
    def test_basic_clustering(self):
        columns = {
            "id": ["1", "2", "3", "4", "5"],
            "sexo": ["M", "F", "M", "F", "M"],
            "edad": ["25", "30", "35", "40", "45"],
            "nombre": ["Juan", "Maria", "Carlos", "Ana", "Pedro"],
        }
        result = cluster_columns(columns, k=2)
        assert len(result["clusters"]) == 2
        total_members = sum(len(c["members"]) for c in result["clusters"])
        assert total_members == 4

    def test_single_column(self):
        result = cluster_columns({"col": ["1", "2", "3"]}, k=1)
        assert len(result["clusters"]) == 1


class TestOptimalK:
    def test_returns_optimal_k(self):
        feature_vectors = [
            [0.9, 0.1, 0.0, 0.0, 0.1],
            [0.8, 0.2, 0.0, 0.0, 0.2],
            [0.1, 0.9, 0.0, 0.0, 0.5],
            [0.2, 0.8, 0.0, 0.0, 0.6],
        ]
        result = optimal_k(feature_vectors, k_max=4)
        assert "optimal_k" in result
        assert result["optimal_k"] >= 1


# === PROFILING ===

TESTS_DIR = Path(__file__).parent


class TestProfileCsv:
    def test_profile_sample_censo(self):
        csv_path = TESTS_DIR / "sample_censo.csv"
        result = profile_csv(str(csv_path))
        assert result["source"] == "sample_censo"
        assert result["total_rows"] == 10
        assert len(result["columns"]) == 10
        # sexo column should be categorical
        sexo_col = next(c for c in result["columns"] if c["name"] == "sexo")
        assert sexo_col["type"] == "categorical"
        assert sexo_col["unique"] == 3  # M, F, H

    def test_profile_sample_hospital(self):
        csv_path = TESTS_DIR / "sample_hospital.csv"
        result = profile_csv(str(csv_path))
        assert result["source"] == "sample_hospital"
        assert result["total_rows"] == 5
        assert len(result["columns"]) == 8

    def test_format_profile_summary(self):
        csv_path = TESTS_DIR / "sample_censo.csv"
        result = profile_csv(str(csv_path))
        text = format_profile_summary(result)
        assert "sample_censo" in text
        assert "sexo" in text


# === SCHEMA MATCH ===

class TestAutoMatch:
    def test_match_censo_hospital(self):
        csv_a = str(TESTS_DIR / "sample_censo.csv")
        csv_b = str(TESTS_DIR / "sample_hospital.csv")
        result = auto_match(csv_a, csv_b)
        assert result["total_pairs"] > 0
        assert "high_confidence" in result
        assert "medium_confidence" in result
        assert "low_confidence" in result

    def test_format_match_result(self):
        csv_a = str(TESTS_DIR / "sample_censo.csv")
        csv_b = str(TESTS_DIR / "sample_hospital.csv")
        result = auto_match(csv_a, csv_b)
        text = format_match_result(result)
        assert "Auto-match" in text
        assert "pares evaluados" in text
