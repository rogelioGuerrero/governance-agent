"""semantic_tools: herramientas deterministicas para analisis semantico y estadistico.

Modulos:
- similarity: cosine, jaccard, overlap, tfidf, composite
- statistics: welch t-test, chi-square, pearson, spearman, distribution summary
- clustering: k-means sobre perfiles de columnas, optimal_k (elbow)
- quality: anomaly detection (IQR/zscore), auto_clean, quality score
- profiling: profile_csv, format_profile_summary
- schema_match: auto_match (batch column matching), format_match_result

Todas las funciones son deterministicas (sin LLM) y usan solo stdlib de Python.
"""
