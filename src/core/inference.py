"""
Inference engine del núcleo abstracto.

Re-exporta src/inference.py que tiene 3 mecanismos abstractos:
1. Patrones de tipo semántico (regex/heurísticas)
2. Listas de referencia (soft standards)
3. Huella de valores (overlap coefficient)

Los patrones específicos de El Salvador (DUI, NIT) son ejemplos que
pueden ser reemplazados o extendidos via Domain Packs con sus propias
listas de referencia en reference_lists/.

El mecanismo es 100% reutilizable — los patrones son configuración, no código.
"""

from src.inference import (
    InferenceResult,
    infer_semantic_type,
    clear_reference_cache,
    _detect_semantic_type,
    _match_reference_list,
    _fingerprint_match,
    _normalize,
    _normalize_set,
)

__all__ = [
    "InferenceResult",
    "infer_semantic_type",
    "clear_reference_cache",
    "detect_semantic_type",
    "match_reference_list",
    "fingerprint_match",
    "normalize",
    "normalize_set",
]

# Aliases públicos
detect_semantic_type = _detect_semantic_type
match_reference_list = _match_reference_list
fingerprint_match = _fingerprint_match
normalize = _normalize
normalize_set = _normalize_set
