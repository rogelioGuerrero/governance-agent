"""
Núcleo abstracto del Governance Agent.

Componentes reutilizables independientes del dominio:
- domain_pack: Carga y validación de Domain Packs (plugins de dominio)
- pack_memory: Memoria persistente de correcciones por pack
- human_loop: Human-in-the-loop no intrusivo
- validator: Validación por capas (estructural → semántica → LLM)
- profiler: Perfilado de datos (CSV/PostgreSQL, domain-agnostic)
- inference: Inferencia sin LLM (patrones, listas de referencia, huellas)
- standards: Estándares dinámicos (registro runtime, no hardcoded)
"""

from .domain_pack import DomainPack, PackLoader, FieldSchema, SolverContract
from .pack_memory import PackMemory, CorrectionRecord, compute_error_signature
from .human_loop import HumanInTheLoop, Question, QuestionLevel
from .validator import ValidationEngine, ValidationResult, ValidationIssue
from .profiler import (
    ColumnProfile,
    TableProfile,
    profile_csv,
    profile_postgresql,
    detect_standards_for_columns,
)
from .inference import (
    InferenceResult,
    infer_semantic_type,
    clear_reference_cache,
)
from .standards import (
    STANDARDS,
    register_standard,
    unregister_standard,
    list_standards,
    import_catalog,
    get_standard_values,
    detect_standard,
    register_pack_standards,
)
from .llm_adapter import LLMAdapter, LLMResult

__all__ = [
    # Domain Pack
    "DomainPack",
    "PackLoader",
    "FieldSchema",
    "SolverContract",
    # Memory
    "PackMemory",
    "CorrectionRecord",
    "compute_error_signature",
    # Human in the loop
    "HumanInTheLoop",
    "Question",
    "QuestionLevel",
    # Validator
    "ValidationEngine",
    "ValidationResult",
    "ValidationIssue",
    # Profiler
    "ColumnProfile",
    "TableProfile",
    "profile_csv",
    "profile_postgresql",
    "detect_standards_for_columns",
    # Inference
    "InferenceResult",
    "infer_semantic_type",
    "clear_reference_cache",
    # Standards
    "STANDARDS",
    "register_standard",
    "unregister_standard",
    "list_standards",
    "import_catalog",
    "get_standard_values",
    "detect_standard",
    "register_pack_standards",
    # LLM
    "LLMAdapter",
    "LLMResult",
]
