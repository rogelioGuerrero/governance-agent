"""
Profiler del núcleo abstracto.

Re-exporta src/profiler.py que ya es domain-agnostic.
Perfilado de datos: descubre tablas, columnas, tipos, distribuciones.

Este módulo es 100% reutilizable — no conoce ningún dominio.
"""

from src.profiler import (
    ColumnProfile,
    TableProfile,
    profile_csv,
    profile_postgresql,
    detect_standards_for_columns,
)

__all__ = [
    "ColumnProfile",
    "TableProfile",
    "profile_csv",
    "profile_postgresql",
    "detect_standards_for_columns",
]
