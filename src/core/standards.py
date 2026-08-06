"""
Standards del núcleo abstracto.

Re-exporta src/standards.py que ya es domain-agnostic.
Los estándares se registran dinámicamente via register_standard()
o import_catalog() cuando se entra a un dominio.

El governance agent no incluye estándares de ningún dominio por defecto.
Los Domain Packs registran sus estándares al cargarse.
"""

from src.standards import (
    STANDARDS,
    register_standard,
    unregister_standard,
    list_standards,
    import_catalog,
    get_standard_values,
    detect_standard,
)

__all__ = [
    "STANDARDS",
    "register_standard",
    "unregister_standard",
    "list_standards",
    "import_catalog",
    "get_standard_values",
    "detect_standard",
]


def register_pack_standards(pack) -> None:
    """
    Registrar estándares desde un Domain Pack.

    Si el pack tiene estándares definidos en metadata['standards'],
    los registra dinámicamente para que inference y validator los usen.
    """
    standards = pack.metadata.get("standards", {})
    for std_id, std_data in standards.items():
        register_standard(
            standard_id=std_id,
            name=std_data.get("name", std_id),
            domain=pack.name,
            standard_type=std_data.get("type", "classifier"),
            values=std_data.get("values", {}),
            regex=std_data.get("regex"),
            name_hints=std_data.get("name_hints", []),
            catalog_file=std_data.get("catalog_file"),
        )
