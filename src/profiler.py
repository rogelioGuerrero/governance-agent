"""
Perfilador de bases de datos.

Fase 1 del proceso: descubre tablas, columnas, tipos, distribuciones,
y detecta posibles estándares.

Soporta PostgreSQL (via psycopg2) y CSV (via librería estándar csv).
"""

import csv
import re
import os
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class ColumnProfile:
    table: str
    column: str
    data_type: str
    nullable: bool = True
    total_count: int = 0
    null_count: int = 0
    unique_count: int = 0
    sample_values: list = field(default_factory=list)
    min_value: Optional[str] = None
    max_value: Optional[str] = None
    inferred_standard: Optional[dict] = None


@dataclass
class TableProfile:
    name: str
    row_count: int = 0
    columns: list[ColumnProfile] = field(default_factory=list)


def profile_csv(file_path: str, max_rows: int = 10000) -> list[TableProfile]:
    """Perfila un archivo CSV y retorna perfiles de tabla (sin pandas)."""
    table_name = os.path.basename(file_path).replace(".csv", "")

    with open(file_path, "r", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        rows = []
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            rows.append(row)

    if not rows:
        return [TableProfile(name=table_name, row_count=0)]

    columns = list(rows[0].keys())
    table = TableProfile(name=table_name, row_count=len(rows))

    for col in columns:
        values = [row.get(col, "") for row in rows]
        non_null = [v for v in values if v and v.strip() and v.strip().lower() not in ("", "na", "n/a", "null", "none")]
        unique = set(non_null)
        samples = list(unique)[:20]

        # Detectar tipo
        data_type = "text"
        if all(re.match(r"^-?\d+$", v) for v in non_null if v):
            data_type = "integer"
        elif all(re.match(r"^-?\d+\.\d+$", v) for v in non_null if v):
            data_type = "float"
        elif all(re.match(r"^\d{4}-\d{2}-\d{2}", v) for v in non_null if v):
            data_type = "date"

        # Min/Max
        min_val = min(non_null) if non_null else None
        max_val = max(non_null) if non_null else None

        profile = ColumnProfile(
            table=table_name,
            column=col,
            data_type=data_type,
            nullable=len(non_null) < len(values),
            total_count=len(values),
            null_count=len(values) - len(non_null),
            unique_count=len(unique),
            sample_values=[str(v) for v in samples],
            min_value=str(min_val) if min_val is not None else None,
            max_value=str(max_val) if max_val is not None else None,
        )

        table.columns.append(profile)

    return [table]


def profile_postgresql(conn_string: str, schema: str = "public") -> list[TableProfile]:
    """
    Perfila todas las tablas de un schema PostgreSQL.

    Requiere psycopg2: uv add psycopg2-binary
    """
    import psycopg2
    from psycopg2.sql import SQL, Identifier, Literal

    conn = psycopg2.connect(conn_string)
    cur = conn.cursor()

    try:
        # Listar tablas
        cur.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = %s AND table_type = 'BASE TABLE'
            ORDER BY table_name
        """, (schema,))
        tables = [row[0] for row in cur.fetchall()]

        profiles = []

        for table_name in tables:
            # Contar filas
            cur.execute(SQL('SELECT COUNT(*) FROM {}.{}').format(Identifier(schema), Identifier(table_name)))
            row_count = cur.fetchone()[0]

            # Columnas
            cur.execute("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema = %s AND table_name = %s
                ORDER BY ordinal_position
            """, (schema, table_name))

            table_profile = TableProfile(name=table_name, row_count=row_count)

            for col_name, col_type, is_nullable in cur.fetchall():
                # Estadísticas por columna
                cur.execute(SQL("""
                    SELECT
                        COUNT(*),
                        COUNT({col}),
                        COUNT(DISTINCT {col})
                    FROM {schema}.{table}
                """).format(
                    col=Identifier(col_name),
                    schema=Identifier(schema),
                    table=Identifier(table_name),
                ))
                total, non_null, unique = cur.fetchone()

                # Sample values
                cur.execute(SQL("""
                    SELECT DISTINCT {col}
                    FROM {schema}.{table}
                    WHERE {col} IS NOT NULL
                    LIMIT 20
                """).format(
                    col=Identifier(col_name),
                    schema=Identifier(schema),
                    table=Identifier(table_name),
                ))
                samples = [str(row[0]) for row in cur.fetchall()]

                # Min/Max para tipos ordenables
                min_val = max_val = None
                if col_type in ("integer", "bigint", "smallint", "numeric", "decimal", "date", "timestamp without time zone"):
                    cur.execute(SQL("""
                        SELECT MIN({col}), MAX({col})
                        FROM {schema}.{table}
                        WHERE {col} IS NOT NULL
                    """).format(
                        col=Identifier(col_name),
                        schema=Identifier(schema),
                        table=Identifier(table_name),
                    ))
                    result = cur.fetchone()
                    min_val = str(result[0]) if result[0] is not None else None
                    max_val = str(result[1]) if result[1] is not None else None

                profile = ColumnProfile(
                    table=table_name,
                    column=col_name,
                    data_type=col_type,
                    nullable=(is_nullable == "YES"),
                    total_count=total,
                    null_count=total - non_null,
                    unique_count=unique,
                    sample_values=samples,
                    min_value=min_val,
                    max_value=max_val,
                )

                table_profile.columns.append(profile)

            profiles.append(table_profile)

        return profiles
    finally:
        cur.close()
        conn.close()


def detect_standards_for_columns(table: TableProfile) -> None:
    """Detectar estándares para cada columna del perfil."""
    from .standards import detect_standard

    for col in table.columns:
        candidates = detect_standard(col.column, col.sample_values)
        if candidates:
            col.inferred_standard = candidates[0]  # Mejor candidato
