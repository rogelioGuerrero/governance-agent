"""Generate ministerio_economia_sv.csv with same data as us_economic_indicators
but with Spanish column names and different conventions.

This simulates a real interoperability scenario: two sources describing
the same economic concepts with different naming conventions.
"""
import csv
from pathlib import Path

src = Path(r"d:\proyectoBolt\governance-agent\data\real\us_economic_indicators.csv")
dst = Path(r"d:\proyectoBolt\governance-agent\data\real\ministerio_economia_sv.csv")

# Mapping: English name -> Spanish name (different convention)
COLUMN_MAP = {
    "date": "fecha_observacion",
    "unemployment_rate": "tasa_desempleo_pct",
    "cpi_all_urban": "ipc_urbano",
    "federal_funds_rate": "tasa_fondos_federales",
    "personal_savings_rate": "ahorro_personal_tasa",
    "labor_force_participation": "participacion_laboral",
    "mortgage_rate_30yr": "tasa_hipoteca_30a",
    "real_disposable_income": "ingreso_disponible_real",
    "consumer_sentiment": "indice_confianza_consumidor",
    "industrial_production": "produccion_industrial",
}

with open(src, "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Write with Spanish column names
with open(dst, "w", encoding="utf-8", newline="") as f:
    new_fields = [COLUMN_MAP.get(h, h) for h in reader.fieldnames]
    writer = csv.DictWriter(f, fieldnames=new_fields)
    writer.writeheader()
    for row in rows:
        new_row = {}
        for old_key, new_key in COLUMN_MAP.items():
            val = row.get(old_key, "")
            # Simulate slight rounding differences (another source rounds differently)
            if val and old_key in ("cpi_all_urban", "real_disposable_income", "industrial_production"):
                try:
                    new_row[new_key] = str(round(float(val), 1))
                except ValueError:
                    new_row[new_key] = val
            else:
                new_row[new_key] = val
        writer.writerow(new_row)

print(f"Created: {dst}")
print(f"Rows: {len(rows)}")
print(f"Columns: {new_fields}")
