"""Generar CSVs sinteticos realistas para las 6 fuentes del grafo combinado.

Cada CSV tiene distribuciones que reflejan los quality scores del grafo:
- hospital: buena calidad, pocos nulos
- censo: calidad media-alta, casi sin nulos
- seguro: calidad media, algunos nulos
- siv: calidad baja, muchos nulos e inconsistencias
- mined: calidad media-alta
- pnud: calidad variable
"""
import csv
import random
import os
from pathlib import Path

random.seed(42)

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DIAGNOSTICOS = ["I10", "I21", "J189", "A090", "E11", "N390", "S720", "C509", "F329", "K359"]
DIAGNOSTICOS_NOMBRES = {
    "I10": "Hipertension", "I21": "Infarto agudo miocardio",
    "J189": "Neumonia", "A090": "Diarrea infecciosa",
    "E11": "Diabetes tipo 2", "N390": "Infeccion urinaria",
    "S720": "Fractura cadera", "C509": "Cancer mama",
    "F329": "Depresion", "K359": "Apendicitis",
}
MUNICIPIOS = ["SMUN01", "SMUN02", "SMUN03", "SMUN04", "SMUN05", "SMUN06",
              "SMUN07", "SMUN08", "SMUN09", "SMUN10"]
NIVELES_EDUC = ["0", "1", "2", "3", "4", "5", "6"]
ESTADOS_VAC = ["completo", "incompleto", "no_iniciado"]


def _null(prob):
    """Retorna string vacio con probabilidad dada."""
    return "" if random.random() < prob else None


def _maybe_null(val, null_prob=0.0):
    if random.random() < null_prob:
        return ""
    return val


def gen_hospital(n=500):
    rows = []
    for i in range(n):
        sexo = random.choice(["M", "F", "M", "F", "M"])  # 60% M
        edad = max(0, min(95, int(random.gauss(52, 18))))
        fecha_nac = f"{2025 - edad}-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
        cie10 = random.choice(DIAGNOSTICOS)
        f_ingreso = f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
        escolaridad = _maybe_null(random.choice(NIVELES_EDUC), 0.40)  # 40% nulos
        peso = round(random.gauss(3200, 500), 0) if random.random() < 0.15 else ""
        municipio = _maybe_null(random.choice(MUNICIPIOS), 0.15)

        rows.append({
            "edad": str(edad),
            "sexo": sexo,
            "cie10": cie10,
            "fecha_nac": fecha_nac,
            "f_ingreso": f_ingreso,
            "escolaridad": escolaridad,
            "peso_nacer": str(int(peso)) if peso else "",
            "municipio": municipio,
        })
    return rows


def gen_censo(n=2000):
    rows = []
    for i in range(n):
        sexo = random.choice(["M", "F"])
        edad = max(0, min(100, int(random.gauss(35, 22))))
        fecha_nac = f"{2025 - edad}-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
        municipio = _maybe_null(random.choice(MUNICIPIOS), 0.05)
        nivel_ed = _maybe_null(random.choice(NIVELES_EDUC), 0.18)
        urbano = random.choice(["U", "R", "U", "U"])  # 75% urbano
        asiste = _maybe_null(random.choice(["si", "no"]), 0.12)
        internet = _maybe_null(random.choice(["si", "no", "si"]), 0.25)

        rows.append({
            "edad": str(edad),
            "sexo": sexo,
            "municipio": municipio,
            "fecha_nacimiento": fecha_nac,
            "nivel_educativo": nivel_ed,
            "residencia_urbana": urbano,
            "asiste_escuela": asiste if edad <= 18 else "",
            "internet_hogar": internet,
        })
    return rows


def gen_seguro(n=800):
    rows = []
    for i in range(n):
        sexo = random.choice(["M", "F"])
        edad = max(0, min(90, int(random.gauss(48, 16))))
        fecha_nac = f"{2025 - edad}-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
        diag = _maybe_null(random.choice(DIAGNOSTICOS), 0.20)
        f_alta = _maybe_null(f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}", 0.15)

        rows.append({
            "sexo_paciente": sexo,
            "diag_cie": diag,
            "fecha_nacimiento": fecha_nac,
            "fecha_alta": f_alta,
        })
    return rows


def gen_siv(n=300):
    rows = []
    for i in range(n):
        edad_nino = _maybe_null(str(max(0, min(15, int(random.gauss(5, 4))))), 0.30)
        f_vac = _maybe_null(f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}", 0.35)
        estado = _maybe_null(random.choice(ESTADOS_VAC), 0.45)  # 45% nulos!

        rows.append({
            "edad_nino": edad_nino,
            "fecha_vacunacion": f_vac,
            "estado_esquema": estado,
        })
    return rows


def gen_mined(n=1200):
    rows = []
    for i in range(n):
        sexo = random.choice(["M", "F"])
        edad = max(4, min(18, int(random.gauss(12, 4))))
        grado = max(1, min(12, int(random.gauss(edad - 5, 1))))
        notas = round(random.gauss(7.5, 1.5), 1)
        notas = max(0.0, min(10.0, notas))
        escuela = f"ESC{random.randint(1,50):03d}"
        ratio = max(10, min(45, int(random.gauss(28, 8))))
        abandono = _maybe_null(random.choice(["si", "no", "no", "no"]), 0.40)
        municipio = _maybe_null(random.choice(MUNICIPIOS), 0.02)
        nivel = str(min(8, max(0, grado // 2)))

        rows.append({
            "grado": str(grado),
            "promedio_notas": f"{notas:.1f}",
            "cod_escuela": escuela,
            "alumnos_por_docente": str(ratio),
            "abandono": abandono,
            "municipio_escuela": municipio,
            "nivel_educativo": nivel,
            "edad_alumno": str(edad),
            "sexo_alumno": sexo,
        })
    return rows


def gen_pnud(n=400):
    rows = []
    for i in range(n):
        idh = round(random.gauss(0.65, 0.12), 3)
        idh = max(0.3, min(0.95, idh))
        pobreza = random.choice(["U", "R", "R", "U"])
        internet = _maybe_null(random.choice(["si", "no"]), 0.30)
        escolaridad = _maybe_null(random.choice(NIVELES_EDUC), 0.15)
        municipio = random.choice(MUNICIPIOS)

        rows.append({
            "idh_municipio": f"{idh:.3f}",
            "pobreza_multidim": pobreza,
            "acceso_internet": internet,
            "escolaridad_promedio": escolaridad,
        })
    return rows


GENERATORS = {
    "hospital.csv": gen_hospital,
    "censo.csv": gen_censo,
    "seguro.csv": gen_seguro,
    "siv.csv": gen_siv,
    "mined.csv": gen_mined,
    "pnud.csv": gen_pnud,
}

for filename, gen_fn in GENERATORS.items():
    rows = gen_fn()
    path = DATA_DIR / filename
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  {filename}: {len(rows)} filas, {len(fieldnames)} columnas -> {path}")

print(f"\nTotal: {len(GENERATORS)} CSVs en {DATA_DIR}")
