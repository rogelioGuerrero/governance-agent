"""
Test end-to-end del orquestador VRP con datos reales.

Ejecuta el ciclo completo:
  datos crudos → governance validation → (corrección LLM si hay errores) → solver VRP

Requiere:
- Solver VRP corriendo en localhost:8000 (o ajustar SOLVER_URL)
- API keys de Groq/Gemini en .env

Si el solver no está corriendo, el test verifica que el orquestador
maneja correctamente la situación (no crash, reporta error de conexión).
"""
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.domain_pack import PackLoader
from src.core.orchestrator import VRPOrchestrator
from src.core.llm_adapter import LLMAdapter

# Cargar pack VRP
pack = PackLoader.from_yaml(str(project_root / "src" / "domain_packs" / "vrp" / "pack.yaml"))

# Cargar datos reales
VRP_FIXTURES = Path("D:/codebase/vrp-solver/tests/fixtures")
with open(VRP_FIXTURES / "coords_bogota_6.json") as f:
    bogota_coords = json.load(f)["coords"]

# URL del solver (ajustar si está en otro lado)
SOLVER_URL = "http://localhost:8000"

print("=== Setup ===")
print(f"Pack: {pack.name}")
print(f"Solver URL: {SOLVER_URL}")
print(f"Coords Bogotá: {len(bogota_coords)} puntos")
print()

# Crear orquestador
llm = LLMAdapter(json_mode=False, temperature=0.3, max_tokens=4000)
orch = VRPOrchestrator(pack=pack, solver_url=SOLVER_URL, llm_adapter=llm)

# ═══════════════════════════════════════════════════════════════════════════
# TEST 1: Datos limpios de Bogotá — debería llegar al solver
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 1: Datos limpios de Bogotá — ciclo completo")
print("=" * 70)

clean_data = {
    "locations": [
        {
            "id": "depot",
            "name": "Depósito Bogotá",
            "coords": bogota_coords[0],
            "type": "depot",
            "service_time": 0,
            "time_window_start": 28800,
            "time_window_end": 72000,
            "weight_demand": 0,
            "volume_demand": 0,
        }
    ] + [
        {
            "id": f"del_{i}",
            "name": f"Entrega Bogotá {i}",
            "coords": bogota_coords[i],
            "type": "delivery",
            "service_time": 300,
            "time_window_start": 36000,
            "time_window_end": 68400,
            "weight_demand": 10.0,
            "volume_demand": 0.5,
        }
        for i in range(1, len(bogota_coords))
    ],
    "vehicles": [
        {
            "id": "veh_1",
            "name": "Vehículo 1",
            "start_location_id": "depot",
            "end_location_id": "depot",
            "start_time": 28800,
            "end_time": 72000,
            "weight_capacity": 100.0,
            "volume_capacity": 10.0,
            "skills": [],
        }
    ],
    "pickups_deliveries": [],
}

result1 = orch.run(clean_data, auto_correct=True)
print(f"Success: {result1.success}")
print(f"Iteraciones: {result1.iterations}")
print(f"Message: {result1.message}")
print(f"Validation issues: {len(result1.validation_issues)}")
for issue in result1.validation_issues:
    print(f"  [{issue['layer']}] [{issue['severity']}] {issue['field_name']}: {issue['message']}")
print(f"Corrections applied: {len(result1.corrections_applied)}")
for c in result1.corrections_applied:
    print(f"  {c}")
print(f"Routes: {len(result1.routes)}")
if result1.routes:
    for r in result1.routes:
        print(f"  Vehículo {r.get('vehicle_id', '?')}: {r.get('total_stops', '?')} paradas, {r.get('total_distance', '?')}m")
print(f"Unassigned: {len(result1.unassigned)}")
print(f"Human questions: {len(result1.human_questions)}")
for q in result1.human_questions:
    print(f"  [{q['level']}] {q['field_name']}: {q['message']}")
if result1.solver_errors:
    print(f"Solver errors: {result1.solver_errors}")
print()

# ═══════════════════════════════════════════════════════════════════════════
# TEST 2: Datos con error crítico — orquestador intenta corregir con LLM
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 2: Datos con error crítico — auto-corrección con LLM")
print("=" * 70)

bad_data = json.loads(json.dumps(clean_data))  # deep copy
# Inyectar: service_time negativo (el LLM debería corregir a 300)
bad_data["locations"][2]["service_time"] = -100
# Inyectar: end_time del vehículo inconsistente con time_windows
bad_data["vehicles"][0]["end_time"] = 40000  # 11:06 — antes que entregas que cierran a las 68400

result2 = orch.run(bad_data, auto_correct=True)
print(f"Success: {result2.success}")
print(f"Iteraciones: {result2.iterations}")
print(f"Message: {result2.message}")
print(f"Validation issues: {len(result2.validation_issues)}")
for issue in result2.validation_issues:
    print(f"  [{issue['layer']}] [{issue['severity']}] {issue['field_name']}: {issue['message']}")
print(f"Corrections applied: {len(result2.corrections_applied)}")
for c in result2.corrections_applied:
    print(f"  {c}")
print(f"Routes: {len(result2.routes)}")
print(f"Solver errors: {result2.solver_errors}")
print()

# ═══════════════════════════════════════════════════════════════════════════
# TEST 3: Datos sin auto-corrección — debería bloquear y reportar
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 3: Datos con error, sin auto-corrección — bloquear")
print("=" * 70)

result3 = orch.run(bad_data, auto_correct=False)
print(f"Success: {result3.success}")
print(f"Iteraciones: {result3.iterations}")
print(f"Message: {result3.message}")
print(f"Validation issues: {len(result3.validation_issues)}")
for issue in result3.validation_issues:
    print(f"  [{issue['layer']}] [{issue['severity']}] {issue['field_name']}: {issue['message']}")
print(f"Solver errors: {result3.solver_errors}")
print()

# ═══════════════════════════════════════════════════════════════════════════
# RESUMEN
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("RESUMEN")
print("=" * 70)
tests = [
    ("Datos limpios", result1.success, result1.iterations, len(result1.routes)),
    ("Datos con error + auto-correct", result2.success, result2.iterations, len(result2.routes)),
    ("Datos con error sin auto-correct", result3.success, result3.iterations, len(result3.routes)),
]
for name, success, iters, routes in tests:
    status = "✓" if success else "○" if iters > 0 else "✗"
    print(f"  {status} {name}: success={success}, iteraciones={iters}, rutas={routes}")
