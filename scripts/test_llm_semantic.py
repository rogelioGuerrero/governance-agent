"""
Test end-to-end: validación semántica con LLM real + datos reales del solver.

Usa:
- Datos reales de coords_bogota_6.json y coords_madrid_15.json
- LLM via LLMAdapter (Groq → Gemini failover)
- Las 3 capas: estructural → custom → semántica LLM

Este test verifica que la Capa 2 (semántica con LLM) funciona:
- Detecta inconsistencias lógicas que el código no puede predecir
- Sugiere correcciones
- Genera preguntas para HITL cuando hay warnings
"""
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.domain_pack import PackLoader
from src.core.pack_memory import PackMemory
from src.core.human_loop import HumanInTheLoop, Question, QuestionLevel
from src.core.validator import ValidationEngine
from src.core.llm_adapter import LLMAdapter

# Cargar pack VRP
pack = PackLoader.from_yaml(str(project_root / "src" / "domain_packs" / "vrp" / "pack.yaml"))

# Cargar datos reales
VRP_FIXTURES = Path("D:/codebase/vrp-solver/tests/fixtures")
with open(VRP_FIXTURES / "coords_bogota_6.json") as f:
    bogota_coords = json.load(f)["coords"]

print("=== Setup ===")
print(f"Pack: {pack.name}")
print(f"Reglas semánticas: {len(pack.semantic_rules)}")
print(f"Coords Bogotá: {len(bogota_coords)} puntos")
print()

# Crear LLM adapter
llm = LLMAdapter(json_mode=True, temperature=0.1, max_tokens=2000, timeout=30)
print(f"LLM adapter creado (json_mode=True, temp=0.1)")
print()

# ═══════════════════════════════════════════════════════════════════════════
# TEST 1: Datos limpios de Bogotá — el LLM no debería encontrar issues
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 1: Bogotá limpio — LLM debería confirmar que todo está OK")
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

memory = PackMemory("vrp")
hitl = HumanInTheLoop(pack_memory=memory)
engine = ValidationEngine(pack=pack, pack_memory=memory, hitl=hitl, llm_client=llm)

result = engine.validate(clean_data)
print(f"Válido: {result.is_valid}")
print(f"Total issues: {len(result.issues)}")
for issue in result.issues:
    print(f"  [{issue.layer}] [{issue.severity}] {issue.field_name}: {issue.message}")
    if issue.suggested_value:
        print(f"    → Sugerencia: {issue.suggested_value}")
print(f"Preguntas humanas: {len(result.human_questions)}")
for q in result.human_questions:
    print(f"  [{q['level']}] {q['field_name']}: {q['message']}")
    if q.get('suggested_value'):
        print(f"    → Sugerencia: {q['suggested_value']}")
print()

# ═══════════════════════════════════════════════════════════════════════════
# TEST 2: Datos con inconsistencia semántica sutil (solo LLM puede detectar)
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 2: Inconsistencia semántica sutil — solo LLM puede detectar")
print("=" * 70)

# Trucar: vehicle con end_time antes que el time_window_end más tardío
# Estructuralmente pasa (ambos son int), custom no lo valida, pero semánticamente es imposible
subtle_data = json.loads(json.dumps(clean_data))  # deep copy
subtle_data["vehicles"][0]["end_time"] = 50000  # 13:53 — antes de las entregas que cierran a las 68400 (19:00)
# Además, poner un service_time negativo (sin sentido pero pasa tipo)
subtle_data["locations"][2]["service_time"] = -100

memory2 = PackMemory("vrp")
hitl2 = HumanInTheLoop(pack_memory=memory2)
engine2 = ValidationEngine(pack=pack, pack_memory=memory2, hitl=hitl2, llm_client=llm)

result2 = engine2.validate(subtle_data)
print(f"Válido: {result2.is_valid}")
print(f"Total issues: {len(result2.issues)}")
for issue in result2.issues:
    print(f"  [{issue.layer}] [{issue.severity}] {issue.field_name}: {issue.message}")
    if issue.suggested_value:
        print(f"    → Sugerencia: {issue.suggested_value}")
print(f"Preguntas humanas: {len(result2.human_questions)}")
for q in result2.human_questions:
    print(f"  [{q['level']}] {q['field_name']}: {q['message']}")
    if q.get('suggested_value'):
        print(f"    → Sugerencia: {q['suggested_value']}")
print()

# ═══════════════════════════════════════════════════════════════════════════
# RESUMEN
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("RESUMEN")
print("=" * 70)
tests = [
    ("Bogotá limpio (LLM)", result.is_valid, len(result.issues), len(result.human_questions)),
    ("Bogotá inconsistencia sutil (LLM)", result2.is_valid, len(result2.issues), len(result2.human_questions)),
]
for name, valid, issues, questions in tests:
    print(f"  {name}: válido={valid}, issues={issues}, preguntas={questions}")
