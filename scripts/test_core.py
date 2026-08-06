"""Test rápido del núcleo: cargar VRP pack y validar datos de ejemplo."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.domain_pack import PackLoader
from src.core.pack_memory import PackMemory
from src.core.human_loop import HumanInTheLoop, Question, QuestionLevel
from src.core.validator import ValidationEngine

# 1. Cargar pack VRP
pack = PackLoader.from_yaml("src/domain_packs/vrp/pack.yaml")
print(f"=== Pack: {pack.name} v{pack.version} ===")
print(f"Reglas semánticas: {len(pack.semantic_rules)}")
print(f"Mapeos: {len(pack.inference_mappings)}")
print(f"Validadores custom: {len(pack.custom_validators)}")
print(f"Metadata area: {pack.metadata.get('area_of_operation', {})}")
print()

# 2. Crear memoria y HITL
memory = PackMemory("vrp")
hitl = HumanInTheLoop(pack_memory=memory)

# 3. Crear engine de validación (sin LLM por ahora)
engine = ValidationEngine(pack=pack, pack_memory=memory, hitl=hitl)

# 4. Datos de ejemplo (simulando datos crudos con problemas)
test_data = {
    "locations": [
        {
            "id": "depot1",
            "name": "Depósito Principal",
            "coords": [-74.07, 4.71],
            "type": "depot",
            "service_time": 0,
            "time_window_start": 28800,
            "time_window_end": 72000,
            "weight_demand": 0,
            "volume_demand": 0,
        },
        {
            "id": "c1",
            "name": "Cliente 1",
            "coords": [-74.05, 4.72],
            "type": "delivery",
            "service_time": 300,
            "time_window_start": 36000,
            "time_window_end": 68400,
            "weight_demand": -10,
            "volume_demand": -0.5,
            "required_skills": [],
        },
    ],
    "vehicles": [
        {
            "id": "van_1",
            "name": "Van",
            "start_location_id": "depot1",
            "end_location_id": "depot1",
            "start_time": 28800,
            "end_time": 72000,
            "weight_capacity": 100,
            "volume_capacity": 10,
            "skills": [],
        },
    ],
    "pickups_deliveries": [],
}

# 5. Validar
result = engine.validate(test_data)
print(f"=== Resultado ===")
print(f"Válido: {result.is_valid}")
print(f"Issues: {len(result.issues)}")
for issue in result.issues:
    print(f"  [{issue.layer}] [{issue.severity}] {issue.field_name}: {issue.message}")
print(f"Acciones: {len(result.actions)}")
for action in result.actions:
    print(f"  {action}")
print(f"Auto-correcciones: {len(result.auto_corrections)}")
print(f"Preguntas humanas: {len(result.human_questions)}")
print()

# 6. Probar memoria
print(f"=== Memoria ===")
stats = memory.get_stats()
print(f"Stats: {stats}")

# 7. Probar HITL
print(f"=== HITL ===")
summary = hitl.get_summary()
print(f"Summary: {summary}")

print("\n✓ Núcleo funcionando correctamente")
