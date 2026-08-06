"""Test del núcleo con datos problemáticos para verificar validadores custom."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.domain_pack import PackLoader
from src.core.pack_memory import PackMemory
from src.core.human_loop import HumanInTheLoop
from src.core.validator import ValidationEngine

pack = PackLoader.from_yaml("src/domain_packs/vrp/pack.yaml")
memory = PackMemory("vrp")
hitl = HumanInTheLoop(pack_memory=memory)
engine = ValidationEngine(pack=pack, pack_memory=memory, hitl=hitl)

# Datos con problemas: coordenada fuera de área, ventana horaria invertida, desbalance pick-delivery
test_data = {
    "locations": [
        {
            "id": "depot1",
            "name": "Depósito",
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
            "name": "Cliente lejano",
            "coords": [-70.0, 10.0],  # fuera de Bogotá
            "type": "delivery",
            "service_time": 300,
            "time_window_start": 68400,  # invertido: start > end
            "time_window_end": 36000,
            "weight_demand": -10,
            "volume_demand": -0.5,
        },
        {
            "id": "c2",
            "name": "Cliente nocturno",
            "coords": [-74.05, 4.72],
            "type": "delivery",
            "service_time": 300,
            "time_window_start": 79200,  # 22:00 - atípico
            "time_window_end": 82800,    # 23:00
            "weight_demand": -5,
            "volume_demand": -0.3,
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
    "pickups_deliveries": [
        {"pickup": "c1", "delivery": "c2"},  # demandas no opuestas: -10 + -5 = -15
    ],
}

result = engine.validate(test_data)
print(f"=== Resultado ===")
print(f"Válido: {result.is_valid}")
print(f"Issues: {len(result.issues)}")
for issue in result.issues:
    print(f"  [{issue.layer}] [{issue.severity}] {issue.field_name}: {issue.message}")
print(f"Preguntas humanas: {len(result.human_questions)}")
for q in result.human_questions:
    print(f"  {q}")
