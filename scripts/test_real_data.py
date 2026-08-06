"""
Test del governance agent con datos REALES del solver VRP.

Usa los fixtures existentes en D:\\codebase\\vrp-solver\\tests\\fixtures\\:
- coords_bogota_6.json: 6 puntos reales en Bogotá (1 depósito + 5 entregas)
- coords_madrid_15.json: 15 puntos reales en Madrid (1 depósito + 14 entregas)

Construye requests completos como los tests del solver (test_solver.py)
y los pasa por el governance agent para validación.
"""
import json
import sys
from pathlib import Path

# Setup path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.domain_pack import PackLoader
from src.core.pack_memory import PackMemory
from src.core.human_loop import HumanInTheLoop
from src.core.validator import ValidationEngine

# Cargar pack VRP
pack = PackLoader.from_yaml(str(project_root / "src" / "domain_packs" / "vrp" / "pack.yaml"))

# Cargar fixtures reales
VRP_FIXTURES = Path("D:/codebase/vrp-solver/tests/fixtures")

with open(VRP_FIXTURES / "coords_bogota_6.json") as f:
    bogota_coords = json.load(f)["coords"]

with open(VRP_FIXTURES / "coords_madrid_15.json") as f:
    madrid_coords = json.load(f)["coords"]

print(f"=== Datos reales cargados ===")
print(f"Bogotá: {len(bogota_coords)} puntos (formato [lat, lng])")
print(f"Madrid: {len(madrid_coords)} puntos (formato [lat, lng])")
print()


def build_bogota_request():
    """Construir request con datos reales de Bogotá (como test_solver.py)."""
    locations = [
        {
            "id": "depot",
            "name": "Depósito Bogotá",
            "coords": bogota_coords[0],  # [4.65, -74.1]
            "type": "depot",
            "service_time": 0,
            "time_window_start": 28800,
            "time_window_end": 72000,
            "weight_demand": 0,
            "volume_demand": 0,
        }
    ]
    for i in range(1, len(bogota_coords)):
        locations.append({
            "id": f"del_{i}",
            "name": f"Entrega Bogotá {i}",
            "coords": bogota_coords[i],
            "type": "delivery",
            "service_time": 300,
            "time_window_start": 36000,
            "time_window_end": 68400,
            "weight_demand": 10.0,
            "volume_demand": 0.5,
        })
    vehicles = [
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
    ]
    return {"locations": locations, "vehicles": vehicles, "pickups_deliveries": []}


def build_madrid_request():
    """Construir request con datos reales de Madrid (como test_solver.py)."""
    locations = [
        {
            "id": "depot",
            "name": "Depósito Central Madrid",
            "coords": madrid_coords[0],  # [40.4168, -3.7038]
            "type": "depot",
            "service_time": 0,
            "time_window_start": 28800,
            "time_window_end": 72000,
            "weight_demand": 0,
            "volume_demand": 0,
        }
    ]
    for i in range(1, len(madrid_coords)):
        locations.append({
            "id": f"del_{i}",
            "name": f"Entrega Madrid {i}",
            "coords": madrid_coords[i],
            "type": "delivery",
            "service_time": 300,
            "time_window_start": 36000,
            "time_window_end": 68400,
            "weight_demand": 10.0,
            "volume_demand": 0.5,
        })
    vehicles = [
        {
            "id": "veh_1",
            "name": "Vehículo 1",
            "start_location_id": "depot",
            "end_location_id": "depot",
            "start_time": 28800,
            "end_time": 72000,
            "weight_capacity": 200.0,
            "volume_capacity": 20.0,
            "skills": [],
        }
    ]
    return {"locations": locations, "vehicles": vehicles, "pickups_deliveries": []}


def build_madrid_pickup_delivery():
    """Construir request con pares pickup-delivery (como test_solver.py)."""
    locations = [
        {
            "id": "depot",
            "name": "Depósito Central Madrid",
            "coords": madrid_coords[0],
            "type": "depot",
            "service_time": 0,
            "time_window_start": 28800,
            "time_window_end": 72000,
            "weight_demand": 0,
            "volume_demand": 0,
        },
        {
            "id": "pickup_1",
            "name": "Recogida Salamanca",
            "coords": madrid_coords[1],
            "type": "pickup",
            "service_time": 300,
            "weight_demand": 15.0,
            "volume_demand": 1.0,
        },
        {
            "id": "delivery_1",
            "name": "Entrega Retiro",
            "coords": madrid_coords[8],
            "type": "delivery",
            "service_time": 300,
            "weight_demand": -15.0,
            "volume_demand": -1.0,
        },
        {
            "id": "pickup_2",
            "name": "Recogida Tetuán",
            "coords": madrid_coords[7],
            "type": "pickup",
            "service_time": 300,
            "weight_demand": 20.0,
            "volume_demand": 1.5,
        },
        {
            "id": "delivery_2",
            "name": "Entrega Vallecas",
            "coords": madrid_coords[4],
            "type": "delivery",
            "service_time": 300,
            "weight_demand": -20.0,
            "volume_demand": -1.5,
        },
    ]
    vehicles = [
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
    ]
    pairs = [
        {"pickup": "pickup_1", "delivery": "delivery_1"},
        {"pickup": "pickup_2", "delivery": "delivery_2"},
    ]
    return {"locations": locations, "vehicles": vehicles, "pickups_deliveries": pairs}


# ═══════════════════════════════════════════════════════════════════════════
# TEST 1: Bogotá (datos dentro del área configurada)
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 1: Bogotá — 6 puntos reales (dentro del área configurada)")
print("=" * 70)

memory = PackMemory("vrp")
hitl = HumanInTheLoop(pack_memory=memory)
engine = ValidationEngine(pack=pack, pack_memory=memory, hitl=hitl)

bogota_data = build_bogota_request()
result = engine.validate(bogota_data)
print(f"Válido: {result.is_valid}")
print(f"Issues: {len(result.issues)}")
for issue in result.issues:
    print(f"  [{issue.layer}] [{issue.severity}] {issue.field_name}: {issue.message}")
print(f"Payload locations: {len(result.payload.get('locations', []))}")
print(f"Payload vehicles: {len(result.payload.get('vehicles', []))}")
print()

# ═══════════════════════════════════════════════════════════════════════════
# TEST 2: Madrid (datos FUERA del área de Bogotá configurada)
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 2: Madrid — 15 puntos reales (FUERA del área de Bogotá)")
print("=" * 70)

memory2 = PackMemory("vrp")
hitl2 = HumanInTheLoop(pack_memory=memory2)
engine2 = ValidationEngine(pack=pack, pack_memory=memory2, hitl=hitl2)

madrid_data = build_madrid_request()
result2 = engine2.validate(madrid_data)
print(f"Válido: {result2.is_valid}")
print(f"Issues: {len(result2.issues)}")
for issue in result2.issues:
    print(f"  [{issue.layer}] [{issue.severity}] {issue.field_name}: {issue.message}")
print(f"Payload locations: {len(result2.payload.get('locations', []))}")
print()

# ═══════════════════════════════════════════════════════════════════════════
# TEST 3: Madrid con pickup-delivery balanceado
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 3: Madrid pickup-delivery — pares balanceados (demandas opuestas)")
print("=" * 70)

memory3 = PackMemory("vrp")
hitl3 = HumanInTheLoop(pack_memory=memory3)
engine3 = ValidationEngine(pack=pack, pack_memory=memory3, hitl=hitl3)

pd_data = build_madrid_pickup_delivery()
result3 = engine3.validate(pd_data)
print(f"Válido: {result3.is_valid}")
print(f"Issues: {len(result3.issues)}")
for issue in result3.issues:
    print(f"  [{issue.layer}] [{issue.severity}] {issue.field_name}: {issue.message}")
print(f"Pickup-delivery pairs: {len(result3.payload.get('pickups_deliveries', []))}")
print()

# ═══════════════════════════════════════════════════════════════════════════
# TEST 4: Datos reales con problemas inyectados
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 4: Bogotá con problemas inyectados (TW invertida + desbalance)")
print("=" * 70)

bad_data = build_bogota_request()
# Invertir time_window en del_1
bad_data["locations"][1]["time_window_start"] = 68400
bad_data["locations"][1]["time_window_end"] = 36000
# Inyectar un par pickup-delivery desbalanceado
bad_data["pickups_deliveries"] = [
    {"pickup": "del_2", "delivery": "del_3"},  # 10 + 10 = 20, no balanceado
]

memory4 = PackMemory("vrp")
hitl4 = HumanInTheLoop(pack_memory=memory4)
engine4 = ValidationEngine(pack=pack, pack_memory=memory4, hitl=hitl4)

result4 = engine4.validate(bad_data)
print(f"Válido: {result4.is_valid}")
print(f"Issues: {len(result4.issues)}")
for issue in result4.issues:
    print(f"  [{issue.layer}] [{issue.severity}] {issue.field_name}: {issue.message}")
print()

# ═══════════════════════════════════════════════════════════════════════════
# RESUMEN
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("RESUMEN")
print("=" * 70)
tests = [
    ("Bogotá 6pts (limpio)", result.is_valid, len(result.issues)),
    ("Madrid 15pts (fuera área)", result2.is_valid, len(result2.issues)),
    ("Madrid pickup-delivery (balanceado)", result3.is_valid, len(result3.issues)),
    ("Bogotá con problemas inyectados", result4.is_valid, len(result4.issues)),
]
for name, valid, issues in tests:
    status = "✓ PASS" if valid else "✗ FAIL"
    print(f"  {status} — {name}: válido={valid}, issues={issues}")
