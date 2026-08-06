"""
Validadores custom para el Domain Pack VRP.

Estas funciones son específicas del dominio VRP y se cargan dinámicamente
desde el pack.yaml. Reciben los datos y el Domain Pack, y retornan
lista de ValidationIssue.

Interface: func(data: dict, pack: DomainPack) -> list[ValidationIssue]
"""

from __future__ import annotations

from typing import Any

# Import relativo al núcleo (core está en src/core, vrp_validators en src/domain_packs/vrp)
import sys
from pathlib import Path as _Path
_root_dir = str(_Path(__file__).parent.parent.parent.parent)
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

from src.core.validator import ValidationIssue


def check_coords_in_bounds(data: dict, pack) -> list[ValidationIssue]:
    """Verificar que las coordenadas estén dentro del área de operación."""
    issues = []
    bounds = pack.metadata.get("area_of_operation")
    if not bounds:
        return issues

    lat_min = bounds.get("lat_min", -90)
    lat_max = bounds.get("lat_max", 90)
    lng_min = bounds.get("lng_min", -180)
    lng_max = bounds.get("lng_max", 180)

    for loc in data.get("locations", []):
        coords = loc.get("coords")
        if not coords or not isinstance(coords, (list, tuple)) or len(coords) < 2:
            continue
        lat, lng = coords[0], coords[1]
        if lat < lat_min or lat > lat_max:
            issues.append(ValidationIssue(
                layer="custom", severity="warning",
                field_name=f"locations.{loc.get('id', '?')}.coords",
                issue_type="out_of_bounds",
                message=f"Latitud {lat} fuera del área de operación ({lat_min}, {lat_max})",
                original_value=coords,
            ))
        if lng < lng_min or lng > lng_max:
            issues.append(ValidationIssue(
                layer="custom", severity="warning",
                field_name=f"locations.{loc.get('id', '?')}.coords",
                issue_type="out_of_bounds",
                message=f"Longitud {lng} fuera del área de operación ({lng_min}, {lng_max})",
                original_value=coords,
            ))

    return issues


def check_time_window_overlap(data: dict, pack) -> list[ValidationIssue]:
    """Verificar coherencia básica de ventanas de tiempo."""
    issues = []
    typical = pack.metadata.get("typical_service_hours", {})
    typical_start = typical.get("start", 0)
    typical_end = typical.get("end", 86400)

    for loc in data.get("locations", []):
        loc_id = loc.get("id", "?")

        # time_window_start y time_window_end (formato legacy)
        tw_start = loc.get("time_window_start")
        tw_end = loc.get("time_window_end")

        if tw_start is not None and tw_end is not None:
            if tw_end < tw_start:
                issues.append(ValidationIssue(
                    layer="custom", severity="error",
                    field_name=f"locations.{loc_id}.time_window",
                    issue_type="time_window_invalid",
                    message=f"Ubicación '{loc_id}' tiene time_window_end ({tw_end}) < time_window_start ({tw_start})",
                ))

            # Verificar si está fuera de horas típicas
            if tw_start < typical_start or tw_end > typical_end:
                issues.append(ValidationIssue(
                    layer="custom", severity="warning",
                    field_name=f"locations.{loc_id}.time_window",
                    issue_type="atypical_hours",
                    message=f"Ubicación '{loc_id}' ventana {tw_start}-{tw_end} fuera de horas típicas ({typical_start}-{typical_end})",
                ))

        # time_windows (formato lista de TimeWindow)
        time_windows = loc.get("time_windows")
        if time_windows and isinstance(time_windows, list):
            for i, tw in enumerate(time_windows):
                if isinstance(tw, dict):
                    tw_s = tw.get("start")
                    tw_e = tw.get("end")
                    if tw_s is not None and tw_e is not None and tw_e < tw_s:
                        issues.append(ValidationIssue(
                            layer="custom", severity="error",
                            field_name=f"locations.{loc_id}.time_windows[{i}]",
                            issue_type="time_window_invalid",
                            message=f"Ubicación '{loc_id}' time_windows[{i}] tiene end < start",
                        ))

    # Verificar horarios de vehículos
    for veh in data.get("vehicles", []):
        veh_id = veh.get("id", "?")
        start_time = veh.get("start_time")
        end_time = veh.get("end_time")
        if start_time is not None and end_time is not None:
            if end_time <= start_time:
                issues.append(ValidationIssue(
                    layer="custom", severity="error",
                    field_name=f"vehicles.{veh_id}.schedule",
                    issue_type="vehicle_schedule_invalid",
                    message=f"Vehículo '{veh_id}' tiene end_time ({end_time}) <= start_time ({start_time})",
                ))

    return issues


def check_pickup_delivery_balance(data: dict, pack) -> list[ValidationIssue]:
    """Verificar que los pares pickup-delivery tengan demandas opuestas."""
    issues = []
    pairs = data.get("pickups_deliveries") or data.get("pickup_delivery_pairs") or []
    locations = {loc.get("id"): loc for loc in data.get("locations", []) if loc.get("id")}

    for i, pair in enumerate(pairs):
        pickup_id = pair.get("pickup") or pair.get("pickup_id")
        delivery_id = pair.get("delivery") or pair.get("delivery_id")

        if not pickup_id or not delivery_id:
            continue
        if pickup_id not in locations or delivery_id not in locations:
            continue

        pickup = locations[pickup_id]
        delivery = locations[delivery_id]

        # Verificar demandas opuestas para weight y volume
        for prefix in ["weight", "volume"]:
            p_key = f"{prefix}_demand"
            p_val = pickup.get(p_key, 0)
            d_val = delivery.get(p_key, 0)
            if p_val != 0 or d_val != 0:
                if p_val + d_val != 0:
                    issues.append(ValidationIssue(
                        layer="custom", severity="warning",
                        field_name=f"pickups_deliveries[{i}]",
                        issue_type="demand_imbalance",
                        message=f"Par {pickup_id}→{delivery_id}: {p_key} no balanceada ({p_val}+{d_val}={p_val + d_val}, esperado 0)",
                    ))

    return issues
