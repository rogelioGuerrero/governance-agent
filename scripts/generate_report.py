#!/usr/bin/env python3
"""Genera un reporte HTML visual de validación de Governance Agent.

Toma un corte de datos con errores inyectados, ejecuta las capas de validación,
y produce un HTML con scorecard, tabla de issues, before/after y datos corregidos.

Uso:
    uv run python scripts/generate_report.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.core.domain_pack import DomainPack, FieldSchema
from src.core.validator import ValidationEngine, ValidationIssue
from src.core.pack_memory import PackMemory
from src.core.human_loop import HumanInTheLoop


def build_demo_pack() -> DomainPack:
    return DomainPack(
        name="ministerio_agricultura",
        schema_fields={
            "productor_id": FieldSchema(name="productor_id", type="string", required=True, description="ID del productor"),
            "cultivo": FieldSchema(name="cultivo", type="string", required=True, enum=["arroz", "maiz", "papa", "cafe", "cacao", "frijol"], description="Cultivo"),
            "hectareas": FieldSchema(name="hectareas", type="float", required=True, min=0.1, max=10000, description="Hectáreas sembradas"),
            "rendimiento": FieldSchema(name="rendimiento", type="float", required=True, min=0, max=100, description="Toneladas por hectárea"),
            "departamento": FieldSchema(name="departamento", type="string", required=True, description="Departamento"),
            "latitud": FieldSchema(name="latitud", type="float", required=True, min=-90, max=90, description="Latitud"),
            "longitud": FieldSchema(name="longitud", type="float", required=True, min=-180, max=180, description="Longitud"),
        },
        semantic_rules=[
            "El rendimiento por hectárea debe ser plausible (arroz: 4-8, maiz: 3-12, papa: 15-25, cafe: 0.8-2.5)",
            "La latitud y longitud deben corresponder al departamento declarado",
            "Las hectáreas no pueden exceder 500 para pequeños productores",
        ],
    )


def build_corte_con_errores() -> list[dict]:
    return [
        {
            "productor_id": "P-001",
            "cultivo": "arroz",
            "hectareas": 5.0,
            "rendimiento": 6.5,
            "departamento": "bolivar",
            "latitud": 8.88,
            "longitud": -74.78,
        },
        {
            "productor_id": "P-002",
            "cultivo": "papa",
            "hectareas": 0.05,
            "rendimiento": 50.0,
            "departamento": "bolivar",
            "latitud": 45.0,
            "longitud": -74.78,
        },
        {
            "productor_id": "P-003",
            "cultivo": "maiz",
            "hectareas": 1200.0,
            "rendimiento": 7.0,
            "departamento": "antioquia",
            "latitud": 6.25,
            "longitud": -75.56,
        },
        {
            "productor_id": "P-004",
            "cultivo": "cafe",
            "hectareas": 3.0,
            "rendimiento": 1.5,
            "departamento": "caldas",
            "latitud": 5.07,
            "longitud": -75.51,
        },
        {
            "productor_id": "P-005",
            "cultivo": "trigo",
            "hectareas": 8.0,
            "rendimiento": 2.5,
            "departamento": "cundinamarca",
            "latitud": 4.71,
            "longitud": -74.07,
        },
    ]


def severity_badge(severity: str) -> str:
    colors = {"error": "#dc2626", "warning": "#f59e0b", "info": "#3b82f6"}
    color = colors.get(severity, "#6b7280")
    return f'<span class="badge" style="background:{color}">{severity.upper()}</span>'


def layer_label(layer: str) -> str:
    labels = {"structural": "Estructural", "semantic": "Semántica IA", "custom": "Dominio"}
    return labels.get(layer, layer)


def issue_row(issue: ValidationIssue, idx: int) -> str:
    original = json.dumps(issue.original_value, ensure_ascii=False) if issue.original_value is not None else "—"
    suggested = json.dumps(issue.suggested_value, ensure_ascii=False) if issue.suggested_value is not None else "—"
    status = "Auto-corregido" if issue.suggested_value is not None else "Requiere revisión"
    status_color = "#16a34a" if issue.suggested_value is not None else "#f59e0b"

    return f"""
    <tr>
      <td>{idx}</td>
      <td>{severity_badge(issue.severity)}</td>
      <td>{layer_label(issue.layer)}</td>
      <td><code>{issue.field_name}</code></td>
      <td>{issue.issue_type}</td>
      <td>{issue.message}</td>
      <td><code>{original}</code></td>
      <td><code style="color:#16a34a">{suggested}</code></td>
      <td style="color:{status_color}">{status}</td>
    </tr>"""


def data_table(records: list[dict], issues: list[ValidationIssue]) -> str:
    if not records:
        return "<p>Sin datos</p>"

    fields = list(records[0].keys())
    issue_fields = {i.field_name for i in issues}

    header = "".join(f"<th>{f}</th>" for f in fields)
    rows = []
    for r in records:
        cells = ""
        for f in fields:
            val = r.get(f, "—")
            val_str = json.dumps(val, ensure_ascii=False) if not isinstance(val, str) else val
            bg = "#fef2f2" if f in issue_fields else ""
            cells += f'<td style="background:{bg}">{val_str}</td>' if bg else f"<td>{val_str}</td>"
        rows.append(f"<tr>{cells}</tr>")

    return f"""
    <table class="data-table">
      <thead><tr>{header}</tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>"""


def generate_html(corte: list[dict], result, pack_name: str) -> str:
    total = len(corte)
    errors = [i for i in result.issues if i.severity == "error"]
    warnings = [i for i in result.issues if i.severity == "warning"]
    infos = [i for i in result.issues if i.severity == "info"]
    auto_corrected = [i for i in result.issues if i.suggested_value is not None]
    need_review = [i for i in result.issues if i.suggested_value is None]

    score = max(0, 100 - len(errors) * 15 - len(warnings) * 5)
    score_color = "#16a34a" if score >= 80 else "#f59e0b" if score >= 50 else "#dc2626"
    score_label = "APROBADO" if score >= 80 else "REVISAR" if score >= 50 else "RECHAZADO"

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    issues_html = "".join(issue_row(i, idx) for idx, i in enumerate(result.issues, 1))
    before_html = data_table(corte, result.issues)

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Reporte de Calidad — Governance Agent</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #f8fafc; color: #1e293b; padding: 24px; }}
  .container {{ max-width: 1100px; margin: 0 auto; }}
  .header {{ background: white; border-radius: 12px; padding: 28px 32px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  .header h1 {{ font-size: 22px; color: #0f172a; margin-bottom: 4px; }}
  .header .subtitle {{ color: #64748b; font-size: 14px; }}
  .scorecard {{ display: grid; grid-template-columns: 200px 1fr; gap: 24px; align-items: center; margin-top: 20px; }}
  .score-circle {{ width: 160px; height: 160px; border-radius: 50%; display: flex; flex-direction: column; align-items: center; justify-content: center; border: 8px solid {score_color}; background: white; }}
  .score-value {{ font-size: 42px; font-weight: 700; color: {score_color}; }}
  .score-label {{ font-size: 12px; font-weight: 600; color: {score_color}; letter-spacing: 1px; }}
  .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }}
  .stat-card {{ background: #f1f5f9; border-radius: 8px; padding: 16px; text-align: center; }}
  .stat-number {{ font-size: 28px; font-weight: 700; }}
  .stat-label {{ font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 4px; }}
  .stat-errors .stat-number {{ color: #dc2626; }}
  .stat-warnings .stat-number {{ color: #f59e0b; }}
  .stat-corrected .stat-number {{ color: #16a34a; }}
  .stat-review .stat-number {{ color: #f59e0b; }}
  .section {{ background: white; border-radius: 12px; padding: 24px 28px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  .section h2 {{ font-size: 16px; color: #0f172a; margin-bottom: 16px; border-bottom: 2px solid #e2e8f0; padding-bottom: 8px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  .data-table th {{ background: #f1f5f9; padding: 10px 12px; text-align: left; font-weight: 600; color: #475569; border-bottom: 2px solid #e2e8f0; }}
  .data-table td {{ padding: 8px 12px; border-bottom: 1px solid #f1f5f9; }}
  .issues-table th {{ background: #f1f5f9; padding: 10px 12px; text-align: left; font-weight: 600; color: #475569; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }}
  .issues-table td {{ padding: 10px 12px; border-bottom: 1px solid #f1f5f9; vertical-align: top; }}
  .badge {{ color: white; padding: 3px 8px; border-radius: 4px; font-size: 10px; font-weight: 700; letter-spacing: 0.5px; }}
  code {{ font-family: 'Consolas', monospace; font-size: 12px; background: #f1f5f9; padding: 2px 6px; border-radius: 3px; }}
  .footer {{ text-align: center; color: #94a3b8; font-size: 12px; padding: 20px; }}
  .legend {{ display: flex; gap: 16px; margin-top: 12px; font-size: 12px; color: #64748b; }}
  .legend-dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 4px; }}
</style>
</head>
<body>
<div class="container">

  <div class="header">
    <h1>Reporte de Calidad de Datos</h1>
    <div class="subtitle">Corte: {pack_name} · {total} registros · Generado: {now}</div>

    <div class="scorecard">
      <div class="score-circle">
        <div class="score-value">{score}</div>
        <div class="score-label">{score_label}</div>
      </div>
      <div class="stats">
        <div class="stat-card stat-errors">
          <div class="stat-number">{len(errors)}</div>
          <div class="stat-label">Errores críticos</div>
        </div>
        <div class="stat-card stat-warnings">
          <div class="stat-number">{len(warnings)}</div>
          <div class="stat-label">Warnings</div>
        </div>
        <div class="stat-card stat-corrected">
          <div class="stat-number">{len(auto_corrected)}</div>
          <div class="stat-label">Auto-corregidos</div>
        </div>
        <div class="stat-card stat-review">
          <div class="stat-number">{len(need_review)}</div>
          <div class="stat-label">Requieren revisión</div>
        </div>
      </div>
    </div>
  </div>

  <div class="section">
    <h2>Datos del corte (campos con issues marcados en rojo)</h2>
    {before_html}
    <div class="legend">
      <span><span class="legend-dot" style="background:#fef2f2"></span>Campo con issue detectado</span>
      <span><span class="legend-dot" style="background:white;border:1px solid #e2e8f0"></span>Campo válido</span>
    </div>
  </div>

  <div class="section">
    <h2>Issues detectados ({len(result.issues)} total)</h2>
    <table class="issues-table">
      <thead>
        <tr>
          <th>#</th>
          <th>Severidad</th>
          <th>Capa</th>
          <th>Campo</th>
          <th>Tipo</th>
          <th>Mensaje</th>
          <th>Valor original</th>
          <th>Corrección</th>
          <th>Estado</th>
        </tr>
      </thead>
      <tbody>
        {issues_html}
      </tbody>
    </table>
  </div>

  <div class="footer">
    Generado por Governance Agent · Código para el Desarrollo · BID
  </div>

</div>
</body>
</html>"""


def main():
    print("=" * 60)
    print("  Governance Agent — Generador de Reporte HTML")
    print("=" * 60)

    pack = build_demo_pack()
    memory = PackMemory("ministerio_agricultura")
    hitl = HumanInTheLoop(pack_memory=memory)
    engine = ValidationEngine(pack=pack, pack_memory=memory, hitl=hitl)

    corte = build_corte_con_errores()

    print(f"\n  Corte: {len(corte)} registros")
    print(f"  Pack: {pack.name}")
    print(f"  Campos: {len(pack.schema_fields)}")

    all_issues = []
    for idx, record in enumerate(corte):
        result = engine.validate(record)
        for issue in result.issues:
            issue.context = f"registro {idx + 1}"
            all_issues.append(issue)

    from dataclasses import dataclass

    @dataclass
    class CombinedResult:
        is_valid: bool = True
        issues: list = None

        def __post_init__(self):
            if self.issues is None:
                self.issues = []

    combined = CombinedResult(is_valid=all(len(i.severity) == 0 or i.severity != "error" for i in all_issues), issues=all_issues)

    html = generate_html(corte, combined, pack.name)

    output_path = root_dir / "docs" / "reporte_calidad.html"
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(html, encoding="utf-8")

    print(f"\n  Reporte generado: {output_path}")
    print(f"  Issues totales: {len(all_issues)}")
    print(f"\n  Abre el archivo en tu navegador para ver el reporte.")


if __name__ == "__main__":
    main()
