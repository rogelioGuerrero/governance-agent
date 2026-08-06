"""Download real economic data from FRED and World Bank, prepare as wide-format CSVs.

Creates two CSVs suitable for the governance-agent profiler:
1. us_economic_indicators.csv — monthly US data (unemployment, CPI, fed funds, etc.)
2. global_economic_indicators.csv — yearly country-level data (GDP, unemployment, inflation, population)

Usage:
    uv run python scripts/download_real_data.py
"""

import csv
import io
import json
import os
import sys
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "real"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- FRED: direct CSV download (no API key needed) ---
FRED_SERIES = {
    "UNRATE": "unemployment_rate",
    "CPIAUCSL": "cpi_all_urban",
    "FEDFUNDS": "federal_funds_rate",
    "PSAVERT": "personal_savings_rate",
    "CIVPART": "labor_force_participation",
    "A191RD3Q086SBEA": "disposable_personal_income",
    "MORTGAGE30US": "mortgage_rate_30yr",
    "DSPIC96": "real_disposable_income",
    "UMCSENT": "consumer_sentiment",
    "INDPRO": "industrial_production",
}


def download_fred_csv(series_id: str) -> dict[str, float]:
    """Download a FRED series as CSV and return {date: value}."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    print(f"  Downloading {series_id} from FRED...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read().decode("utf-8")
    except Exception as e:
        print(f"    WARNING: {series_id} failed: {e}")
        return {}

    reader = csv.DictReader(io.StringIO(data))
    result = {}
    for row in reader:
        date = row.get("observation_date", row.get("DATE", ""))
        val = row.get(series_id, "")
        if val and val != ".":
            try:
                result[date] = float(val)
            except ValueError:
                pass
    return result


def build_us_economic_csv():
    """Download multiple FRED series and merge into one wide CSV."""
    print("\n=== Downloading US Economic Indicators from FRED ===")
    all_data: dict[str, dict[str, float]] = {}

    for series_id, col_name in FRED_SERIES.items():
        series_data = download_fred_csv(series_id)
        if series_data:
            all_data[col_name] = series_data
            print(f"    {col_name}: {len(series_data)} observations")

    if not all_data:
        print("ERROR: No FRED data downloaded")
        return

    # Collect all dates
    all_dates = set()
    for series in all_data.values():
        all_dates.update(series.keys())
    all_dates = sorted(all_dates)

    col_names = list(all_data.keys())

    # Filter to 2015-2024 for manageable size
    all_dates = [d for d in all_dates if d >= "2015-01-01" and d <= "2024-12-31"]

    # Only keep dates where at least 5 series have values (filters out weekly dates)
    all_dates = [
        d for d in all_dates
        if sum(1 for col in col_names if d in all_data[col]) >= 5
    ]

    out_path = OUTPUT_DIR / "us_economic_indicators.csv"

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["date"] + col_names)
        for date in all_dates:
            row = [date]
            for col in col_names:
                val = all_data[col].get(date, "")
                row.append(val if val != "" else "")
            writer.writerow(row)

    print(f"\n  Saved: {out_path} ({len(all_dates)} rows, {len(col_names) + 1} columns)")


# --- World Bank: API with CSV download ---
WB_INDICATORS = {
    "NY.GDP.MKTP.CD": "gdp_usd",
    "SL.UEM.TOTL.ZS": "unemployment_pct",
    "FP.CPI.TOTL.ZG": "inflation_cpi_pct",
    "SP.POP.TOTL": "population_total",
    "SE.ADT.LITR.ZS": "literacy_rate_pct",
    "SH.XPD.CHEX.GD.ZS": "health_expenditure_gdp_pct",
    "SE.XPD.TOTL.GD.ZS": "education_expenditure_gdp_pct",
    "NY.GDP.PCAP.CD": "gdp_per_capita_usd",
    "SP.DYN.LE00.IN": "life_expectancy_years",
    "EN.ATM.CO2E.PC": "co2_emissions_metric_tons_per_capita",
    "EG.USE.ELEC.KH.PC": "electricity_consumption_kwh_per_capita",
    "IT.NET.USER.ZS": "internet_users_pct",
}

WB_COUNTRIES = [
    "USA", "CHN", "JPN", "DEU", "GBR", "FRA", "BRA", "IND", "CAN", "KOR",
    "MEX", "ESP", "ITA", "RUS", "AUS", "ARG", "SAU", "TUR", "IDN", "ZAF",
    "EGY", "NGA", "KEN", "COL", "PER", "CHL", "VNM", "THA", "POL", "NLD",
]


def download_worldbank_csv() -> list[dict]:
    """Download World Bank data for multiple indicators and countries."""
    print("\n=== Downloading Global Economic Indicators from World Bank ===")
    all_rows: list[dict] = []

    for indicator_code, col_name in WB_INDICATORS.items():
        countries_str = ";".join(WB_COUNTRIES)
        url = (
            f"https://api.worldbank.org/v2/country/{countries_str}/"
            f"indicator/{indicator_code}?format=json&per_page=1000&date=2015:2024"
        )
        print(f"  Downloading {col_name} ({indicator_code})...")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print(f"    WARNING: {col_name} failed: {e}")
            continue

        if len(data) < 2 or not data[1]:
            print(f"    WARNING: {col_name} returned no data")
            continue

        count = 0
        for item in data[1]:
            country = item.get("country", {}).get("id", "")
            year = item.get("date", "")
            value = item.get("value", None)
            if value is not None and country and year:
                all_rows.append({
                    "country_code": country,
                    "country_name": item.get("country", {}).get("value", ""),
                    "year": year,
                    "indicator": col_name,
                    "value": value,
                })
                count += 1
        print(f"    {col_name}: {count} observations")

    return all_rows


def build_global_economic_csv():
    """Download World Bank data and pivot to wide format."""
    rows = download_worldbank_csv()
    if not rows:
        print("ERROR: No World Bank data downloaded")
        return

    # Pivot: country_code, country_name, year, indicator1, indicator2, ...
    indicators = sorted(set(r["indicator"] for r in rows))
    countries = sorted(set((r["country_code"], r["country_name"]) for r in rows))
    years = sorted(set(r["year"] for r in rows))

    # Build lookup: (country, year) -> {indicator: value}
    lookup: dict[tuple, dict] = {}
    for r in rows:
        key = (r["country_code"], r["year"])
        if key not in lookup:
            lookup[key] = {}
        lookup[key][r["indicator"]] = r["value"]

    out_path = OUTPUT_DIR / "global_economic_indicators.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["country_code", "country_name", "year"] + indicators)
        for cc, cn in countries:
            for year in years:
                key = (cc, year)
                if key in lookup:
                    row = [cc, cn, year]
                    for ind in indicators:
                        row.append(lookup[key].get(ind, ""))
                    writer.writerow(row)

    print(f"\n  Saved: {out_path} ({len(countries)} countries x {len(years)} years, {len(indicators) + 3} columns)")


def main():
    print("Downloading real economic data for governance-agent testing...")
    build_us_economic_csv()
    build_global_economic_csv()
    print("\n=== Done! ===")
    print(f"Files saved in: {OUTPUT_DIR}")
    for f in OUTPUT_DIR.glob("*.csv"):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
