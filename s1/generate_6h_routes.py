#!/usr/bin/env python3
"""
Generate 6-hour route files from existing route definitions.
Creates scenario variants: weekday, weekend, event_day

Outputs: k1_routes_6h_weekday.rou.xml, k1_routes_6h_weekend.rou.xml, k1_routes_6h_event.rou.xml

Algorithm:
- Parse all flows (including commented ones) from the existing 3h route file
- Group by base route id (strip _light/_rush/_peak/_declining/_building)
- For each base route, compute a base vehsPerHour from available definitions (mean) or default
- Create 6 time periods of 3600s each over 21600s (6 hours)
- Apply scenario multipliers per period to base rate to form vehsPerHour for each period
- Preserve via and vehicle type
- Write scenario route files and a small header
"""

import re
import xml.etree.ElementTree as ET
from statistics import mean
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]
INPUT_ROUTE = ROOT / 'k1_routes_3h_varying.rou.xml'
OUTPUT_PREFIX = ROOT / 'k1_routes_6h'

# Time periods (6 x 3600s = 21600s => 6 hours)
PERIODS = [
    (0, 3600, 'early_morning'),
    (3600, 7200, 'morning_peak'),
    (7200, 10800, 'midday'),
    (10800, 14400, 'afternoon'),
    (14400, 18000, 'evening_peak'),
    (18000, 21600, 'night')
]

# Scenario multipliers per period
SCENARIOS = {
    'weekday': [0.5, 1.6, 1.0, 0.9, 1.7, 0.4],
    'weekend': [0.6, 0.9, 1.2, 1.0, 1.1, 0.5],
    'event':   [0.4, 1.0, 1.8, 1.5, 2.0, 0.6]
}

FLOW_REGEX = re.compile(r'<flow\s+id="([^"]+)"\s+[^>]*from="([^"]+)"\s+[^>]*to="([^"]+)"(?:\s+via="([^"]*)")?[^>]*vehsPerHour="([0-9\.]+)"[^>]*type="([^"]+)"[^>]*/?>')


def parse_flows(raw_text):
    """Return dict: base_name -> list of flow dicts (id, from, to, via, vehsPerHour, type, commented)"""
    flows = []
    for m in FLOW_REGEX.finditer(raw_text):
        fid = m.group(1)
        ffrom = m.group(2)
        fto = m.group(3)
        via = m.group(4) or ''
        rate = float(m.group(5))
        vtype = m.group(6)
        # Determine if in comment
        start = m.start()
        before = raw_text[:start]
        is_commented = before.count('<!--') > before.count('-->')
        flows.append({'id': fid, 'from': ffrom, 'to': fto, 'via': via, 'rate': rate, 'type': vtype, 'commented': is_commented})

    # Group by base name
    groups = {}
    for f in flows:
        parts = f['id'].split('_')
        base = '_'.join(parts[:-1]) if len(parts) > 1 else f['id']
        groups.setdefault(base, []).append(f)
    return groups


def compute_base_rate(group):
    rates = [f['rate'] for f in group if f.get('rate')]
    if rates:
        return max(1.0, mean(rates))
    return 50.0


def build_route_entries(groups, scenario_name):
    multipliers = SCENARIOS[scenario_name]
    lines = []
    # Header
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">')
    lines.append('    <vType id="DEFAULT_VEHTYPE"/>')
    lines.append('    <vType id="passenger" vClass="passenger"/>')
    lines.append('    <vType id="bus" vClass="bus"/>')
    lines.append('    <vType id="truck" vClass="truck"/>')
    lines.append('')

    for base, group in sorted(groups.items()):
        # pick first as representative
        rep = group[0]
        from_edge = rep['from']
        to_edge = rep['to']
        via = rep['via']
        vtype = rep['type']
        base_rate = compute_base_rate(group)

        # Create one flow per period
        for i, (t0, t1, pname) in enumerate(PERIODS):
            rate = max(1, int(base_rate * multipliers[i]))
            flow_id = f"{base}_{pname}"
            if via:
                lines.append(f'    <flow id="{flow_id}" begin="{t0:.2f}" from="{from_edge}" to="{to_edge}" via="{via}" end="{t1:.2f}" vehsPerHour="{rate}" type="{vtype}"/>')
            else:
                lines.append(f'    <flow id="{flow_id}" begin="{t0:.2f}" from="{from_edge}" to="{to_edge}" end="{t1:.2f}" vehsPerHour="{rate}" type="{vtype}"/>')
        lines.append('')

    lines.append('</routes>')
    return '\n'.join(lines)


def main():
    raw = INPUT_ROUTE.read_text(encoding='utf-8')
    groups = parse_flows(raw)
    print(f"Found {len(groups)} unique route groups to expand")

    for scen in SCENARIOS.keys():
        out = OUTPUT_PREFIX.with_name(f'k1_routes_6h_{scen}.rou.xml')
        content = build_route_entries(groups, scen)
        out.write_text(content, encoding='utf-8')
        print(f"Wrote scenario: {out.name}")

    print("\nDone. Run validate_routes.py on each generated file to verify connectivity.")

if __name__ == '__main__':
    main()
