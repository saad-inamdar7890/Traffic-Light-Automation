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

Optional: --duarouter flag to run SUMO duarouter and produce precomputed static routes
          This converts <flow> entries to <vehicle> or <trip> with explicit routes,
          eliminating runtime routing failures.
"""

import re
import xml.etree.ElementTree as ET
from statistics import mean
from pathlib import Path
import subprocess
import shutil
import argparse
import os

ROOT = Path(__file__).resolve().parents[0]
INPUT_ROUTE = ROOT / 'k1_routes_3h_varying.rou.xml'
OUTPUT_PREFIX = ROOT / 'k1_routes_6h'
NETWORK_FILE = ROOT / 'k1.net.xml'

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
    """Return dict: base_name -> list of flow dicts (id, from, to, via, vehsPerHour, type, commented) - ONLY NON-COMMENTED"""
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
        
        # ONLY include flows that are NOT commented
        if not is_commented:
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


def run_duarouter(route_file: Path, network_file: Path, output_file: Path):
    """
    Run SUMO duarouter to precompute static routes from flows.
    
    This converts <flow> entries to explicit <vehicle> or <trip> entries with
    precomputed routes, which eliminates runtime DijkstraRouter failures.
    
    Args:
        route_file: Input route file with flows
        network_file: SUMO network file (.net.xml)
        output_file: Output file with precomputed routes
    
    Returns:
        True if successful, False otherwise
    """
    # Find duarouter binary
    duarouter_bin = shutil.which('duarouter')
    if not duarouter_bin:
        # Try common SUMO installation paths
        sumo_home = os.environ.get('SUMO_HOME', '')
        if sumo_home:
            candidate = Path(sumo_home) / 'bin' / 'duarouter'
            if candidate.exists():
                duarouter_bin = str(candidate)
            candidate_exe = Path(sumo_home) / 'bin' / 'duarouter.exe'
            if candidate_exe.exists():
                duarouter_bin = str(candidate_exe)
    
    if not duarouter_bin:
        print(f"  ⚠️  duarouter not found in PATH or SUMO_HOME. Skipping precomputation.")
        print(f"      Set SUMO_HOME or add SUMO bin directory to PATH to enable this feature.")
        return False
    
    print(f"  → Running duarouter on {route_file.name}...")
    
    # Build duarouter command
    cmd = [
        duarouter_bin,
        '-n', str(network_file),
        '-r', str(route_file),
        '-o', str(output_file),
        '--ignore-errors',           # Continue on routing errors (log them)
        '--repair',                  # Try to repair invalid routes
        '--remove-loops',            # Remove U-turns
        '--routing-algorithm', 'dijkstra',
        '--no-warnings',
        '--write-license',
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            print(f"  ✅ Precomputed routes written to: {output_file.name}")
            return True
        else:
            print(f"  ⚠️  duarouter returned code {result.returncode}")
            if result.stderr:
                # Print first few lines of stderr
                lines = result.stderr.strip().split('\n')
                for line in lines[:10]:
                    print(f"      {line}")
                if len(lines) > 10:
                    print(f"      ... ({len(lines) - 10} more lines)")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"  ⚠️  duarouter timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"  ⚠️  duarouter failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Generate 6-hour route files from 3h definitions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_6h_routes.py                    # Generate flow-based routes only
  python generate_6h_routes.py --duarouter        # Also precompute static routes
  python generate_6h_routes.py --duarouter-only   # Only precompute (skip generation)
"""
    )
    parser.add_argument(
        '--duarouter', '-d',
        action='store_true',
        help='Run duarouter to precompute static routes after generating flows'
    )
    parser.add_argument(
        '--duarouter-only',
        action='store_true',
        help='Only run duarouter on existing route files (skip flow generation)'
    )
    parser.add_argument(
        '--network', '-n',
        type=Path,
        default=NETWORK_FILE,
        help=f'Path to network file (default: {NETWORK_FILE.name})'
    )
    args = parser.parse_args()

    generated_files = []

    if not args.duarouter_only:
        raw = INPUT_ROUTE.read_text(encoding='utf-8')
        groups = parse_flows(raw)
        print(f"Found {len(groups)} unique route groups to expand")

        for scen in SCENARIOS.keys():
            out = OUTPUT_PREFIX.with_name(f'k1_routes_6h_{scen}.rou.xml')
            content = build_route_entries(groups, scen)
            out.write_text(content, encoding='utf-8')
            print(f"Wrote scenario: {out.name}")
            generated_files.append(out)

        print("\nDone generating flow-based routes.")
    else:
        # Collect existing route files
        for scen in SCENARIOS.keys():
            f = OUTPUT_PREFIX.with_name(f'k1_routes_6h_{scen}.rou.xml')
            if f.exists():
                generated_files.append(f)
        print(f"Found {len(generated_files)} existing route files for duarouter.")

    # Run duarouter if requested
    if args.duarouter or args.duarouter_only:
        print("\n" + "="*60)
        print("DUAROUTER: Precomputing static routes")
        print("="*60)
        
        if not args.network.exists():
            print(f"❌ Network file not found: {args.network}")
            return
        
        success_count = 0
        for route_file in generated_files:
            precomputed_file = route_file.with_name(
                route_file.stem.replace('.rou', '') + '_precomputed.rou.xml'
            )
            if run_duarouter(route_file, args.network, precomputed_file):
                success_count += 1
        
        print(f"\n✅ Precomputed {success_count}/{len(generated_files)} route files successfully.")
        if success_count > 0:
            print("\nTo use precomputed routes, update your .sumocfg to reference the *_precomputed.rou.xml files.")
            print("Precomputed routes eliminate runtime DijkstraRouter failures.")
    else:
        print("\nRun validate_routes.py on each generated file to verify connectivity.")
        print("Tip: Use --duarouter to precompute static routes and avoid runtime routing errors.")


if __name__ == '__main__':
    main()
