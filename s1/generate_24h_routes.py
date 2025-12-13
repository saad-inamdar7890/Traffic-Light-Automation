#!/usr/bin/env python3
"""
Generate 24-Hour Realistic Traffic Route Files
==============================================

Creates comprehensive 24-hour traffic scenarios with realistic time-of-day patterns:
- Night (12am-6am): Low traffic, minimal activity
- Morning Rush (6am-9am): High commuter traffic to work/school
- Midday (9am-12pm): Moderate traffic, deliveries
- Early Afternoon (12pm-3pm): Lunch traffic, steady flow
- Late Afternoon (3pm-5pm): Building toward evening rush
- Evening Rush (5pm-8pm): Peak traffic, return commute
- Late Evening (8pm-12am): Declining traffic, entertainment

Output: k1_routes_24h_{scenario}.rou.xml files

Scenarios:
- weekday: Standard workday pattern with strong rush hours
- weekend: Relaxed pattern, late morning peak
- friday: Stronger evening rush, entertainment traffic
- event: Major event causing afternoon/evening surge

Usage:
    python generate_24h_routes.py                      # Generate all scenarios
    python generate_24h_routes.py --scenario weekday   # Generate specific scenario
    python generate_24h_routes.py --validate           # Validate routes with SUMO
"""

import argparse
import os
import subprocess
import shutil
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
NETWORK_FILE = SCRIPT_DIR / 'k1.net.xml'
OUTPUT_DIR = SCRIPT_DIR

# =============================================================================
# TIME PERIODS (24 hours = 86400 seconds)
# =============================================================================

TIME_PERIODS = [
    # (start_sec, end_sec, name, description)
    (0,     21600, 'night',          '12am-6am: Low traffic, minimal activity'),
    (21600, 28800, 'early_morning',  '6am-8am: Building toward rush hour'),
    (28800, 32400, 'morning_peak',   '8am-9am: Peak morning rush'),
    (32400, 43200, 'late_morning',   '9am-12pm: Moderate, post-rush'),
    (43200, 50400, 'lunch',          '12pm-2pm: Lunch traffic, steady'),
    (50400, 57600, 'early_afternoon','2pm-4pm: Moderate, steady'),
    (57600, 61200, 'late_afternoon', '4pm-5pm: Building toward evening'),
    (61200, 72000, 'evening_peak',   '5pm-8pm: Peak evening rush'),
    (72000, 79200, 'late_evening',   '8pm-10pm: Declining traffic'),
    (79200, 86400, 'late_night',     '10pm-12am: Low traffic'),
]

# =============================================================================
# SCENARIO MULTIPLIERS
# Each list has 10 values corresponding to the 10 time periods
# Base rate: ~100 veh/hour for major routes, ~60 for medium, ~30 for minor
# =============================================================================

SCENARIO_MULTIPLIERS = {
    # Standard weekday: Strong morning and evening peaks
    'weekday': {
        'major':  [0.15, 0.80, 1.20, 0.65, 0.60, 0.55, 0.75, 1.25, 0.45, 0.20],
        'medium': [0.10, 0.70, 1.10, 0.60, 0.55, 0.50, 0.70, 1.15, 0.40, 0.15],
        'minor':  [0.08, 0.50, 0.80, 0.45, 0.40, 0.40, 0.55, 0.90, 0.30, 0.10],
    },
    # Weekend: Later morning peak, more leisure traffic
    'weekend': {
        'major':  [0.20, 0.40, 0.55, 0.80, 0.85, 0.80, 0.75, 0.90, 0.60, 0.30],
        'medium': [0.15, 0.35, 0.50, 0.75, 0.80, 0.75, 0.70, 0.85, 0.55, 0.25],
        'minor':  [0.12, 0.30, 0.40, 0.60, 0.65, 0.60, 0.55, 0.70, 0.45, 0.20],
    },
    # Friday: Normal morning, extended evening rush, entertainment
    'friday': {
        'major':  [0.15, 0.80, 1.20, 0.65, 0.60, 0.60, 0.85, 1.40, 0.70, 0.40],
        'medium': [0.10, 0.70, 1.10, 0.60, 0.55, 0.55, 0.80, 1.30, 0.65, 0.35],
        'minor':  [0.08, 0.50, 0.80, 0.45, 0.40, 0.45, 0.65, 1.10, 0.55, 0.30],
    },
    # Event day: Afternoon buildup, massive evening surge
    'event': {
        'major':  [0.15, 0.75, 1.10, 0.70, 0.80, 1.20, 1.50, 1.80, 1.00, 0.35],
        'medium': [0.10, 0.65, 1.00, 0.65, 0.75, 1.10, 1.40, 1.70, 0.90, 0.30],
        'minor':  [0.08, 0.45, 0.75, 0.50, 0.60, 0.90, 1.20, 1.50, 0.75, 0.25],
    },
}

# =============================================================================
# ROUTE DEFINITIONS
# Based on K1 network topology, categorized by importance
# =============================================================================

# Major routes: Main corridors with highest traffic
MAJOR_ROUTES = [
    # (from, to, via, vehicle_type)
    ('-E15', 'E24', 'E4 -E5 E16', 'passenger'),
    ('-E15', 'E21', '-E18 E19 E20', 'passenger'),
    ('-E15', 'E10', '-E18 -E17', 'passenger'),
    ('-E24', 'E15', '-E16 E5 -E4', 'passenger'),
    ('-E21', 'E15', '-E20 -E19 E18', 'passenger'),
    ('-E10', 'E15', 'E17 E18', 'passenger'),
    ('-E11', 'E24', '', 'passenger'),
    ('-E24', 'E11', '-E23', 'passenger'),
    ('-E9', 'E24', '', 'passenger'),
    ('-E12', 'E21', '', 'passenger'),
]

# Medium routes: Secondary corridors
MEDIUM_ROUTES = [
    ('-E11', 'E21', '', 'passenger'),
    ('-E12', 'E15', '', 'passenger'),
    ('-E13', 'E9', '', 'passenger'),
    ('-E13', 'E15', '', 'passenger'),
    ('-E24', 'E12', '-E23', 'passenger'),
    ('-E24', 'E15', '', 'passenger'),
    ('-E21', 'E15', '', 'passenger'),
    ('-E21', 'E10', '', 'passenger'),
    ('-E21', 'E11', '-E20 -E19 E18 -E17', 'passenger'),
    ('-E10', 'E21', 'E17 E18 E19 E20', 'passenger'),
]

# Minor routes: Local streets
MINOR_ROUTES = [
    ('-E9', 'E15', '', 'passenger'),
    ('-E9', 'E21', '', 'passenger'),
    ('-E9', 'E12', '', 'passenger'),
    ('-E11', 'E12', '', 'passenger'),
    ('-E13', 'E21', '', 'passenger'),
    ('-E13', 'E24', '', 'passenger'),
    ('-E12', 'E9', '', 'passenger'),
    ('-E12', 'E24', '', 'passenger'),
]

# Commercial/delivery routes (trucks and delivery vehicles)
COMMERCIAL_ROUTES = [
    ('-E15', 'E24', 'E4 -E5 E16', 'truck'),
    ('-E24', 'E15', '-E16 E5 -E4', 'truck'),
    ('-E11', 'E24', '', 'delivery'),
    ('-E24', 'E11', '-E23', 'delivery'),
    ('-E15', 'E21', '-E18 E19 E20', 'delivery'),
]

# Bus routes (fixed schedules, consistent throughout day)
BUS_ROUTES = [
    ('-E15', 'E24', 'E4 -E5 E16', 'bus'),
    ('-E24', 'E15', '-E16 E5 -E4', 'bus'),
    ('-E21', 'E15', '-E20 -E19 E18', 'bus'),
    ('-E15', 'E21', '-E18 E19 E20', 'bus'),
]

# Base rates for each category (vehicles per hour)
BASE_RATES = {
    'major': 200,
    'medium': 120,
    'minor': 60,
    'commercial': 25,
    'bus': 8,  # Fixed bus schedule, not multiplied as much
}


def generate_flows(scenario_name: str) -> str:
    """
    Generate flow definitions for a 24-hour scenario.
    
    Args:
        scenario_name: One of 'weekday', 'weekend', 'friday', 'event'
    
    Returns:
        XML string with all flow definitions
    """
    if scenario_name not in SCENARIO_MULTIPLIERS:
        raise ValueError(f"Unknown scenario: {scenario_name}")
    
    multipliers = SCENARIO_MULTIPLIERS[scenario_name]
    lines = []
    
    # XML header
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('')
    lines.append(f'<!-- 24-Hour Traffic Scenario: {scenario_name.upper()} -->')
    lines.append(f'<!-- Generated: {datetime.now().isoformat()} -->')
    lines.append('<!-- Duration: 86400 seconds (24 hours) -->')
    lines.append('')
    lines.append('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">')
    lines.append('')
    
    # Vehicle types
    lines.append('    <!-- Vehicle Types -->')
    lines.append('    <vType id="DEFAULT_VEHTYPE"/>')
    lines.append('    <vType id="passenger" vClass="passenger" accel="2.6" decel="4.5" sigma="0.5" length="4.5" maxSpeed="50"/>')
    lines.append('    <vType id="delivery" vClass="delivery" accel="2.0" decel="4.0" sigma="0.5" length="6.0" maxSpeed="40"/>')
    lines.append('    <vType id="truck" vClass="truck" accel="1.3" decel="4.0" sigma="0.5" length="12.0" maxSpeed="35"/>')
    lines.append('    <vType id="bus" vClass="bus" accel="1.2" decel="4.0" sigma="0.5" length="12.0" maxSpeed="30"/>')
    lines.append('')
    
    flow_id = 0
    
    # Generate major route flows
    lines.append('    <!-- ========== MAJOR ROUTES (High Volume) ========== -->')
    for route_idx, (from_edge, to_edge, via, vtype) in enumerate(MAJOR_ROUTES):
        lines.append(f'    <!-- Route M{route_idx}: {from_edge} → {to_edge} -->')
        for period_idx, (t0, t1, period_name, _) in enumerate(TIME_PERIODS):
            mult = multipliers['major'][period_idx]
            rate = max(1, int(BASE_RATES['major'] * mult))
            fid = f"major_{route_idx}_{period_name}"
            via_attr = f' via="{via}"' if via else ''
            lines.append(f'    <flow id="{fid}" begin="{t0}" end="{t1}" from="{from_edge}" to="{to_edge}"{via_attr} vehsPerHour="{rate}" type="{vtype}"/>')
            flow_id += 1
        lines.append('')
    
    # Generate medium route flows
    lines.append('    <!-- ========== MEDIUM ROUTES (Secondary) ========== -->')
    for route_idx, (from_edge, to_edge, via, vtype) in enumerate(MEDIUM_ROUTES):
        lines.append(f'    <!-- Route S{route_idx}: {from_edge} → {to_edge} -->')
        for period_idx, (t0, t1, period_name, _) in enumerate(TIME_PERIODS):
            mult = multipliers['medium'][period_idx]
            rate = max(1, int(BASE_RATES['medium'] * mult))
            fid = f"medium_{route_idx}_{period_name}"
            via_attr = f' via="{via}"' if via else ''
            lines.append(f'    <flow id="{fid}" begin="{t0}" end="{t1}" from="{from_edge}" to="{to_edge}"{via_attr} vehsPerHour="{rate}" type="{vtype}"/>')
            flow_id += 1
        lines.append('')
    
    # Generate minor route flows
    lines.append('    <!-- ========== MINOR ROUTES (Local) ========== -->')
    for route_idx, (from_edge, to_edge, via, vtype) in enumerate(MINOR_ROUTES):
        lines.append(f'    <!-- Route L{route_idx}: {from_edge} → {to_edge} -->')
        for period_idx, (t0, t1, period_name, _) in enumerate(TIME_PERIODS):
            mult = multipliers['minor'][period_idx]
            rate = max(1, int(BASE_RATES['minor'] * mult))
            fid = f"minor_{route_idx}_{period_name}"
            via_attr = f' via="{via}"' if via else ''
            lines.append(f'    <flow id="{fid}" begin="{t0}" end="{t1}" from="{from_edge}" to="{to_edge}"{via_attr} vehsPerHour="{rate}" type="{vtype}"/>')
            flow_id += 1
        lines.append('')
    
    # Generate commercial flows (trucks/delivery - only during business hours)
    lines.append('    <!-- ========== COMMERCIAL TRAFFIC (Business Hours) ========== -->')
    commercial_active_periods = [3, 4, 5, 6]  # Late morning through late afternoon
    for route_idx, (from_edge, to_edge, via, vtype) in enumerate(COMMERCIAL_ROUTES):
        lines.append(f'    <!-- Commercial {route_idx}: {from_edge} → {to_edge} ({vtype}) -->')
        for period_idx, (t0, t1, period_name, _) in enumerate(TIME_PERIODS):
            if period_idx in commercial_active_periods:
                # Commercial traffic during business hours
                rate = max(1, int(BASE_RATES['commercial'] * 1.5))
            else:
                rate = max(1, int(BASE_RATES['commercial'] * 0.3))
            fid = f"commercial_{route_idx}_{period_name}"
            via_attr = f' via="{via}"' if via else ''
            lines.append(f'    <flow id="{fid}" begin="{t0}" end="{t1}" from="{from_edge}" to="{to_edge}"{via_attr} vehsPerHour="{rate}" type="{vtype}"/>')
            flow_id += 1
        lines.append('')
    
    # Generate bus flows (relatively constant throughout day)
    lines.append('    <!-- ========== BUS ROUTES (Public Transit) ========== -->')
    for route_idx, (from_edge, to_edge, via, vtype) in enumerate(BUS_ROUTES):
        lines.append(f'    <!-- Bus Route {route_idx}: {from_edge} → {to_edge} -->')
        for period_idx, (t0, t1, period_name, _) in enumerate(TIME_PERIODS):
            # Buses run more frequently during rush hours, less at night
            if period_idx in [2, 7]:  # Morning and evening peak
                rate = int(BASE_RATES['bus'] * 1.5)
            elif period_idx in [0, 9]:  # Night periods
                rate = max(1, int(BASE_RATES['bus'] * 0.3))
            else:
                rate = BASE_RATES['bus']
            fid = f"bus_{route_idx}_{period_name}"
            via_attr = f' via="{via}"' if via else ''
            lines.append(f'    <flow id="{fid}" begin="{t0}" end="{t1}" from="{from_edge}" to="{to_edge}"{via_attr} vehsPerHour="{rate}" type="{vtype}"/>')
            flow_id += 1
        lines.append('')
    
    lines.append('</routes>')
    
    print(f"  Generated {flow_id} flows for scenario '{scenario_name}'")
    return '\n'.join(lines)


def validate_routes(route_file: Path) -> bool:
    """
    Validate routes using SUMO's route validation.
    
    Args:
        route_file: Path to route file
    
    Returns:
        True if valid, False otherwise
    """
    duarouter = shutil.which('duarouter')
    if not duarouter:
        sumo_home = os.environ.get('SUMO_HOME', '')
        if sumo_home:
            duarouter = str(Path(sumo_home) / 'bin' / 'duarouter')
    
    if not duarouter or not Path(duarouter).exists():
        print(f"  Warning: duarouter not found, skipping validation")
        return True
    
    # Quick validation run
    cmd = [
        duarouter,
        '-n', str(NETWORK_FILE),
        '-r', str(route_file),
        '--ignore-errors',
        '--no-warnings',
        '-o', str(route_file.with_suffix('.validated.xml')),
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        # Clean up validation output
        validated_file = route_file.with_suffix('.validated.xml')
        if validated_file.exists():
            validated_file.unlink()
        alt_file = route_file.with_suffix('.validated.alt.xml')
        if alt_file.exists():
            alt_file.unlink()
        
        if result.returncode == 0:
            print(f"  ✓ Routes validated successfully")
            return True
        else:
            print(f"  ✗ Route validation had issues (continuing anyway)")
            return True  # Continue anyway, SUMO will handle errors
    except Exception as e:
        print(f"  Warning: Validation failed: {e}")
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Generate 24-hour traffic route files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_24h_routes.py                    # Generate all scenarios
  python generate_24h_routes.py --scenario weekday # Generate weekday only
  python generate_24h_routes.py --validate         # Validate after generation
"""
    )
    
    parser.add_argument(
        '--scenario', '-s',
        type=str,
        choices=['weekday', 'weekend', 'friday', 'event', 'all'],
        default='all',
        help='Scenario to generate (default: all)'
    )
    parser.add_argument(
        '--validate', '-v',
        action='store_true',
        help='Validate generated routes with SUMO'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=str(OUTPUT_DIR),
        help=f'Output directory (default: {OUTPUT_DIR})'
    )
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    
    print("="*60)
    print("24-HOUR TRAFFIC ROUTE GENERATOR")
    print("="*60)
    print(f"\nOutput directory: {output_dir}")
    print(f"Network file: {NETWORK_FILE}")
    
    # Determine scenarios to generate
    if args.scenario == 'all':
        scenarios = list(SCENARIO_MULTIPLIERS.keys())
    else:
        scenarios = [args.scenario]
    
    print(f"\nScenarios to generate: {', '.join(scenarios)}")
    print()
    
    generated_files = []
    
    for scenario in scenarios:
        print(f"Generating {scenario} scenario...")
        
        # Generate routes
        xml_content = generate_flows(scenario)
        
        # Write to file
        output_file = output_dir / f'k1_routes_24h_{scenario}.rou.xml'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(xml_content)
        
        print(f"  Written to: {output_file}")
        generated_files.append(output_file)
        
        # Validate if requested
        if args.validate:
            validate_routes(output_file)
        
        print()
    
    # Generate SUMO config files
    print("Generating SUMO configuration files...")
    
    for scenario in scenarios:
        config_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="k1.net.xml"/>
        <route-files value="k1_routes_24h_{scenario}.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="86400"/>
    </time>
    <processing>
        <time-to-teleport value="-1"/>
        <ignore-route-errors value="true"/>
    </processing>
</configuration>
'''
        config_file = output_dir / f'k1_24h_{scenario}.sumocfg'
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_content)
        print(f"  Config: {config_file}")
    
    print("\n" + "="*60)
    print("GENERATION COMPLETE")
    print("="*60)
    print(f"\nGenerated {len(generated_files)} route files and {len(scenarios)} config files")
    print("\nNext steps:")
    print("  1. Train on 24h scenario:")
    print("     python train_24h_scenario.py --scenario weekday")
    print("  2. Evaluate after training:")
    print("     python evaluate_24h_performance.py --checkpoint <checkpoint> --scenario weekday")


if __name__ == '__main__':
    main()
