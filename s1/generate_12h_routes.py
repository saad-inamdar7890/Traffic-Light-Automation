#!/usr/bin/env python3
"""
Generate 12-Hour Realistic Traffic Route Files
==============================================

Creates comprehensive 12-hour traffic scenarios (6am-6pm) capturing key patterns:
- Early Morning (6am-8am): Building toward rush hour
- Morning Rush (8am-9am): Peak morning rush
- Late Morning (9am-12pm): Moderate, post-rush
- Lunch (12pm-2pm): Lunch traffic, steady
- Early Afternoon (2pm-4pm): Moderate, steady
- Late Afternoon (4pm-6pm): Building toward evening rush

Output: k1_routes_12h_{scenario}.rou.xml files

Scenarios:
- weekday: Standard workday pattern with strong rush hours
- weekend: Relaxed pattern, late morning peak
- friday: Stronger afternoon traffic, pre-evening rush
- event: Major event causing afternoon surge

Usage:
    python generate_12h_routes.py                      # Generate all scenarios
    python generate_12h_routes.py --scenario weekday   # Generate specific scenario
    python generate_12h_routes.py --validate           # Validate routes with SUMO
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
# TIME PERIODS (12 hours = 43200 seconds, 6am-6pm)
# =============================================================================

TIME_PERIODS = [
    # (start_sec, end_sec, name, description)
    (0,     7200,  'early_morning',  '6am-8am: Building toward rush hour'),
    (7200,  10800, 'morning_peak',   '8am-9am: Peak morning rush'),
    (10800, 21600, 'late_morning',   '9am-12pm: Moderate, post-rush'),
    (21600, 28800, 'lunch',          '12pm-2pm: Lunch traffic, steady'),
    (28800, 36000, 'early_afternoon','2pm-4pm: Moderate, steady'),
    (36000, 43200, 'late_afternoon', '4pm-6pm: Building toward evening'),
]

# =============================================================================
# SCENARIO MULTIPLIERS
# Each list has 6 values corresponding to the 6 time periods
# Base rate: ~100 veh/hour for major routes, ~60 for medium, ~30 for minor
# =============================================================================

SCENARIO_MULTIPLIERS = {
    # Standard weekday: Strong morning peak and evening rush hour
    'weekday': {
        'major':  [0.80, 1.20, 0.65, 0.60, 0.55, 2.00],  # Late afternoon = 2x rush hour
        'medium': [0.70, 1.10, 0.60, 0.55, 0.50, 1.80],
        'minor':  [0.50, 0.80, 0.45, 0.40, 0.40, 1.40],
    },
    # Weekend: Later morning peak, more leisure traffic
    'weekend': {
        'major':  [0.40, 0.55, 0.80, 0.85, 0.80, 0.75],
        'medium': [0.35, 0.50, 0.75, 0.80, 0.75, 0.70],
        'minor':  [0.30, 0.40, 0.60, 0.65, 0.60, 0.55],
    },
    # Friday: Normal morning, building afternoon
    'friday': {
        'major':  [0.80, 1.20, 0.65, 0.60, 0.70, 0.95],
        'medium': [0.70, 1.10, 0.60, 0.55, 0.65, 0.90],
        'minor':  [0.50, 0.80, 0.45, 0.40, 0.55, 0.75],
    },
    # Event day: Afternoon buildup, heavy late afternoon
    'event': {
        'major':  [0.75, 1.10, 0.70, 0.90, 1.20, 1.50],
        'medium': [0.65, 1.00, 0.65, 0.85, 1.10, 1.40],
        'minor':  [0.45, 0.75, 0.50, 0.70, 0.90, 1.20],
    },
}

# =============================================================================
# ROUTE DEFINITIONS (Same as 24h version)
# =============================================================================

# Major routes: Main corridors with highest traffic
MAJOR_ROUTES = [
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

# Commercial/delivery routes
COMMERCIAL_ROUTES = [
    ('-E15', 'E24', 'E4 -E5 E16', 'truck'),
    ('-E24', 'E15', '-E16 E5 -E4', 'truck'),
    ('-E11', 'E24', '', 'delivery'),
    ('-E24', 'E11', '-E23', 'delivery'),
    ('-E15', 'E21', '-E18 E19 E20', 'delivery'),
]

# Bus routes
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
    'bus': 8,
}


def generate_flows(scenario_name: str) -> str:
    """Generate flow definitions for a 12-hour scenario."""
    if scenario_name not in SCENARIO_MULTIPLIERS:
        raise ValueError(f"Unknown scenario: {scenario_name}")
    
    multipliers = SCENARIO_MULTIPLIERS[scenario_name]
    lines = []
    
    # XML header
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">')
    lines.append('')
    
    # Vehicle types
    lines.append('    <!-- Vehicle Types -->')
    lines.append('    <vType id="passenger" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="50" guiShape="passenger"/>')
    lines.append('    <vType id="truck" accel="1.3" decel="4.0" sigma="0.5" length="12" maxSpeed="36" guiShape="truck"/>')
    lines.append('    <vType id="bus" accel="1.2" decel="4.0" sigma="0.3" length="12" maxSpeed="25" guiShape="bus"/>')
    lines.append('    <vType id="delivery" accel="2.0" decel="4.0" sigma="0.5" length="7" maxSpeed="40" guiShape="delivery"/>')
    lines.append('')
    
    flow_id = 0
    
    # Generate flows for each route category
    route_categories = [
        ('major', MAJOR_ROUTES),
        ('medium', MEDIUM_ROUTES),
        ('minor', MINOR_ROUTES),
        ('commercial', COMMERCIAL_ROUTES),
        ('bus', BUS_ROUTES),
    ]
    
    for category, routes in route_categories:
        lines.append(f'    <!-- {category.upper()} Routes -->')
        base_rate = BASE_RATES[category]
        
        # Get multipliers for this category
        if category in ['commercial', 'bus']:
            cat_multipliers = multipliers.get('medium', [0.5] * 6)
            if category == 'bus':
                cat_multipliers = [0.8] * 6  # Bus is more consistent
        else:
            cat_multipliers = multipliers[category]
        
        for route_from, route_to, route_via, vtype in routes:
            for period_idx, (start, end, period_name, _) in enumerate(TIME_PERIODS):
                mult = cat_multipliers[period_idx]
                rate = base_rate * mult / len(routes)
                
                if rate < 1:
                    rate = 1
                
                # Convert to per-hour probability
                veh_per_sec = rate / 3600.0
                period_duration = end - start
                
                flow_attrs = [
                    f'id="f_{flow_id}_{period_name}"',
                    f'from="{route_from}"',
                    f'to="{route_to}"',
                    f'begin="{start}"',
                    f'end="{end}"',
                    f'probability="{veh_per_sec:.6f}"',
                    f'type="{vtype}"',
                ]
                
                if route_via:
                    flow_attrs.insert(3, f'via="{route_via}"')
                
                lines.append(f'    <flow {" ".join(flow_attrs)}/>')
                flow_id += 1
        
        lines.append('')
    
    lines.append('</routes>')
    
    return '\n'.join(lines)


def generate_sumocfg(scenario_name: str) -> str:
    """Generate SUMO configuration file for 12-hour scenario."""
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="k1.net.xml"/>
        <route-files value="k1_routes_12h_{scenario_name}.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="43200"/>
        <step-length value="1"/>
    </time>
    <processing>
        <time-to-teleport value="300"/>
        <collision.action value="warn"/>
    </processing>
    <report>
        <verbose value="false"/>
        <no-step-log value="true"/>
    </report>
</configuration>
'''


def validate_routes(scenario_name: str) -> bool:
    """Validate routes using SUMO."""
    route_file = OUTPUT_DIR / f'k1_routes_12h_{scenario_name}.rou.xml'
    config_file = OUTPUT_DIR / f'k1_12h_{scenario_name}.sumocfg'
    
    if not route_file.exists():
        print(f"Error: Route file not found: {route_file}")
        return False
    
    try:
        result = subprocess.run(
            ['sumo', '-c', str(config_file), '--no-step-log', '--no-warnings', '--end', '100'],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            print(f"✓ Routes validated successfully: {scenario_name}")
            return True
        else:
            print(f"✗ Validation failed for {scenario_name}:")
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print(f"✗ Validation timed out for {scenario_name}")
        return False
    except FileNotFoundError:
        print("Warning: SUMO not found. Skipping validation.")
        return True


def main():
    parser = argparse.ArgumentParser(description='Generate 12-hour traffic route files')
    parser.add_argument('--scenario', choices=['weekday', 'weekend', 'friday', 'event', 'all'],
                        default='all', help='Scenario to generate')
    parser.add_argument('--validate', action='store_true', help='Validate generated routes')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    
    args = parser.parse_args()
    
    global OUTPUT_DIR
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    scenarios = ['weekday', 'weekend', 'friday', 'event'] if args.scenario == 'all' else [args.scenario]
    
    print("=" * 60)
    print("12-Hour Traffic Route Generator (6am-6pm)")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Network file: {NETWORK_FILE}")
    print(f"Scenarios: {', '.join(scenarios)}")
    print()
    
    for scenario in scenarios:
        print(f"\nGenerating {scenario} scenario...")
        
        # Generate route file
        route_xml = generate_flows(scenario)
        route_file = OUTPUT_DIR / f'k1_routes_12h_{scenario}.rou.xml'
        with open(route_file, 'w') as f:
            f.write(route_xml)
        print(f"  ✓ Created: {route_file.name}")
        
        # Generate config file
        config_xml = generate_sumocfg(scenario)
        config_file = OUTPUT_DIR / f'k1_12h_{scenario}.sumocfg'
        with open(config_file, 'w') as f:
            f.write(config_xml)
        print(f"  ✓ Created: {config_file.name}")
        
        # Count flows
        flow_count = route_xml.count('<flow ')
        print(f"  ✓ Generated {flow_count} flows across 6 time periods")
        
        # Validate if requested
        if args.validate:
            validate_routes(scenario)
    
    print("\n" + "=" * 60)
    print("Route generation complete!")
    print("=" * 60)
    print("\nTo use in training:")
    print("  python train_12h_scenario.py --scenario weekday")
    print("\nTo validate routes:")
    print("  sumo -c k1_12h_weekday.sumocfg --start --quit-on-end --end 100")


if __name__ == '__main__':
    main()
