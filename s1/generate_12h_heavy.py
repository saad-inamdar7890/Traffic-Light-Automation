#!/usr/bin/env python3
"""
Generate 12-Hour Heavy Traffic Scenario for Extended Training
=============================================================

Creates a 12-hour heavy traffic scenario using validated routes from original.rou.xml.
This allows the model to adapt to sustained heavy traffic over a longer duration.

Duration: 12 hours (43200 seconds)
Traffic: Heavy (~8,240 vehicles/hour sustained)

Usage:
    python generate_12h_heavy.py
    python generate_12h_heavy.py --validate
"""

import argparse
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR

# =============================================================================
# VALIDATED ROUTES FROM original.rou.xml
# All 39 routes with heavy traffic base rates
# =============================================================================

ROUTES = [
    # From -E9 (5 routes)
    ('-E9', 'E13', 200),
    ('-E9', 'E6', 200),
    ('-E9', 'E24', 240),
    ('-E9', 'E10', 160),
    ('-E9', 'E21', 180),
    
    # From -E10 (5 routes)  
    ('-E10', 'E6', 200),
    ('-E10', 'E21', 300),
    ('-E10', 'E15', 360),  # Major corridor - highest traffic
    ('-E10', 'E24', 240),
    ('-E10', 'E13', 160),
    
    # From -E11 (5 routes)
    ('-E11', 'E6', 180),
    ('-E11', 'E9', 160),
    ('-E11', 'E13', 140),
    ('-E11', 'E15', 200),
    ('-E11', 'E24', 300),  # Major route
    
    # From -E12 (4 routes)
    ('-E12', 'E9', 160),
    ('-E12', 'E6', 180),
    ('-E12', 'E15', 200),
    ('-E12', 'E24', 220),
    
    # From -E13 (4 routes)
    ('-E13', 'E24', 240),
    ('-E13', 'E6', 200),
    ('-E13', 'E15', 220),
    ('-E13', 'E10', 180),
    
    # From -E21 (4 routes)
    ('-E21', 'E6', 200),
    ('-E21', 'E15', 360),  # Major corridor
    ('-E21', 'E12', 160),
    ('-E21', 'E11', 180),
    
    # From -E24 (4 routes)
    ('-E24', 'E11', 200),
    ('-E24', 'E15', 360),  # Major corridor
    ('-E24', 'E13', 180),
    ('-E24', 'E6', 200),
    
    # From -E6 (7 routes)
    ('-E6', 'E24', 300),  # Major route
    ('-E6', 'E9', 160),
    ('-E6', 'E21', 280),
    ('-E6', 'E13', 180),
    ('-E6', 'E12', 160),
    ('-E6', 'E11', 180),
    ('-E6', 'E15', 320),  # Major corridor
]

# Total: 39 routes
DURATION_HOURS = 12
DURATION_SECONDS = DURATION_HOURS * 3600  # 43200 seconds


def generate_route_file() -> str:
    """Generate route file XML content for 12-hour heavy traffic."""
    
    lines = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('')
    lines.append('<!-- 12-Hour Heavy Traffic Scenario for Extended Training -->')
    lines.append('<!-- 39 validated routes from original.rou.xml -->')
    lines.append('<!-- Heavy traffic: ~8,240 vehicles/hour sustained -->')
    lines.append(f'<!-- Total duration: {DURATION_HOURS} hours ({DURATION_SECONDS} seconds) -->')
    lines.append('')
    lines.append('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">')
    lines.append('')
    
    # Vehicle type
    lines.append('    <!-- Vehicle Type -->')
    lines.append('    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="50"/>')
    lines.append('')
    
    # Generate flows
    lines.append(f'    <!-- Heavy Traffic Flows (39 routes, {DURATION_HOURS} hours) -->')
    
    current_origin = None
    total_veh_per_hour = 0
    
    for idx, (from_edge, to_edge, veh_per_hour) in enumerate(ROUTES):
        # Add comment for new origin
        if from_edge != current_origin:
            if current_origin is not None:
                lines.append('')
            lines.append(f'    <!-- Routes from {from_edge} -->')
            current_origin = from_edge
        
        flow_id = f"f_{idx}"
        total_veh_per_hour += veh_per_hour
        
        # Flow runs for entire 12 hours
        lines.append(f'    <flow id="{flow_id}" '
                    f'begin="0" end="{DURATION_SECONDS}" '
                    f'from="{from_edge}" to="{to_edge}" '
                    f'vehsPerHour="{veh_per_hour}" type="car"/>')
    
    lines.append('')
    lines.append('</routes>')
    
    total_vehicles = total_veh_per_hour * DURATION_HOURS
    print(f"  Traffic rate: {total_veh_per_hour} vehicles/hour")
    print(f"  Total vehicles (12h): ~{total_vehicles:,} vehicles")
    
    return '\n'.join(lines)


def generate_sumocfg() -> str:
    """Generate SUMO configuration file for 12-hour scenario."""
    return f'''<?xml version="1.0" encoding="UTF-8"?>

<!-- 12-Hour Heavy Traffic Configuration -->

<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="k1.net.xml"/>
        <route-files value="k1_routes_12h_heavy.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="{DURATION_SECONDS}"/>
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


def validate_routes() -> bool:
    """Validate routes using SUMO (quick check)."""
    config_file = OUTPUT_DIR / 'k1_12h_heavy.sumocfg'
    
    try:
        # Just validate first 100 seconds
        result = subprocess.run(
            ['sumo', '-c', str(config_file), '--no-step-log', '--no-warnings', '--end', '100'],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            print("  ✓ Routes validated successfully")
            return True
        else:
            print("  ✗ Validation failed:")
            print(result.stderr[:500])
            return False
    except subprocess.TimeoutExpired:
        print("  ✗ Validation timed out")
        return False
    except FileNotFoundError:
        print("  Warning: SUMO not found. Skipping validation.")
        return True


def main():
    parser = argparse.ArgumentParser(description='Generate 12-hour heavy traffic scenario')
    parser.add_argument('--validate', action='store_true', help='Validate generated routes')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("12-Hour Heavy Traffic Scenario Generator")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Total routes: {len(ROUTES)}")
    print(f"Duration: {DURATION_HOURS} hours ({DURATION_SECONDS:,} seconds)")
    print()
    
    # Generate route file
    route_xml = generate_route_file()
    route_file = OUTPUT_DIR / 'k1_routes_12h_heavy.rou.xml'
    with open(route_file, 'w') as f:
        f.write(route_xml)
    print(f"  ✓ Created: {route_file.name}")
    
    # Generate config file
    config_xml = generate_sumocfg()
    config_file = OUTPUT_DIR / 'k1_12h_heavy.sumocfg'
    with open(config_file, 'w') as f:
        f.write(config_xml)
    print(f"  ✓ Created: {config_file.name}")
    
    # Validate if requested
    if args.validate:
        validate_routes()
    
    print()
    print("=" * 60)
    print("To use in training:")
    print("  python train_12h_scenario.py")
    print()
    print("To validate manually:")
    print("  sumo -c k1_12h_heavy.sumocfg --start --quit-on-end")
    print("=" * 60)


if __name__ == '__main__':
    main()
