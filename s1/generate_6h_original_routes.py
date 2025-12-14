#!/usr/bin/env python3
"""
Generate 6-Hour Evaluation Scenario with Original Routes
=========================================================

Creates a 6-hour evaluation scenario using ALL 39 validated routes from original.rou.xml.
This ensures fair comparison between Fixed-Time and MAPPO on the same routes used for training.

Duration: 6 hours (21600 seconds)
Routes: All 39 original routes with time-varying traffic

Time Periods:
- Early Morning (0-3600): 6am-7am - Building up
- Morning Peak (3600-7200): 7am-8am - Rush hour
- Midday (7200-10800): 8am-9am - Post-rush
- Afternoon (10800-14400): 9am-10am - Moderate
- Evening Peak (14400-18000): 10am-11am - Second rush
- Night (18000-21600): 11am-12pm - Tapering off

Usage:
    python generate_6h_original_routes.py
    python generate_6h_original_routes.py --validate
"""

import argparse
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR

# =============================================================================
# TIME PERIODS (6 hours = 21600 seconds)
# =============================================================================

TIME_PERIODS = [
    # (start_sec, end_sec, name, multiplier)
    (0,     3600,  'early_morning', 0.60),   # Building up
    (3600,  7200,  'morning_peak',  1.20),   # Rush hour
    (7200,  10800, 'midday',        0.80),   # Post-rush
    (10800, 14400, 'afternoon',     0.65),   # Moderate
    (14400, 18000, 'evening_peak',  1.50),   # Evening rush
    (18000, 21600, 'night',         0.40),   # Tapering off
]

# =============================================================================
# ALL 39 VALIDATED ROUTES FROM original.rou.xml
# Format: (from_edge, to_edge, base_veh_per_hour)
# =============================================================================

ROUTES = [
    # From -E9 (5 routes)
    ('-E9', 'E13', 100),
    ('-E9', 'E6', 100),
    ('-E9', 'E24', 120),
    ('-E9', 'E10', 80),
    ('-E9', 'E21', 90),
    
    # From -E10 (6 routes)
    ('-E10', 'E6', 100),
    ('-E10', 'E21', 150),
    ('-E10', 'E15', 180),
    ('-E10', 'E24', 120),
    ('-E10', 'E13', 80),
    ('-E10', 'E15', 180),  # Duplicate in original - major corridor
    
    # From -E11 (5 routes)
    ('-E11', 'E6', 90),
    ('-E11', 'E9', 80),
    ('-E11', 'E13', 70),
    ('-E11', 'E15', 100),
    ('-E11', 'E24', 150),
    
    # From -E12 (4 routes)
    ('-E12', 'E9', 80),
    ('-E12', 'E6', 90),
    ('-E12', 'E15', 100),
    ('-E12', 'E24', 110),
    
    # From -E13 (4 routes)
    ('-E13', 'E24', 120),
    ('-E13', 'E6', 100),
    ('-E13', 'E15', 110),
    ('-E13', 'E10', 90),
    
    # From -E21 (4 routes)
    ('-E21', 'E6', 100),
    ('-E21', 'E15', 180),
    ('-E21', 'E11', 90),
    ('-E21', 'E12', 80),
    
    # From -E24 (4 routes)
    ('-E24', 'E11', 100),
    ('-E24', 'E15', 180),
    ('-E24', 'E13', 90),
    ('-E24', 'E6', 100),
    
    # From -E6 (7 routes)
    ('-E6', 'E24', 150),
    ('-E6', 'E9', 80),
    ('-E6', 'E21', 140),
    ('-E6', 'E13', 90),
    ('-E6', 'E12', 80),
    ('-E6', 'E11', 90),
    ('-E6', 'E15', 160),
]

DURATION_HOURS = 6
DURATION_SECONDS = DURATION_HOURS * 3600  # 21600 seconds


def generate_route_file() -> str:
    """Generate route file XML content for 6-hour evaluation with original routes."""
    
    lines = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('')
    lines.append('<!-- 6-Hour Evaluation Scenario with Original Routes -->')
    lines.append('<!-- All 39 validated routes from original.rou.xml -->')
    lines.append('<!-- For fair comparison: Fixed-Time vs MAPPO -->')
    lines.append(f'<!-- Duration: {DURATION_HOURS} hours ({DURATION_SECONDS} seconds) -->')
    lines.append('')
    lines.append('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">')
    lines.append('')
    
    # Vehicle type
    lines.append('    <!-- Vehicle Type -->')
    lines.append('    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="50"/>')
    lines.append('')
    
    # Generate flows sorted by begin time (SUMO requirement!)
    # Collect all flows first, then sort by begin time
    lines.append('    <!-- Traffic Flows (39 routes x 6 periods = 234 flows) -->')
    lines.append('    <!-- SORTED BY BEGIN TIME for SUMO compatibility -->')
    
    all_flows = []
    total_vehicles = 0
    
    for idx, (from_edge, to_edge, base_rate) in enumerate(ROUTES):
        for start, end, period_name, multiplier in TIME_PERIODS:
            veh_per_hour = int(base_rate * multiplier)
            duration_hours = (end - start) / 3600
            vehicles_this_period = veh_per_hour * duration_hours
            total_vehicles += vehicles_this_period
            
            flow_id = f"f_{idx}_{period_name}"
            
            all_flows.append({
                'begin': start,
                'end': end,
                'from': from_edge,
                'to': to_edge,
                'vph': veh_per_hour,
                'id': flow_id,
                'period': period_name
            })
    
    # Sort by begin time, then by flow id for consistent ordering
    all_flows.sort(key=lambda x: (x['begin'], x['id']))
    
    # Group by period for readability
    current_period = None
    for flow in all_flows:
        if flow['period'] != current_period:
            if current_period is not None:
                lines.append('')
            period_start = flow['begin']
            period_end = flow['end']
            lines.append(f'    <!-- {flow["period"].replace("_", " ").title()} ({period_start}-{period_end}s) -->')
            current_period = flow['period']
        
        lines.append(f'    <flow id="{flow["id"]}" type="car" '
                    f'begin="{flow["begin"]}" end="{flow["end"]}" '
                    f'from="{flow["from"]}" to="{flow["to"]}" '
                    f'vehsPerHour="{flow["vph"]}"/>')
    
    flow_count = len(all_flows)
    
    lines.append('')
    lines.append('</routes>')
    
    print(f"  Total flows: {flow_count}")
    print(f"  Estimated vehicles: ~{int(total_vehicles):,}")
    
    return '\n'.join(lines)


def generate_sumocfg() -> str:
    """Generate SUMO configuration file for 6-hour evaluation."""
    return f'''<?xml version="1.0" encoding="UTF-8"?>

<!-- 6-Hour Evaluation Configuration with Original Routes -->
<!-- For fair Fixed-Time vs MAPPO comparison -->

<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="k1.net.xml"/>
        <route-files value="k1_routes_6h_original.rou.xml"/>
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
    """Validate routes using SUMO."""
    config_file = OUTPUT_DIR / 'k1_6h_original.sumocfg'
    
    try:
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
    parser = argparse.ArgumentParser(description='Generate 6-hour evaluation scenario with original routes')
    parser.add_argument('--validate', action='store_true', help='Validate generated routes')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("6-Hour Evaluation Scenario Generator (Original Routes)")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Total routes: {len(ROUTES)}")
    print(f"Time periods: {len(TIME_PERIODS)}")
    print(f"Duration: {DURATION_HOURS} hours ({DURATION_SECONDS:,} seconds)")
    print()
    
    # Generate route file
    route_xml = generate_route_file()
    route_file = OUTPUT_DIR / 'k1_routes_6h_original.rou.xml'
    with open(route_file, 'w') as f:
        f.write(route_xml)
    print(f"  ✓ Created: {route_file.name}")
    
    # Generate config file
    config_xml = generate_sumocfg()
    config_file = OUTPUT_DIR / 'k1_6h_original.sumocfg'
    with open(config_file, 'w') as f:
        f.write(config_xml)
    print(f"  ✓ Created: {config_file.name}")
    
    # Validate if requested
    if args.validate:
        print()
        validate_routes()
    
    print()
    print("=" * 60)
    print("To evaluate Fixed-Time vs MAPPO:")
    print("  python evaluate_fixed_vs_mappo.py --checkpoint YOUR_CHECKPOINT --scenario original")
    print()
    print("To validate manually:")
    print("  sumo -c k1_6h_original.sumocfg --start --quit-on-end")
    print("=" * 60)


if __name__ == '__main__':
    main()
