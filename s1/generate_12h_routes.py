#!/usr/bin/env python3
"""
Generate 12-Hour Realistic Traffic Route Files
==============================================

Uses the proper validated routes from original.rou.xml with time-varying flow rates.
All 39 routes are validated to have proper paths (no U-turns, no single-edge routes).

12-hour scenario (6am-6pm) with 6 time periods:
- Early Morning (6-8am): Building toward rush hour
- Morning Peak (8-9am): Rush hour
- Late Morning (9am-12pm): Post-rush moderate
- Lunch (12-2pm): Steady traffic
- Early Afternoon (2-4pm): Quiet period  
- Late Afternoon (4-6pm): Evening rush hour (2x traffic)

Usage:
    python generate_12h_routes.py --scenario weekday
    python generate_12h_routes.py --scenario all --validate
"""

import argparse
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR

# =============================================================================
# TIME PERIODS (12 hours = 43200 seconds, 6am-6pm)
# =============================================================================

TIME_PERIODS = [
    # (start_sec, end_sec, name, description)
    (0,     7200,  'early_morning',  '6am-8am'),
    (7200,  10800, 'morning_peak',   '8am-9am'),
    (10800, 21600, 'late_morning',   '9am-12pm'),
    (21600, 28800, 'lunch',          '12pm-2pm'),
    (28800, 36000, 'early_afternoon','2pm-4pm'),
    (36000, 43200, 'late_afternoon', '4pm-6pm'),
]

# =============================================================================
# VALIDATED ROUTES FROM original.rou.xml
# These are proper routes with no U-turns or single-edge paths
# Format: (id, from_edge, to_edge, base_veh_per_hour)
# =============================================================================

ROUTES = [
    # From -E9 (5 routes)
    ('f_0', '-E9', 'E13', 100),
    ('f_1', '-E9', 'E6', 100),
    ('f_2', '-E9', 'E24', 120),
    ('f_3', '-E9', 'E10', 80),
    ('f_4', '-E9', 'E21', 90),
    
    # From -E10 (5 routes)
    ('f_5', '-E10', 'E6', 100),
    ('f_6', '-E10', 'E21', 150),
    ('f_7', '-E10', 'E15', 180),
    ('f_8', '-E10', 'E24', 120),
    ('f_9', '-E10', 'E13', 80),
    ('f_10', '-E10', 'E15', 180),  # Major corridor
    
    # From -E11 (5 routes)
    ('f_11', '-E11', 'E6', 90),
    ('f_12', '-E11', 'E9', 80),
    ('f_13', '-E11', 'E13', 70),
    ('f_14', '-E11', 'E15', 100),
    ('f_15', '-E11', 'E24', 150),
    
    # From -E12 (4 routes)
    ('f_16', '-E12', 'E9', 80),
    ('f_17', '-E12', 'E6', 90),
    ('f_18', '-E12', 'E15', 100),
    ('f_19', '-E12', 'E24', 110),
    
    # From -E13 (4 routes)
    ('f_20', '-E13', 'E24', 120),
    ('f_21', '-E13', 'E6', 100),
    ('f_22', '-E13', 'E15', 110),
    ('f_23', '-E13', 'E10', 90),
    
    # From -E21 (4 routes)
    ('f_24', '-E21', 'E6', 100),
    ('f_25', '-E21', 'E15', 180),  # Major corridor
    ('f_26', '-E21', 'E11', 90),
    ('f_27', '-E21', 'E12', 80),
    
    # From -E24 (4 routes)
    ('f_28', '-E24', 'E11', 100),
    ('f_29', '-E24', 'E15', 180),  # Major corridor
    ('f_30', '-E24', 'E13', 90),
    ('f_31', '-E24', 'E6', 100),
    
    # From -E6 (7 routes)
    ('f_32', '-E6', 'E24', 150),
    ('f_33', '-E6', 'E9', 80),
    ('f_34', '-E6', 'E21', 140),
    ('f_35', '-E6', 'E13', 90),
    ('f_36', '-E6', 'E12', 80),
    ('f_37', '-E6', 'E11', 90),
    ('f_38', '-E6', 'E15', 160),  # Major corridor
]

# Total: 39 routes (matching original.rou.xml)

# =============================================================================
# SCENARIO MULTIPLIERS
# Each list has 6 values for the 6 time periods
# [early_morning, morning_peak, late_morning, lunch, early_afternoon, late_afternoon]
# =============================================================================

SCENARIO_MULTIPLIERS = {
    # Standard weekday: Strong morning peak and evening rush hour (2x)
    'weekday': [0.60, 1.20, 0.65, 0.55, 0.50, 2.00],
    
    # Weekend: Later morning peak, more leisure traffic
    'weekend': [0.40, 0.55, 0.80, 0.85, 0.75, 1.20],
    
    # Friday: Normal morning, heavy afternoon buildup
    'friday': [0.65, 1.20, 0.65, 0.60, 0.80, 2.20],
    
    # Event day: Heavy afternoon and evening
    'event': [0.60, 1.10, 0.70, 1.00, 1.50, 2.50],
}


def generate_route_file(scenario_name: str) -> str:
    """Generate route file XML content for a 12-hour scenario."""
    if scenario_name not in SCENARIO_MULTIPLIERS:
        raise ValueError(f"Unknown scenario: {scenario_name}")
    
    multipliers = SCENARIO_MULTIPLIERS[scenario_name]
    
    lines = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('')
    lines.append(f'<!-- 12-Hour {scenario_name.title()} Traffic Scenario (6am-6pm) -->')
    lines.append('<!-- Generated from validated routes in original.rou.xml -->')
    lines.append('<!-- 39 routes x 6 time periods = 234 flows -->')
    lines.append('')
    lines.append('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">')
    lines.append('')
    
    # Vehicle type (single type for simplicity)
    lines.append('    <!-- Vehicle Type -->')
    lines.append('    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="50"/>')
    lines.append('')
    
    # Generate flows for each route and time period
    lines.append('    <!-- Traffic Flows (39 routes x 6 periods) -->')
    lines.append('')
    
    flow_count = 0
    current_origin = None
    
    for flow_id, from_edge, to_edge, base_rate in ROUTES:
        # Add comment for new origin
        if from_edge != current_origin:
            if current_origin is not None:
                lines.append('')
            lines.append(f'    <!-- Routes from {from_edge} -->')
            current_origin = from_edge
        
        for period_idx, (start, end, period_name, desc) in enumerate(TIME_PERIODS):
            mult = multipliers[period_idx]
            veh_per_hour = max(1, base_rate * mult)  # Minimum 1 veh/hr
            
            unique_flow_id = f"{flow_id}_{period_name}"
            
            lines.append(f'    <flow id="{unique_flow_id}" '
                        f'begin="{start}" end="{end}" '
                        f'from="{from_edge}" to="{to_edge}" '
                        f'vehsPerHour="{veh_per_hour:.0f}" type="car"/>')
            flow_count += 1
    
    lines.append('')
    lines.append('</routes>')
    
    return '\n'.join(lines), flow_count


def generate_sumocfg(scenario_name: str) -> str:
    """Generate SUMO configuration file for 12-hour scenario."""
    return f'''<?xml version="1.0" encoding="UTF-8"?>

<!-- 12-Hour {scenario_name.title()} Traffic Configuration (6am-6pm) -->

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
    config_file = OUTPUT_DIR / f'k1_12h_{scenario_name}.sumocfg'
    
    try:
        result = subprocess.run(
            ['sumo', '-c', str(config_file), '--no-step-log', '--no-warnings', '--end', '100'],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            print(f"  ✓ Routes validated successfully")
            return True
        else:
            print(f"  ✗ Validation failed:")
            print(result.stderr[:500])
            return False
    except subprocess.TimeoutExpired:
        print(f"  ✗ Validation timed out")
        return False
    except FileNotFoundError:
        print("  Warning: SUMO not found. Skipping validation.")
        return True


def print_traffic_summary(scenario_name: str):
    """Print a summary of traffic rates for each period."""
    multipliers = SCENARIO_MULTIPLIERS[scenario_name]
    
    # Calculate total base rate
    total_base = sum(r[3] for r in ROUTES)
    
    print(f"\n  Traffic Summary:")
    print(f"  {'Period':<20} {'Time':<10} {'Mult':>6} {'Total veh/hr':>12}")
    print(f"  {'-'*50}")
    
    for period_idx, (start, end, period_name, desc) in enumerate(TIME_PERIODS):
        mult = multipliers[period_idx]
        total_rate = total_base * mult
        print(f"  {period_name:<20} {desc:<10} {mult:>5.2f}x {total_rate:>12.0f}")
    
    print(f"\n  Base total: {total_base} veh/hr across {len(ROUTES)} routes")


def main():
    parser = argparse.ArgumentParser(description='Generate 12-hour traffic route files')
    parser.add_argument('--scenario', choices=['weekday', 'weekend', 'friday', 'event', 'all'],
                        default='weekday', help='Scenario to generate')
    parser.add_argument('--validate', action='store_true', help='Validate generated routes')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    
    args = parser.parse_args()
    
    global OUTPUT_DIR
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    scenarios = ['weekday', 'weekend', 'friday', 'event'] if args.scenario == 'all' else [args.scenario]
    
    print("=" * 60)
    print("12-Hour Traffic Route Generator")
    print("Using validated routes from original.rou.xml")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Total routes: {len(ROUTES)}")
    print(f"Time periods: {len(TIME_PERIODS)}")
    print(f"Scenarios: {', '.join(scenarios)}")
    
    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"Generating {scenario} scenario...")
        print(f"{'='*60}")
        
        # Generate route file
        route_xml, flow_count = generate_route_file(scenario)
        route_file = OUTPUT_DIR / f'k1_routes_12h_{scenario}.rou.xml'
        with open(route_file, 'w') as f:
            f.write(route_xml)
        print(f"  ✓ Created: {route_file.name}")
        print(f"  ✓ Generated {flow_count} flows ({len(ROUTES)} routes x {len(TIME_PERIODS)} periods)")
        
        # Generate config file
        config_xml = generate_sumocfg(scenario)
        config_file = OUTPUT_DIR / f'k1_12h_{scenario}.sumocfg'
        with open(config_file, 'w') as f:
            f.write(config_xml)
        print(f"  ✓ Created: {config_file.name}")
        
        # Print traffic summary
        print_traffic_summary(scenario)
        
        # Validate if requested
        if args.validate:
            validate_routes(scenario)
    
    print(f"\n{'='*60}")
    print("Route generation complete!")
    print("=" * 60)
    print("\nTo use in training:")
    print("  python train_12h_scenario.py --scenario weekday")
    print("\nTo validate routes manually:")
    print("  sumo -c k1_12h_weekday.sumocfg --start --quit-on-end --end 100")


if __name__ == '__main__':
    main()
