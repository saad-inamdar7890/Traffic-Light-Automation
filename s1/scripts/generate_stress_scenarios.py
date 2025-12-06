#!/usr/bin/env python3
"""
Generate stress scenario route files by scaling/modifying flows from the weekday route file.

Scenarios:
- gridlock: 2.5x all traffic (extreme congestion)
- incident: asymmetric - reduce some routes, increase alternates
- spike: normal until hour 3, then 4x surge, then gradual recovery
- night_surge: very low hours 0-3, then massive surge hours 4-6

Usage:
    python generate_stress_scenarios.py
"""

import xml.etree.ElementTree as ET
from pathlib import Path
import copy


def parse_routes(path: Path):
    """Parse route file and return (tree, root, vtypes, flows)"""
    tree = ET.parse(path)
    root = tree.getroot()
    
    vtypes = []
    flows = []
    
    for child in root:
        if child.tag == 'vType':
            vtypes.append(child)
        elif child.tag == 'flow':
            flows.append(child)
    
    return tree, root, vtypes, flows


def get_time_period(flow):
    """Determine time period from flow ID or begin/end times"""
    flow_id = flow.attrib.get('id', '')
    begin = int(float(flow.attrib.get('begin', 0)))
    
    # Check ID for period hints
    if 'early_morning' in flow_id or begin < 3600:
        return 'early_morning'
    elif 'morning_peak' in flow_id or begin < 7200:
        return 'morning_peak'
    elif 'midday' in flow_id or begin < 10800:
        return 'midday'
    elif 'afternoon' in flow_id or begin < 14400:
        return 'afternoon'
    elif 'evening_peak' in flow_id or begin < 18000:
        return 'evening_peak'
    else:
        return 'night'


def scale_vph(flow, multiplier):
    """Scale vehsPerHour by multiplier, minimum 1"""
    vph = flow.attrib.get('vehsPerHour')
    if vph:
        new_vph = max(1, int(round(float(vph) * multiplier)))
        flow.attrib['vehsPerHour'] = str(new_vph)


def generate_gridlock(vtypes, flows, out_path: Path):
    """Generate gridlock scenario: 2.5x all traffic"""
    root = ET.Element('routes', {
        'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance',
        'xsi:noNamespaceSchemaLocation': 'http://sumo.dlr.de/xsd/routes_file.xsd'
    })
    
    # Add comment
    root.append(ET.Comment(' GRIDLOCK STRESS SCENARIO: 2.5x peak traffic '))
    
    # Add vtypes
    for vt in vtypes:
        root.append(copy.deepcopy(vt))
    
    # Scale all flows by 2.5x
    for flow in flows:
        new_flow = copy.deepcopy(flow)
        new_flow.attrib['id'] = 'grid_' + new_flow.attrib['id']
        scale_vph(new_flow, 2.5)
        root.append(new_flow)
    
    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(out_path, encoding='utf-8', xml_declaration=True)
    print(f"Generated: {out_path}")


def generate_incident(vtypes, flows, out_path: Path):
    """Generate incident scenario: asymmetric flow (some routes reduced, others increased)"""
    root = ET.Element('routes', {
        'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance',
        'xsi:noNamespaceSchemaLocation': 'http://sumo.dlr.de/xsd/routes_file.xsd'
    })
    
    root.append(ET.Comment(' INCIDENT SCENARIO: Asymmetric traffic flow '))
    
    for vt in vtypes:
        root.append(copy.deepcopy(vt))
    
    for flow in flows:
        new_flow = copy.deepcopy(flow)
        new_flow.attrib['id'] = 'inc_' + new_flow.attrib['id']
        
        flow_id = flow.attrib.get('id', '').lower()
        
        # Reduce E9->* routes (simulating blocked direction)
        if 'e9_e13' in flow_id or 'e9_e6' in flow_id:
            scale_vph(new_flow, 0.3)  # 70% reduction
        # Increase alternate routes (rerouted traffic)
        elif 'e10_' in flow_id or 'e6_e21' in flow_id:
            scale_vph(new_flow, 1.8)  # 80% increase
        # Others stay roughly normal
        else:
            scale_vph(new_flow, 1.0)
        
        root.append(new_flow)
    
    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(out_path, encoding='utf-8', xml_declaration=True)
    print(f"Generated: {out_path}")


def generate_spike(vtypes, flows, out_path: Path):
    """Generate spike scenario: sudden surge at hour 3 (10800-14400s)"""
    root = ET.Element('routes', {
        'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance',
        'xsi:noNamespaceSchemaLocation': 'http://sumo.dlr.de/xsd/routes_file.xsd'
    })
    
    root.append(ET.Comment(' SPIKE SCENARIO: Sudden demand surge at hour 3 '))
    
    for vt in vtypes:
        root.append(copy.deepcopy(vt))
    
    for flow in flows:
        new_flow = copy.deepcopy(flow)
        new_flow.attrib['id'] = 'spk_' + new_flow.attrib['id']
        
        period = get_time_period(flow)
        
        # Time-based multipliers
        if period in ['early_morning', 'morning_peak']:
            scale_vph(new_flow, 0.7)  # Lower before surge
        elif period == 'midday':
            scale_vph(new_flow, 4.0)  # MASSIVE SURGE
        elif period == 'afternoon':
            scale_vph(new_flow, 2.0)  # Recovery period
        elif period == 'evening_peak':
            scale_vph(new_flow, 1.2)  # Slightly elevated
        else:
            scale_vph(new_flow, 0.8)  # Back to low
        
        root.append(new_flow)
    
    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(out_path, encoding='utf-8', xml_declaration=True)
    print(f"Generated: {out_path}")


def generate_night_surge(vtypes, flows, out_path: Path):
    """Generate night surge scenario: very low early, massive surge hours 4-6"""
    root = ET.Element('routes', {
        'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance',
        'xsi:noNamespaceSchemaLocation': 'http://sumo.dlr.de/xsd/routes_file.xsd'
    })
    
    root.append(ET.Comment(' NIGHT SURGE SCENARIO: Late-night event traffic '))
    
    for vt in vtypes:
        root.append(copy.deepcopy(vt))
    
    for flow in flows:
        new_flow = copy.deepcopy(flow)
        new_flow.attrib['id'] = 'ngt_' + new_flow.attrib['id']
        
        period = get_time_period(flow)
        
        # Very quiet early, then surge
        if period == 'early_morning':
            scale_vph(new_flow, 0.2)  # Very quiet
        elif period == 'morning_peak':
            scale_vph(new_flow, 0.15)  # Even quieter
        elif period == 'midday':
            scale_vph(new_flow, 0.15)  # Still quiet
        elif period == 'afternoon':
            scale_vph(new_flow, 1.5)  # Building up
        elif period == 'evening_peak':
            scale_vph(new_flow, 3.5)  # SURGE
        else:  # night
            scale_vph(new_flow, 3.0)  # Continued surge
        
        root.append(new_flow)
    
    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(out_path, encoding='utf-8', xml_declaration=True)
    print(f"Generated: {out_path}")


def main():
    script_dir = Path(__file__).parent
    s1_dir = script_dir.parent
    
    weekday_path = s1_dir / 'k1_routes_6h_weekday.rou.xml'
    
    if not weekday_path.exists():
        print(f"Error: Weekday route file not found: {weekday_path}")
        return 1
    
    print(f"Reading base routes from: {weekday_path}")
    tree, root, vtypes, flows = parse_routes(weekday_path)
    print(f"  Found {len(vtypes)} vehicle types, {len(flows)} flows")
    
    # Generate all stress scenarios
    generate_gridlock(vtypes, flows, s1_dir / 'k1_routes_6h_gridlock.rou.xml')
    generate_incident(vtypes, flows, s1_dir / 'k1_routes_6h_incident.rou.xml')
    generate_spike(vtypes, flows, s1_dir / 'k1_routes_6h_spike.rou.xml')
    generate_night_surge(vtypes, flows, s1_dir / 'k1_routes_6h_night_surge.rou.xml')
    
    print("\nAll stress scenarios generated successfully!")
    print("Run SUMO validation:")
    print("  sumo -c k1_6h_gridlock.sumocfg --end 60")
    print("  sumo -c k1_6h_incident.sumocfg --end 60")
    print("  sumo -c k1_6h_spike.sumocfg --end 60")
    print("  sumo -c k1_6h_night_surge.sumocfg --end 60")
    
    return 0


if __name__ == '__main__':
    exit(main())
