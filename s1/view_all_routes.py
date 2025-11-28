#!/usr/bin/env python3
"""
Generate a test route file with one vehicle per route for visualization
This creates a minimal route file showing all possible routes clearly in SUMO GUI
"""

import xml.etree.ElementTree as ET
from collections import defaultdict

def extract_unique_routes(input_file, output_file):
    """Extract one vehicle per unique route for visualization"""
    
    # Parse input file
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    tree = ET.parse(input_file)
    root = tree.getroot()
    
    # Create new route file
    new_root = ET.Element('routes')
    new_root.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
    new_root.set('xsi:noNamespaceSchemaLocation', 'http://sumo.dlr.de/xsd/routes_file.xsd')
    
    # Add comment
    comment = ET.Comment(' Visualization file - one vehicle per unique route ')
    new_root.append(comment)
    
    # Copy vTypes
    for vtype in root.findall('.//vType'):
        new_root.append(vtype)
    
    # Track unique routes
    seen_routes = set()
    vehicle_id = 0
    
    # Process all flows (including commented ones using regex on raw content)
    import re
    flow_pattern = r'<flow\s+id="([^"]+)"\s+[^>]*from="([^"]+)"\s+[^>]*to="([^"]+)"(?:\s+via="([^"]*)")?[^>]*type="([^"]+)"[^>]*/>'
    
    for match in re.finditer(flow_pattern, content):
        flow_id = match.group(1)
        from_edge = match.group(2)
        to_edge = match.group(3)
        via_attr = match.group(4) or ''
        vtype = match.group(5)
        
        # Create unique route signature
        route_sig = (from_edge, to_edge, via_attr, vtype)
        
        if route_sig not in seen_routes:
            seen_routes.add(route_sig)
            
            # Create a single vehicle for this route
            vehicle = ET.SubElement(new_root, 'vehicle')
            vehicle.set('id', f'test_{flow_id}')
            vehicle.set('depart', str(vehicle_id * 2))  # Stagger departures
            vehicle.set('from', from_edge)
            vehicle.set('to', to_edge)
            if via_attr:
                vehicle.set('via', via_attr)
            vehicle.set('type', vtype)
            vehicle.set('color', get_color_for_route(flow_id))
            
            vehicle_id += 1
    
    # Write output
    tree = ET.ElementTree(new_root)
    ET.indent(tree, space='    ')
    tree.write(output_file, encoding='UTF-8', xml_declaration=True)
    
    print(f"✓ Created {output_file} with {vehicle_id} test vehicles")
    print(f"  Unique routes found: {len(seen_routes)}")

def get_color_for_route(flow_id):
    """Assign colors based on route type"""
    if flow_id.startswith('f_'):
        return "0,0,255"  # Blue for regular flows
    elif flow_id.startswith('b_'):
        return "255,165,0"  # Orange for buses
    elif flow_id.startswith('t_'):
        return "0,128,0"  # Green for trucks
    else:
        return "255,0,0"  # Red for others

if __name__ == '__main__':
    input_file = 'd:/Codes/Projects/Traffic-Light-Automation/s1/k1_routes_3h_varying.rou.xml'
    output_file = 'd:/Codes/Projects/Traffic-Light-Automation/s1/k1_routes_visual.rou.xml'
    
    extract_unique_routes(input_file, output_file)
    
    print("\n" + "="*80)
    print("To view routes in SUMO GUI:")
    print("="*80)
    print("\n  cd d:\\Codes\\Projects\\Traffic-Light-Automation\\s1")
    print("  sumo-gui -c view_routes.sumocfg -r k1_routes_visual.rou.xml")
    print("\nOR double-click view_routes.sumocfg in Windows Explorer")
    print("\nIn SUMO GUI:")
    print("  - Press SPACE to start simulation")
    print("  - Right-click vehicles to see their route")
    print("  - View -> Show Route to highlight paths")
    print("  - Colors: Blue=cars, Orange=buses, Green=trucks")
    print("="*80)
