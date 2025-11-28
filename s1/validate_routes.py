#!/usr/bin/env python3
"""
Route Validation Tool for K1 Network
Checks all routes in k1_routes_3h_varying.rou.xml against k1.net.xml
Generates a detailed report of valid/invalid routes with suggested fixes
"""

import xml.etree.ElementTree as ET
from collections import defaultdict
import sys

def parse_network(net_file):
    """Parse SUMO network file and build edge connectivity graph"""
    tree = ET.parse(net_file)
    root = tree.getroot()
    
    # Build edge dictionary: edge_id -> (from_junction, to_junction)
    edges = {}
    for edge in root.findall('.//edge'):
        edge_id = edge.get('id')
        from_junction = edge.get('from')
        to_junction = edge.get('to')
        if edge_id and from_junction and to_junction:
            edges[edge_id] = (from_junction, to_junction)
    
    # Build connectivity graph: junction -> list of outgoing edges
    outgoing = defaultdict(list)
    for edge_id, (from_j, to_j) in edges.items():
        outgoing[from_j].append(edge_id)
    
    return edges, outgoing

def find_path(from_edge, to_edge, edges, outgoing, max_depth=10):
    """
    Find a path from from_edge to to_edge using BFS
    Returns: (is_reachable, path_edges) where path_edges is list of intermediate edges
    """
    if from_edge not in edges or to_edge not in edges:
        return False, []
    
    # Get junctions
    _, from_junction = edges[from_edge]  # We start at the end of from_edge
    to_junction, _ = edges[to_edge]      # We need to reach the start of to_edge
    
    # BFS to find path
    queue = [(from_junction, [])]
    visited = {from_junction}
    
    while queue:
        current_junction, path = queue.pop(0)
        
        if len(path) > max_depth:
            continue
        
        # Check if we reached the target
        if current_junction == to_junction:
            return True, path
        
        # Explore outgoing edges
        for edge_id in outgoing[current_junction]:
            if edge_id in edges:
                _, next_junction = edges[edge_id]
                if next_junction not in visited:
                    visited.add(next_junction)
                    queue.append((next_junction, path + [edge_id]))
    
    return False, []

def validate_route_with_via(from_edge, to_edge, via_edges, edges, outgoing):
    """
    Validate a route with via edges
    Returns: (is_valid, error_message, suggested_via)
    """
    if not via_edges:
        # No via specified, check direct reachability
        reachable, path = find_path(from_edge, to_edge, edges, outgoing)
        if reachable:
            suggested = ' '.join(path) if path else '(direct)'
            return True, None, suggested
        else:
            # Try to find any path
            reachable, path = find_path(from_edge, to_edge, edges, outgoing, max_depth=20)
            if reachable:
                suggested = ' '.join(path)
                return False, f"No direct route, needs via", suggested
            else:
                return False, f"No path exists", None
    
    # Validate via route: from_edge -> via[0] -> via[1] -> ... -> to_edge
    current = from_edge
    all_edges = via_edges + [to_edge]
    
    for i, next_edge in enumerate(all_edges):
        reachable, path = find_path(current, next_edge, edges, outgoing)
        if not reachable:
            return False, f"Cannot reach {next_edge} from {current}", None
        current = next_edge
    
    return True, None, None

def parse_routes(route_file):
    """Parse route file and extract all flows (active and commented)"""
    import re
    
    flows = []
    
    # Read raw file to get commented flows too
    with open(route_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all flow tags (commented or not)
    flow_pattern = r'<flow\s+id="([^"]+)"\s+[^>]*from="([^"]+)"\s+[^>]*to="([^"]+)"(?:\s+via="([^"]*)")?[^>]*/?>'
    
    for match in re.finditer(flow_pattern, content):
        flow_id = match.group(1)
        from_edge = match.group(2)
        to_edge = match.group(3)
        via_attr = match.group(4) or ''
        via_edges = via_attr.split() if via_attr else []
        
        # Check if this flow is within a comment block
        # Look backwards from match position to find if we're in a comment
        text_before = content[:match.start()]
        
        # Count comment starts and ends before this position
        comment_starts = text_before.count('<!--')
        comment_ends = text_before.count('-->')
        is_commented = comment_starts > comment_ends
        
        flows.append({
            'id': flow_id,
            'from': from_edge,
            'to': to_edge,
            'via': via_edges,
            'via_str': via_attr,
            'is_commented': is_commented
        })
    
    return flows

def main():
    net_file = 'd:/Codes/Projects/Traffic-Light-Automation/s1/k1.net.xml'
    route_file = 'd:/Codes/Projects/Traffic-Light-Automation/s1/k1_routes_3h_varying.rou.xml'
    
    print("="*80)
    print("K1 NETWORK ROUTE VALIDATION TOOL")
    print("="*80)
    print(f"\nNetwork file: {net_file}")
    print(f"Route file: {route_file}")
    
    # Parse network
    print("\n[1/3] Parsing network topology...")
    edges, outgoing = parse_network(net_file)
    print(f"  ✓ Found {len(edges)} edges")
    print(f"  ✓ Found {len(outgoing)} junctions")
    
    # Parse routes
    print("\n[2/3] Parsing route definitions...")
    flows = parse_routes(route_file)
    active_flows = [f for f in flows if not f['is_commented']]
    commented_flows = [f for f in flows if f['is_commented']]
    print(f"  ✓ Found {len(active_flows)} active flows")
    print(f"  ✓ Found {len(commented_flows)} commented flows")
    
    # Validate each route
    print("\n[3/3] Validating routes...")
    print("\n" + "="*80)
    
    valid_routes = []
    invalid_routes = []
    valid_routes_commented = []
    invalid_routes_commented = []
    route_groups = defaultdict(list)
    
    # Group flows by base route (strip time suffix)
    for flow in flows:
        base_name = '_'.join(flow['id'].split('_')[:-1])  # Remove _light, _peak, etc.
        route_groups[base_name].append(flow)
    
    # Check each unique route
    for base_name in sorted(route_groups.keys()):
        group = route_groups[base_name]
        first_flow = group[0]
        
        from_edge = first_flow['from']
        to_edge = first_flow['to']
        via_edges = first_flow['via']
        is_commented = first_flow['is_commented']
        
        is_valid, error, suggested = validate_route_with_via(
            from_edge, to_edge, via_edges, edges, outgoing
        )
        
        route_info = {
            'base_name': base_name,
            'from': from_edge,
            'to': to_edge,
            'via': via_edges,
            'via_str': first_flow['via_str'],
            'count': len(group),
            'is_valid': is_valid,
            'error': error,
            'suggested': suggested,
            'is_commented': is_commented
        }
        
        if is_commented:
            if is_valid:
                valid_routes_commented.append(route_info)
            else:
                invalid_routes_commented.append(route_info)
        else:
            if is_valid:
                valid_routes.append(route_info)
            else:
                invalid_routes.append(route_info)
    
    # Print results
    print("\n✅ VALID ROUTES (ACTIVE):")
    print("-"*80)
    if valid_routes:
        for r in valid_routes:
            via_display = f"via=\"{r['via_str']}\"" if r['via_str'] else "(no via)"
            print(f"\n  {r['base_name']} ({r['count']} flows)")
            print(f"    from=\"{r['from']}\" to=\"{r['to']}\" {via_display}")
            if r['suggested'] and r['suggested'] != '(direct)':
                print(f"    Path: {r['suggested']}")
    else:
        print("  (none)")
    
    print("\n\n❌ INVALID ROUTES (ACTIVE):")
    print("-"*80)
    if invalid_routes:
        for r in invalid_routes:
            via_display = f"via=\"{r['via_str']}\"" if r['via_str'] else "(no via)"
            print(f"\n  {r['base_name']} ({r['count']} flows) - {r['error']}")
            print(f"    from=\"{r['from']}\" to=\"{r['to']}\" {via_display}")
            if r['suggested']:
                print(f"    💡 Suggested fix: via=\"{r['suggested']}\"")
            else:
                print(f"    ⚠️  No path found - edge may not be reachable")
    else:
        print("  (none)")
    
    print("\n\n" + "="*80)
    print("COMMENTED OUT ROUTES:")
    print("="*80)
    
    print("\n✅ VALID (can be re-enabled):")
    print("-"*80)
    if valid_routes_commented:
        for r in valid_routes_commented:
            via_display = f"via=\"{r['via_str']}\"" if r['via_str'] else "(no via)"
            print(f"\n  {r['base_name']} ({r['count']} flows)")
            print(f"    from=\"{r['from']}\" to=\"{r['to']}\" {via_display}")
            if r['suggested'] and r['suggested'] != '(direct)':
                print(f"    Path: {r['suggested']}")
            print(f"    💡 This route is VALID - can uncomment and use!")
    else:
        print("  (none)")
    
    print("\n\n❌ INVALID (need fixes):")
    print("-"*80)
    if invalid_routes_commented:
        for r in invalid_routes_commented:
            via_display = f"via=\"{r['via_str']}\"" if r['via_str'] else "(no via)"
            print(f"\n  {r['base_name']} ({r['count']} flows) - {r['error']}")
            print(f"    from=\"{r['from']}\" to=\"{r['to']}\" {via_display}")
            if r['suggested']:
                print(f"    💡 Suggested fix: via=\"{r['suggested']}\"")
            else:
                print(f"    ⚠️  No path found - edge may not be reachable")
    else:
        print("  (none)")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    total_routes = len(valid_routes) + len(invalid_routes) + len(valid_routes_commented) + len(invalid_routes_commented)
    total_active = len(valid_routes) + len(invalid_routes)
    total_commented = len(valid_routes_commented) + len(invalid_routes_commented)
    
    print(f"\n  ACTIVE ROUTES:")
    print(f"    Total: {total_active}")
    print(f"    ✅ Valid: {len(valid_routes)}")
    print(f"    ❌ Invalid: {len(invalid_routes)}")
    if total_active > 0:
        print(f"    Success rate: {len(valid_routes)/total_active*100:.1f}%")
    
    print(f"\n  COMMENTED ROUTES:")
    print(f"    Total: {total_commented}")
    print(f"    ✅ Valid (can re-enable): {len(valid_routes_commented)}")
    print(f"    ❌ Invalid (need fixes): {len(invalid_routes_commented)}")
    
    print(f"\n  OVERALL:")
    print(f"    Total routes checked: {total_routes}")
    print(f"    ✅ Valid routes: {len(valid_routes) + len(valid_routes_commented)}")
    print(f"    ❌ Invalid routes: {len(invalid_routes) + len(invalid_routes_commented)}")
    
    # Generate fix commands
    if invalid_routes or invalid_routes_commented:
        print("\n" + "="*80)
        print("SUGGESTED FIXES")
        print("="*80)
        
        all_invalid = invalid_routes + invalid_routes_commented
        
        print("\nCopy these corrected route definitions:\n")
        for r in all_invalid:
            status = "(COMMENTED)" if r['is_commented'] else "(ACTIVE)"
            if r['suggested']:
                print(f"<!-- {r['base_name']}: {r['from']} to {r['to']} - CORRECTED {status} -->")
                print(f'<flow id="{r["base_name"]}_XXX" ... from="{r["from"]}" to="{r["to"]}" via="{r["suggested"]}" .../>')
            else:
                print(f"<!-- {r['base_name']}: {r['from']} to {r['to']} - NO PATH {status} - DISABLE -->")
            print()
    
    print("\n" + "="*80)
    return 0 if not (invalid_routes or invalid_routes_commented) else 1

if __name__ == '__main__':
    sys.exit(main())
