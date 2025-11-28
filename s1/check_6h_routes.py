#!/usr/bin/env python3
"""
Check 6h route file reachability against k1.net.xml
Usage: python check_6h_routes.py <route-file>
"""
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

NET_FILE = 'd:/Codes/Projects/Traffic-Light-Automation/s1/k1.net.xml'

def parse_network(net_file):
    tree = ET.parse(net_file)
    root = tree.getroot()
    edges = {}
    outgoing = defaultdict(list)
    for edge in root.findall('.//edge'):
        eid = edge.get('id')
        f = edge.get('from')
        t = edge.get('to')
        if eid and f and t:
            edges[eid] = (f,t)
    for eid,(f,t) in edges.items():
        outgoing[f].append(eid)
    return edges, outgoing

from collections import deque

def find_path(from_edge, to_edge, edges, outgoing, max_depth=50):
    if from_edge not in edges or to_edge not in edges:
        return False, []
    _, from_j = edges[from_edge]
    to_j, _ = edges[to_edge]
    q = deque()
    q.append((from_j, []))
    visited = {from_j}
    while q:
        cur_j, path = q.popleft()
        if cur_j == to_j:
            return True, path
        if len(path) > max_depth:
            continue
        for e in outgoing.get(cur_j, []):
            _, nxt = edges[e]
            if nxt not in visited:
                visited.add(nxt)
                q.append((nxt, path + [e]))
    return False, []

import re

def parse_flows_from_file(route_file):
    txt = open(route_file, 'r', encoding='utf-8').read()
    pattern = re.compile(r'<flow\s+id="([^"]+)"\s+[^>]*from="([^"]+)"\s+[^>]*to="([^"]+)"(?:\s+via="([^"]*)")?[^>]*/?>')
    flows = []
    for m in pattern.finditer(txt):
        fid = m.group(1)
        f = m.group(2)
        t = m.group(3)
        via = m.group(4) or ''
        via_edges = via.split() if via else []
        # Determine if commented
        start = m.start()
        before = txt[:start]
        is_commented = before.count('<!--') > before.count('-->')
        flows.append({'id':fid,'from':f,'to':t,'via':via_edges,'is_commented':is_commented})
    return flows


def main():
    if len(sys.argv) < 2:
        print('Usage: python check_6h_routes.py <route-file>')
        sys.exit(1)
    route_file = sys.argv[1]
    edges,outgoing = parse_network(NET_FILE)
    flows = parse_flows_from_file(route_file)
    print(f'Parsed {len(flows)} flows from {route_file}')
    bad = []
    for f in flows:
        if f['is_commented']:
            continue
        cur = f['from']
        ok = True
        for target in (f['via'] + [f['to']]):
            reachable, path = find_path(cur, target, edges, outgoing)
            if not reachable:
                bad.append((f['id'], cur, target))
                ok = False
                break
            cur = target
    if not bad:
        print('All active flows appear reachable (edge-level).')
    else:
        print('Flows with unreachable segments:')
        for fid,frm,tgt in bad:
            print(f'  {fid}: cannot reach {tgt} from {frm}')

if __name__=='__main__':
    main()
