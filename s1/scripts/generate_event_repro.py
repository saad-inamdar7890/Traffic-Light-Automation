"""
generate_event_repro.py
-----------------------
Create a reproducible 'event_repro' route file by duplicating existing flows
and adding a late-time surge (e.g. between 18000-18600 seconds) to increase
load around the time your training previously collapsed.

Usage:
  python generate_event_repro.py --input ./s1/k1_routes_6h_event.rou.xml \
      --out ./s1/k1_routes_6h_event_repro.rou.xml --mult 3 --begin 18000 --end 18600

This script preserves `via` attributes and all original flow properties, but
adds extra surge flows with increased `vehsPerHour` during the late window.
"""
import argparse
import xml.etree.ElementTree as ET
import copy


def generate_repro(input_path, out_path, mult=3.0, begin=18000, end=18600):
    tree = ET.parse(input_path)
    root = tree.getroot()

    # SUMO route files often have <routes> as root
    routes_elem = root
    if routes_elem.tag != 'routes':
        # try to find the routes element
        routes_elem = root.find('routes') or root

    flow_elems = [e for e in list(routes_elem) if e.tag == 'flow']

    added = 0
    for f in flow_elems:
        # Only duplicate flows that have a vehsPerHour attribute > 0
        vph = f.attrib.get('vehsPerHour') or f.attrib.get('vehs/hour')
        try:
            vph_val = float(vph) if vph is not None else 0.0
        except Exception:
            vph_val = 0.0

        if vph_val <= 0:
            continue

        new_f = copy.deepcopy(f)
        # Make a new id and set times and scaled vehsPerHour
        old_id = new_f.attrib.get('id', 'flow')
        new_f.attrib['id'] = f"{old_id}_surge"
        new_vph = max(1.0, vph_val * float(mult))
        new_f.attrib['vehsPerHour'] = str(int(new_vph))
        new_f.attrib['begin'] = str(begin)
        new_f.attrib['end'] = str(end)

        # Append the new flow
        routes_elem.append(new_f)
        added += 1

    tree.write(out_path, encoding='utf-8', xml_declaration=True)
    print(f"Generated repro route file: {out_path} (added {added} surge flows, mult={mult})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='s1/k1_routes_6h_event.rou.xml')
    parser.add_argument('--out', type=str, default='s1/k1_routes_6h_event_repro.rou.xml')
    parser.add_argument('--mult', type=float, default=3.0, help='Multiplier for vehsPerHour during surge')
    parser.add_argument('--begin', type=int, default=18000, help='Begin time for surge (seconds)')
    parser.add_argument('--end', type=int, default=18600, help='End time for surge (seconds)')
    args = parser.parse_args()

    generate_repro(args.input, args.out, mult=args.mult, begin=args.begin, end=args.end)
