#!/usr/bin/env python3
"""
Generate a gridlock stress route file by scaling flows from the weekday route file.

Usage:
    python scripts/generate_gridlock_from_weekday.py --weekday ../k1_routes_6h_weekday.rou.xml \
        --out ../k1_routes_6h_gridlock.rou.xml --mult 2.5

This script preserves `from`, `to`, `via` attributes and multiplies `vehsPerHour`.
If a flow has no `vehsPerHour` attribute, it is left unchanged.
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


def scale_flows(weekday_path: Path, out_path: Path, mult: float):
    tree = ET.parse(weekday_path)
    root = tree.getroot()

    # Create new root
    new_root = ET.Element(root.tag, root.attrib)

    # Copy VType elements first (preserve types)
    for child in root:
        if child.tag.endswith('vType'):
            new_root.append(child)

    # Process flow and trip elements
    for child in root:
        tag = child.tag
        if tag.endswith('flow') or tag.endswith('trip'):
            new_child = ET.Element(child.tag, child.attrib)
            # Adjust ID to include gridlock prefix if not already
            if 'id' in new_child.attrib and not new_child.attrib['id'].startswith('grid_'):
                new_child.attrib['id'] = 'grid_' + new_child.attrib['id']
            # Scale vehsPerHour if present
            vph = new_child.attrib.get('vehsPerHour')
            if vph:
                try:
                    new_vph = max(1, int(round(float(vph) * mult)))
                    new_child.attrib['vehsPerHour'] = str(new_vph)
                except Exception:
                    # leave unchanged if cannot parse
                    pass
            # Keep child text/children if any
            new_root.append(new_child)

    # Write to output with XML declaration and pretty newline formatting
    tree = ET.ElementTree(new_root)
    ET.indent(tree, space="    ")
    tree.write(out_path, encoding='utf-8', xml_declaration=True)
    print(f"Generated gridlock route file: {out_path} (multiplier {mult})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weekday', required=True, help='Path to weekday route file')
    parser.add_argument('--out', required=True, help='Output gridlock route file')
    parser.add_argument('--mult', type=float, default=2.5, help='Multiplier for vehsPerHour')
    args = parser.parse_args()

    weekday_path = Path(args.weekday).resolve()
    out_path = Path(args.out).resolve()

    if not weekday_path.exists():
        print(f"Weekday route file not found: {weekday_path}")
        raise SystemExit(1)

    scale_flows(weekday_path, out_path, args.mult)
