"""
One-time MJCF prep: inject <site> elements for sole contact points into the K1 XML.

Usage:
    python scripts/add_sole_sites.py

Output:
    assets/booster_k1/K1_serial_with_sites.xml

This is required before using --ground_mode soft in retarget_no_penetration.py.
"""

import sys
import os
import pathlib
import xml.etree.ElementTree as ET

# Allow running from repo root
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from general_motion_retargeting.params import ROBOT_XML_DICT
from general_motion_retargeting.sole_points import get_sole_points

ROBOT_TYPE = "booster_k1"
CORNER_NAMES = ["fl", "fr", "rl", "rr"]


def _infer_side(body_name: str) -> str:
    if "left" in body_name.lower():
        return "left"
    if "right" in body_name.lower():
        return "right"
    return body_name


def _indent(elem, level=0):
    """Add pretty-print indentation to an ElementTree in-place."""
    indent = "\n" + "  " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = indent
        for child in elem:
            _indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indent
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = indent
    if not level:
        elem.tail = "\n"


def main():
    xml_path = str(ROBOT_XML_DICT[ROBOT_TYPE])
    output_path = xml_path.replace(".xml", "_with_sites.xml")

    sole_config = get_sole_points(ROBOT_TYPE)

    print(f"[add_sole_sites] Reading: {xml_path}")
    ET.register_namespace("", "")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Collect all <body> elements by name for fast lookup
    body_map = {}
    for body_elem in root.iter("body"):
        name = body_elem.get("name")
        if name:
            body_map[name] = body_elem

    added = []
    for body_name, points in sole_config.items():
        if body_name not in body_map:
            print(f"[warn] Body '{body_name}' not found in XML, skipping")
            continue

        body_elem = body_map[body_name]
        side = _infer_side(body_name)

        for i, (px, py, pz) in enumerate(points):
            corner = CORNER_NAMES[i] if i < len(CORNER_NAMES) else str(i)
            site_name = f"{side}_sole_{corner}"

            # Check if site already exists
            existing = any(
                s.get("name") == site_name for s in body_elem.findall("site")
            )
            if existing:
                print(f"  [skip] Site '{site_name}' already exists")
                continue

            site_elem = ET.SubElement(body_elem, "site")
            site_elem.set("name", site_name)
            site_elem.set("pos", f"{px:.4f} {py:.4f} {pz:.4f}")
            site_elem.set("size", "0.005")
            site_elem.set("rgba", "1 0 0 0.5")
            added.append(site_name)
            print(f"  [add] {site_name} @ body '{body_name}' pos=({px:.4f}, {py:.4f}, {pz:.4f})")

    _indent(root)
    tree.write(output_path, encoding="unicode", xml_declaration=True)

    print(f"\n[add_sole_sites] Added {len(added)} sites.")
    print(f"[add_sole_sites] Output: {output_path}")


if __name__ == "__main__":
    main()
