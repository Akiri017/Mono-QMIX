"""
Assign diverse vehicle types to background traffic trips.

Distribution (Philippine urban traffic approximation):
  thesis_car   55%
  thesis_moto  25%
  thesis_bus   10%
  thesis_truck 10%

Runs in-place on each trips file. Seed=42 for reproducibility.
"""

import random
import re
from pathlib import Path

TYPES = ["thesis_car", "thesis_moto", "thesis_truck", "thesis_bicycle"]
WEIGHTS = [0.62, 0.25, 0.10, 0.03]

FILES = [
    "4by4_map/trips_low.xml",
    "4by4_map/trips_high.xml",
    "bgc_core/trips_low.xml",
    "bgc_core/trips_med.xml",
    "bgc_core/trips_high.xml",
    "bgc_full/trips_low.xml",
    "bgc_full/trips_med.xml",
    "bgc_full/trips_highEnough.xml",
]

rng = random.Random(42)

for path_str in FILES:
    path = Path(path_str)
    if not path.exists():
        print(f"SKIP (not found): {path}")
        continue

    text = path.read_text(encoding="utf-8")
    assigned = 0

    def replace_type(m):
        global assigned
        tag = m.group(0)
        vtype = rng.choices(TYPES, WEIGHTS)[0]
        assigned += 1
        # Remove existing type= attribute if present (handles both quote styles)
        tag = re.sub(r"""\s+type=["'][^"']*["']""", "", tag)
        tag = tag.replace("<trip ", f'<trip type="{vtype}" ', 1)
        return tag

    new_text = re.sub(r"<trip\b[^/]*/?>", replace_type, text)
    path.write_text(new_text, encoding="utf-8")
    print(f"{path}: assigned types to {assigned} trips")

print("\nDone. Distribution used:", dict(zip(TYPES, WEIGHTS)))
