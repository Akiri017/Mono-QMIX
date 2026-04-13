# Hotfix: Invalid HBEFA3 emission classes in vtypes.add.xml files

**Date:** 2026-04-13
**Status:** Complete
**Affected areas:** `bgc_full/vtypes.add.xml`, `bgc_core/vtypes.add.xml`, `4by4_map/vtypes.add.xml`

## Context
Commit `efb05ac` ("added bicycles") caused a full pipeline failure — SUMO refused
to start on every episode, training and evaluation both exited with code 1. The
root cause was invalid HBEFA3 emission class names introduced in the earlier commit
`d2bb766` ("added diverse agent types"). `efb05ac` did not introduce the bad classes
itself, but reformatted the XML and added a new vtype, which caused SUMO 1.25.0 to
re-validate all vtypes on startup and surface the latent errors.

## Discussion and decisions

Investigation was done via git diff on `efb05ac` and a repo-wide grep for the
offending class names. Two vtypes had invalid emission classes:

**Motorcycle (`thesis_moto`):**
- `HBEFA3/MC_G_EU4` — original value from `d2bb766`. Failed: class does not exist
  in SUMO 1.25.0.
- `HBEFA3/MC_EU4` — attempted fix based on SUMO naming conventions (no fuel
  subcategory for motorcycles). Also failed: HBEFA3 motorcycle emission models are
  not implemented in SUMO 1.25.0 at all — no `MC_` entries exist in the install's
  data directory.
- `zero` — final decision. Honest and explicit: motorcycles are excluded from
  emissions tracking rather than silently using a wrong model. A passenger car proxy
  (`HBEFA3/PC_G_EU4`) was rejected as inaccurate for thesis use.

**Truck (`thesis_truck`):**
- `HBEFA3/RT_le20t_D_EU4` — original value from `d2bb766`. Failed: the `RT_` prefix
  is not a valid HBEFA3 namespace in SUMO 1.25.0.
- `HBEFA3/HDV_D_EU4` — final decision. Found by inspecting
  `C:\Program Files (x86)\Eclipse\Sumo\tools\contributed\sumopy\coremodules\demand\vehicles.py`,
  SUMO's own internal emission catalog. `HDV_D_EU4` (Heavy Duty Vehicle, Diesel,
  Euro 4) is the correct SUMO 1.25.0 name for what `RT_le20t_D_EU4` was intended
  to represent, and is used as the default in SUMO's own tooling.

## What changed

| vType | vClass | Before | After | Emissions tracked |
|---|---|---|---|---|
| `thesis_car` | `passenger` | `HBEFA3/PC_G_EU4` | `HBEFA3/PC_G_EU4` | Yes (unchanged) |
| `thesis_moto` | `motorcycle` | `HBEFA3/MC_G_EU4` | `zero` | No |
| `thesis_truck` | `truck` | `HBEFA3/RT_le20t_D_EU4` | `HBEFA3/HDV_D_EU4` | Yes |
| `thesis_bicycle` | `bicycle` | `zero` | `zero` | No (unchanged) |

Applied identically across all three map scenarios: `bgc_full`, `bgc_core`,
`4by4_map`.

## Implementation notes
- SUMO 1.25.0 does not ship HBEFA3 motorcycle emission models. Neither `MC_G_EU4`
  nor `MC_EU4` exist in the installation. Verified by searching the entire SUMO data
  directory for any `MC_` references — none found.
- The valid HBEFA3 class list was sourced from SUMO's internal Python tooling
  (`sumopy/coremodules/demand/vehicles.py`), which is more reliable than guessing
  from documentation.
- The `RT_` prefix used in the original file does not correspond to any HBEFA3
  category in this SUMO version. The correct heavy vehicle prefix is `HDV_`.

## Status and follow-up
- **Status:** Complete (pending smoke test confirmation on next run)
- **Follow-up items:**
  - Motorcycle emissions (`thesis_moto`) are permanently excluded from the
    `Total Emissions` metric due to SUMO 1.25.0 limitations. Add a footnote in the
    thesis methodology section noting that motorcycle emissions use the zero model
    and do not contribute to reported emissions figures.
  - Consider auditing any other scenario-specific route or additional files that may
    reference these same invalid class names (e.g. inline `vType` definitions inside
    `.rou.xml` files).
