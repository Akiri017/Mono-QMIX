import argparse
import random
import sys
from pathlib import Path

import xml.etree.ElementTree as ET


def _read_net_and_edges(net_file: Path):
    """Read SUMO net and return (net, candidate_edges).

    Candidate edges exclude internal/special edges and must allow passenger vehicles.
    """
    try:
        import sumolib  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "sumolib is required. Run: python -m pip install -r requirements.txt"
        ) from exc

    net = sumolib.net.readNet(str(net_file))
    edges = []
    for e in net.getEdges():
        if e.isSpecial():
            continue
        # Prefer passenger-compatible edges.
        try:
            if not e.allows("passenger"):
                continue
        except Exception:
            pass
        edges.append(e)

    if not edges:
        raise RuntimeError(f"No usable edges found in net: {net_file}")
    return net, edges


def _write_controlled_routes(
    routes_path: Path,
    trips_path: Path,
    *,
    net,
    edges,
    n: int,
    depart_window: float,
    seed: int,
    max_attempts_per_vehicle: int = 500,
) -> None:
    rng = random.Random(seed)

    routes_root = ET.Element(
        "routes",
        {
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/routes_file.xsd",
        },
    )

    trips_root = ET.Element(
        "routes",
        {
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/routes_file.xsd",
        },
    )

    records: list[dict] = []

    for i in range(n):
        depart = rng.random() * depart_window
        path_edges = None
        from_edge = None
        to_edge = None

        for _ in range(max_attempts_per_vehicle):
            from_edge = rng.choice(edges)
            to_edge = rng.choice(edges)
            if to_edge == from_edge:
                continue

            path_edges, _cost = net.getShortestPath(from_edge, to_edge)
            if path_edges is not None and len(path_edges) > 0:
                break
            path_edges = None

        if path_edges is None or from_edge is None or to_edge is None:
            raise RuntimeError(
                f"Could not find feasible OD pair for ctrl_{i} after {max_attempts_per_vehicle} attempts. "
                "Net may be too disconnected for the current sampling strategy."
            )

        records.append(
            {
                "id": f"ctrl_{i}",
                "depart": float(f"{depart:.2f}"),
                "from": from_edge.getID(),
                "to": to_edge.getID(),
                "edges": " ".join(e.getID() for e in path_edges),
            }
        )

    # SUMO prefers route files sorted by departure time.
    records.sort(key=lambda r: r["depart"])

    for r in records:
        ET.SubElement(
            trips_root,
            "trip",
            {
                "id": r["id"],
                "depart": f"{r['depart']:.2f}",
                "from": r["from"],
                "to": r["to"],
            },
        )

        veh = ET.SubElement(routes_root, "vehicle", {"id": r["id"], "depart": f"{r['depart']:.2f}"})
        ET.SubElement(veh, "route", {"edges": r["edges"]})

    routes_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(routes_root).write(routes_path, encoding="utf-8", xml_declaration=True)
    ET.ElementTree(trips_root).write(trips_path, encoding="utf-8", xml_declaration=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate fixed-N controlled fleet init routes for SUMO.")
    ap.add_argument("--net", type=Path, required=True, help="Path to .net.xml")
    ap.add_argument("--n", type=int, default=32, help="Number of controlled vehicles")
    ap.add_argument(
        "--depart-window",
        type=float,
        default=10.0,
        help="Depart time window in seconds (vehicles depart uniformly in [0, window])",
    )
    ap.add_argument("--seed", type=int, default=1, help="Random seed")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("sumo/scenarios/4by4_map"),
        help="Output directory for controlled_init.* files",
    )

    args = ap.parse_args()
    net_file: Path = args.net
    out_dir: Path = args.out_dir

    if not net_file.exists():
        print(f"ERROR: net file not found: {net_file}", file=sys.stderr)
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    trips_path = out_dir / "controlled_init.trips.xml"
    routes_path = out_dir / "controlled_init.rou.xml"

    net, edges = _read_net_and_edges(net_file)
    _write_controlled_routes(
        routes_path,
        trips_path,
        net=net,
        edges=edges,
        n=args.n,
        depart_window=args.depart_window,
        seed=args.seed,
    )

    print(f"Wrote: {routes_path}")
    print(f"Wrote: {trips_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
