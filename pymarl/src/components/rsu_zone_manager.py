"""
RSU Zone Manager — Phase 0, Civiq Scaffolding

Assigns vehicles to their nearest Road-Side Unit (RSU) zone using
Euclidean distance. Designed to be instantiated from a YAML config
produced by scripts/extract_rsu_coords.py.

Usage:
    import yaml
    from components.rsu_zone_manager import RSUZoneManager

    with open("config/envs/rsu/synthetic_4x4.yaml") as f:
        cfg = yaml.safe_load(f)
    zm = RSUZoneManager(cfg)
    zones = zm.assign_vehicles_to_zones({"v0": (100.0, 200.0), ...})
"""

import numpy as np
from typing import Dict, List


class RSUZoneManager:
    """
    Nearest-RSU vehicle partitioner.

    Parameters
    ----------
    rsu_config : dict
        Loaded from config/envs/rsu/<map>.yaml.  Expected keys:
          rsu_positions      — list of dicts with keys 'id', 'x', 'y'
          max_rsus           — int, total padded RSU slots
          max_agents_per_rsu — int, cap used for assertions / masking info
    """

    def __init__(self, rsu_config: dict):
        positions = rsu_config["rsu_positions"]
        self.n_rsus: int = len(positions)
        self.max_rsus: int = int(rsu_config["max_rsus"])
        self.max_agents_per_rsu: int = int(rsu_config["max_agents_per_rsu"])

        # Ordered list of RSU ids (index == rsu_id used throughout)
        self.rsu_ids: List[str] = [p["id"] for p in positions]

        # (n_rsus, 2) float array for fast vectorised distance computation
        self._rsu_xy = np.array(
            [[float(p["x"]), float(p["y"])] for p in positions],
            dtype=np.float64,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assign_vehicles_to_zones(
        self, vehicle_positions: Dict[str, tuple]
    ) -> Dict[int, List[str]]:
        """
        Assign every vehicle to its nearest RSU.

        Parameters
        ----------
        vehicle_positions : dict
            {vehicle_id: (x, y)}  — may be empty.

        Returns
        -------
        dict
            {rsu_index (int): [vehicle_id, ...]}
            All n_rsus indices are present as keys (even if list is empty).
            Every vehicle appears in exactly one list.
        """
        # Initialise all zone lists (guarantees every key present)
        zones: Dict[int, List[str]] = {i: [] for i in range(self.n_rsus)}

        if not vehicle_positions:
            return zones

        veh_ids = list(vehicle_positions.keys())
        # (n_vehicles, 2)
        veh_xy = np.array(
            [vehicle_positions[v] for v in veh_ids], dtype=np.float64
        )

        # (n_vehicles, n_rsus) squared-distance matrix — avoids sqrt
        diff = veh_xy[:, np.newaxis, :] - self._rsu_xy[np.newaxis, :, :]
        sq_dist = (diff ** 2).sum(axis=2)

        # argmin gives nearest RSU index; ties broken by lower index (numpy default)
        nearest = np.argmin(sq_dist, axis=1)

        for vid, rsu_idx in zip(veh_ids, nearest.tolist()):
            zones[rsu_idx].append(vid)

        return zones

    def get_rsu_mask(self) -> np.ndarray:
        """
        Return active-RSU mask of shape (max_rsus,).

        Active slots (0 .. n_rsus-1) → 1.0
        Padded slots (n_rsus .. max_rsus-1) → 0.0
        """
        mask = np.zeros(self.max_rsus, dtype=np.float32)
        mask[: self.n_rsus] = 1.0
        return mask
