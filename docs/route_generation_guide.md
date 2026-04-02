# K-Shortest Paths Route Generation Guide

This document provides technical details for implementing candidate route generation using k-shortest paths in SUMO.

## Overview

For the QMIX rerouting task, each agent needs to select from a discrete set of M candidate routes from its current position to its destination. These candidates are computed using **k-shortest paths** algorithms.

## Implementation Using sumolib

SUMO provides `sumolib` which includes utilities for computing k-shortest paths on a road network.

### Basic approach

```python
import sumolib

# Load the network
net = sumolib.net.readNet("path/to/network.net.xml")

# Get current edge and destination edge
current_edge = net.getEdge(current_edge_id)
destination_edge = net.getEdge(destination_edge_id)

# Compute k shortest paths
k = 4  # number of candidate routes
cost_metric = "length"  # or "traveltime"

routes = []
for route in sumolib.route.kShortestPaths(net, current_edge, destination_edge, k=k):
    routes.append([edge.getID() for edge in route])
```

### Cost metrics

**Static (length-based):**
- Uses edge length (meters) as cost
- Deterministic: same routes for same (current_edge, destination) pair
- Pros: Fast, stable, simple
- Cons: Ignores current traffic conditions

**Dynamic (travel-time-based):**
- Uses current/predicted travel time as cost
- Can be updated from TraCI: `traci.edge.getAdaptedTraveltime(edge_id)`
- Pros: Time-aware, adapts to congestion
- Cons: More complex, requires TraCI synchronization

**Recommendation:** Start with **static length-based** for initial implementation, optionally upgrade to dynamic later.

## Handling Edge Cases

### Fewer than k routes exist

When the network structure provides fewer than k distinct routes from current edge to destination:

```python
def get_candidate_routes(net, current_edge_id, dest_edge_id, k=4):
    """
    Generate k candidate routes, padding with no-op if fewer routes exist.

    Returns:
        routes: list of route (list of edge IDs)
        mask: binary mask indicating valid routes
    """
    current_edge = net.getEdge(current_edge_id)
    dest_edge = net.getEdge(dest_edge_id)

    routes = []
    for route in sumolib.route.kShortestPaths(net, current_edge, dest_edge, k=k):
        routes.append([edge.getID() for edge in route])

    # Pad with None if fewer than k routes
    num_valid = len(routes)
    while len(routes) < k:
        routes.append(None)  # or [] for no-op/keep current route

    # Create availability mask
    mask = [1 if route is not None else 0 for route in routes]

    return routes, mask
```

### No-op action (keep current route)

**Option A:** Reserve action 0 as explicit "keep current route"
- Actions 1 to M-1 are reroute candidates
- Action 0 means "do not change route"

**Option B:** Use no-op only when fewer than M routes exist
- If k routes computed, actions 0 to k-1 are valid
- Actions k to M-1 are masked invalid (and implicitly map to no-op)

**Recommendation:** Use **Option A** for clearer semantics and to give the agent explicit control over whether to reroute.

## Filtering Duplicate Routes

K-shortest paths may return duplicate or near-duplicate routes. Filter these:

```python
def deduplicate_routes(routes, similarity_threshold=0.9):
    """
    Remove routes that share more than similarity_threshold fraction of edges.
    """
    unique_routes = []
    for route in routes:
        is_duplicate = False
        for existing in unique_routes:
            overlap = len(set(route) & set(existing)) / max(len(route), len(existing))
            if overlap > similarity_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_routes.append(route)
    return unique_routes
```

## Efficiency Considerations

### Precomputation vs. online computation

**Precompute all routes (offline):**
- For small networks, precompute k-shortest paths for all (src, dst) pairs
- Store in a lookup table: `routes_cache[(src_edge, dst_edge)] = [route1, route2, ...]`
- Pros: Very fast at runtime
- Cons: Memory intensive, doesn't support dynamic travel times

**Compute on-demand (online):**
- Compute k-shortest paths at each decision timestep for each agent
- Pros: Supports dynamic travel times, lower memory
- Cons: Slower (but still fast enough for k=4 on a 4x4 grid)

**Recommendation:** Use **on-demand computation** initially. Profile if performance becomes an issue.

### Caching within an episode — **IMPLEMENTED (2026-04-02)**

If using static cost metric (edge length), cache routes within an episode:

```python
class RouteCandidateCache:
    def __init__(self, net, k=4):
        self.net = net
        self.k = k
        self.cache = {}

    def get_routes(self, current_edge_id, dest_edge_id):
        key = (current_edge_id, dest_edge_id)
        if key not in self.cache:
            routes, mask = compute_k_shortest_paths(self.net, current_edge_id, dest_edge_id, self.k)
            self.cache[key] = (routes, mask)
        return self.cache[key]

    def clear(self):
        """Call at episode reset"""
        self.cache.clear()
```

## Integration with TraCI

### Applying a reroute action

Once an agent selects an action (index into candidate routes):

```python
import traci

def apply_reroute_action(vehicle_id, selected_route):
    """
    Apply a route to a vehicle via TraCI.

    Args:
        vehicle_id: SUMO vehicle ID
        selected_route: list of edge IDs [edge_0, edge_1, ..., edge_n]
    """
    if selected_route is None or len(selected_route) == 0:
        # No-op: keep current route
        return

    # Set the vehicle's route
    traci.vehicle.setRoute(vehicle_id, selected_route)
```

### Getting current edge and destination

```python
def get_vehicle_routing_info(vehicle_id):
    """
    Get current edge and planned destination of a vehicle.

    Returns:
        current_edge_id: str
        destination_edge_id: str
    """
    current_edge_id = traci.vehicle.getRoadID(vehicle_id)
    route = traci.vehicle.getRoute(vehicle_id)
    destination_edge_id = route[-1] if len(route) > 0 else current_edge_id

    return current_edge_id, destination_edge_id
```

## Example: Complete route generation function

```python
import sumolib
import traci

def generate_candidate_routes(net, vehicle_id, k=4, cost_metric="length"):
    """
    Generate k candidate routes for a vehicle.

    Args:
        net: sumolib network object
        vehicle_id: SUMO vehicle ID
        k: number of candidate routes (M)
        cost_metric: "length" or "traveltime"

    Returns:
        routes: list of k routes (each route is a list of edge IDs)
        mask: binary mask of shape (k,) indicating valid routes
    """
    # Get current state
    current_edge_id = traci.vehicle.getRoadID(vehicle_id)
    current_route = traci.vehicle.getRoute(vehicle_id)

    # Handle special cases
    if current_edge_id.startswith(':'):
        # Vehicle is on internal edge (junction), skip rerouting
        return [None] * k, [0] * k

    if len(current_route) == 0:
        # No destination, cannot reroute
        return [None] * k, [0] * k

    destination_edge_id = current_route[-1]

    # Get edges
    try:
        current_edge = net.getEdge(current_edge_id)
        dest_edge = net.getEdge(destination_edge_id)
    except:
        # Edge not found
        return [None] * k, [0] * k

    # Compute k-shortest paths
    routes = []
    for route in sumolib.route.kShortestPaths(net, current_edge, dest_edge, k=k):
        routes.append([edge.getID() for edge in route])

    # Deduplicate if needed
    routes = deduplicate_routes(routes)

    # Pad with None (no-op)
    num_valid = len(routes)
    while len(routes) < k:
        routes.append(None)

    # Create mask (1 = valid, 0 = invalid/no-op)
    mask = [1 if route is not None else 0 for route in routes]

    return routes, mask

def deduplicate_routes(routes, threshold=0.85):
    """Remove near-duplicate routes."""
    if len(routes) <= 1:
        return routes

    unique = [routes[0]]
    for route in routes[1:]:
        is_dup = False
        for existing in unique:
            overlap = len(set(route) & set(existing)) / max(len(route), len(existing))
            if overlap > threshold:
                is_dup = True
                break
        if not is_dup:
            unique.append(route)
    return unique
```

## Verification

Test route generation:

```python
# Load network
net = sumolib.net.readNet("path/to/network.net.xml")

# Start SUMO with TraCI
traci.start(["sumo", "-c", "path/to/config.sumocfg"])
traci.simulationStep()

# Get a vehicle
vehicle_id = traci.vehicle.getIDList()[0]

# Generate routes
routes, mask = generate_candidate_routes(net, vehicle_id, k=4)

print(f"Vehicle {vehicle_id}:")
print(f"  Current edge: {traci.vehicle.getRoadID(vehicle_id)}")
print(f"  Destination: {traci.vehicle.getRoute(vehicle_id)[-1]}")
print(f"  Candidate routes ({sum(mask)} valid):")
for i, (route, valid) in enumerate(zip(routes, mask)):
    if valid:
        print(f"    Action {i}: {' -> '.join(route[:3])}... ({len(route)} edges)")
    else:
        print(f"    Action {i}: [INVALID/NO-OP]")

traci.close()
```

Expected output:
```
Vehicle controlled_0:
  Current edge: A0B0
  Destination: D3C3
  Candidate routes (4 valid):
    Action 0: A0B0 -> B0C0 -> C0D0... (12 edges)
    Action 1: A0B0 -> A0A1 -> A1B1... (13 edges)
    Action 2: A0B0 -> B0B1 -> B1C1... (14 edges)
    Action 3: A0B0 -> B0C0 -> C0C1... (15 edges)
```

## Summary

- Use `sumolib.route.kShortestPaths` for computing candidate routes
- Start with static edge length cost, optionally upgrade to dynamic travel time
- Handle edge cases: fewer than k routes, no-op action, internal edges
- Filter duplicate routes to ensure diversity
- Apply reroutes via `traci.vehicle.setRoute()`
- Cache routes within episodes if using static cost metric

For implementation in Step 4, integrate this logic into the `SUMOGridRerouteEnv` environment class.
