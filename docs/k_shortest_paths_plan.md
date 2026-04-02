# K-Shortest Paths Implementation Plan

**Date:** 2026-03-31  
**File to change:** `pymarl/src/envs/sumo_grid_reroute.py`  
**Scope:** Replace the stub in `_compute_k_shortest_paths()` and add one helper method `_dijkstra()`

---

## 1. Background and Motivation

Each of the 32 agents selects from `n_actions=4` candidate routes per decision period. These candidates are pre-computed by `_compute_k_shortest_paths(from_edge_id, to_edge_id, k=4)` and stored in `self.route_candidates[agent_id]`.

The current implementation computes one true shortest path and then fills the remaining 3 slots by appending the same first route again. The practical consequence is that all 4 actions are identical — the agent is making a binary decision (take route 1 / take route 1 again) with no real route diversity to learn from.

This is a correctness issue for the 500k smoke test because that run is being used to decide whether the 2M-step training budget can be reduced. If the policy converges faster than expected, the 2M-step budget may be cut. But convergence speed is directly tied to the complexity of the action space. With 4 identical actions, the learning problem is trivially easier than intended. A convergence signal from a broken action space is not a valid basis for a budget decision.

---

## 2. The Problem in the Current Code

**File:** `pymarl/src/envs/sumo_grid_reroute.py`, lines 928–986

```python
def _compute_k_shortest_paths(self, from_edge_id, to_edge_id, k):
    ...
    result = self.net.getShortestPath(from_edge, to_edge)
    if result and result[0]:
        first_route = [edge.getID() for edge in result[0]]
        routes.append(first_route)

        if k > 1 and len(first_route) > 2:
            for i in range(1, min(k, 4)):
                # TODO: Implement proper k-shortest paths algorithm
                if i < len(routes):
                    routes.append(first_route)   # <-- duplicate appended k-1 times
```

The `if i < len(routes)` condition is always false on the first iteration (routes has 1 entry, i=1, so 1 < 1 is false). On iteration 2, routes still has 1 entry, so 1 < 1 is still false. The loop body never executes. `routes` ends with exactly 1 entry, which is then padded with `None` for the remaining 3 slots, yielding a mask of `[1, 0, 0, 0]`.

In effect, every agent always has exactly 1 valid action.

---

## 3. Algorithm Choice: Yen's K-Shortest Simple Paths

Yen's algorithm (1971) is the standard for finding k shortest loopless paths in a directed graph. It is appropriate here because:

- It guarantees the k paths are returned in non-decreasing cost order.
- It produces truly distinct paths (no loops), which is required for meaningful route diversity in a road network.
- It requires only repeated calls to a shortest-path primitive — no changes to the underlying graph structure need to be permanent.

### Alternatives considered

| Algorithm | Reason rejected |
|---|---|
| Eppstein's algorithm | More complex to implement; overkill for k=4 in a small grid network |
| Random edge dropout heuristic | Non-deterministic; cannot guarantee diversity or validity of alternatives |
| `sumolib.net.getShortestPath()` with no modification | Does not support edge exclusion — cannot produce alternatives |
| Penalising edge weights in sumolib internals | Would mutate the shared `net` object; unsafe across concurrent calls and hard to undo reliably |

---

## 4. sumolib Graph API

`sumolib` does not expose Yen's algorithm or edge exclusion directly. However, it exposes enough of the underlying graph to implement a custom Dijkstra, which is what Yen's algorithm requires as its inner loop.

| API | What it returns |
|---|---|
| `self.net.getEdge(edge_id)` | `sumolib.net.edge.Edge` object |
| `edge.getOutgoing()` | List of `sumolib.net.connection.Connection` objects leaving this edge |
| `conn.getTo()` | The destination `Edge` of this connection |
| `edge.getLength()` | Length in metres (float) |
| `edge.getSpeed()` | Speed limit in m/s (float) |
| `edge.getID()` | String edge ID |

Travel time cost for an edge: `edge.getLength() / edge.getSpeed()`.  
This is the standard SUMO routing cost and matches what `net.getShortestPath()` uses internally.

Internal junction edges (IDs starting with `':'`) are passable but should not be included in the returned route list, as TraCI's `vehicle.setRoute()` expects only non-internal edge IDs.

---

## 5. Implementation Plan

### 5.1 New method: `_dijkstra(from_edge, to_edge, forbidden_edges)`

**Location:** Add after `_compute_k_shortest_paths()`, around line 987.

**Signature:**
```python
def _dijkstra(
    self,
    from_edge: "sumolib.net.edge.Edge",
    to_edge: "sumolib.net.edge.Edge",
    forbidden_edges: set = None,
) -> List["sumolib.net.edge.Edge"] | None:
```

**Purpose:** Standard Dijkstra on the sumolib edge graph, returning the sequence of `Edge` objects from `from_edge` to `to_edge` (inclusive), or `None` if no path exists. The `forbidden_edges` parameter accepts a set of `Edge` objects that will be skipped during traversal — this is how Yen's algorithm generates alternatives.

**Algorithm (pseudocode):**
```
forbidden = forbidden_edges or empty set
heap = [(cost=0, path=[from_edge])]
visited = set()

while heap is not empty:
    cost, path = heappop(heap)
    current = path[-1]

    if current in visited:
        continue
    visited.add(current)

    if current == to_edge:
        return path

    for conn in current.getOutgoing():
        next_edge = conn.getTo()
        if next_edge in visited:
            continue
        if next_edge in forbidden:
            continue
        step_cost = next_edge.getLength() / max(next_edge.getSpeed(), 0.1)
        heappush(heap, (cost + step_cost, path + [next_edge]))

return None  # no path found
```

**Notes:**
- Use `heapq` (stdlib, no new dependencies).
- Path is stored as a list of `Edge` objects inside the heap. For k=4 in a small 4×4 grid, path length is short (typically 3–8 edges) and memory pressure is negligible.
- If `from_edge == to_edge`, return `[from_edge]` immediately — the vehicle is already at its destination.
- Cap `getSpeed()` at a minimum of 0.1 to avoid division by zero on zero-speed edges.

---

### 5.2 Refactored method: `_compute_k_shortest_paths()`

**Location:** Lines 928–986 (replace entirely).

**Yen's algorithm — pseudocode:**

```
A = []        # accepted k-shortest paths (list of Edge lists)
B = []        # candidate heap

# Step 1: find the first shortest path
p0 = _dijkstra(from_edge, to_edge, forbidden_edges=set())
if p0 is None:
    return [None]*k, [0]*k
A.append(p0)

# Step 2: iterate to find paths 2..k
for i in range(1, k):
    prev_path = A[i-1]

    for spur_idx in range(len(prev_path) - 1):
        spur_node = prev_path[spur_idx]          # edge where the spur starts
        root_path = prev_path[:spur_idx + 1]     # path from source to spur_node

        forbidden = set()

        # Remove edges used by already-accepted paths that share this root
        for accepted in A:
            if accepted[:spur_idx + 1] == root_path:
                # The edge immediately after spur_node in this accepted path is forbidden
                if spur_idx + 1 < len(accepted):
                    forbidden.add(accepted[spur_idx + 1])

        # Remove all edges in the root path (except spur_node itself)
        # to prevent loops back through already-traversed nodes
        for edge in root_path[:-1]:
            forbidden.add(edge)

        spur_path = _dijkstra(spur_node, to_edge, forbidden_edges=forbidden)

        if spur_path is not None:
            candidate = root_path[:-1] + spur_path   # root (up to spur) + spur to dest
            if candidate not in B:
                heappush(B, (path_cost(candidate), candidate))

    if not B:
        break   # fewer than k paths exist in this graph

    # Accept the lowest-cost candidate
    _, best = heappop(B)
    A.append(best)

# Convert Edge lists to edge ID lists, filter internal edges
routes = []
for path in A:
    edge_ids = [e.getID() for e in path if not e.getID().startswith(':')]
    routes.append(edge_ids)
```

**After computing routes:**

1. Pass `routes` through the existing `_deduplicate_routes(routes, threshold=0.85)`. Yen's algorithm guarantees structural distinctness, but the deduplication step adds a safety net against near-identical paths that are technically different in the graph but share >85% of their edges — meaningless from a routing perspective.
2. Pad with `None` to length k.
3. Build the mask as before: `[1 if r is not None else 0 for r in routes]`.
4. If `action_noop_as_keep_route`, force `mask[0] = 1`.

---

### 5.3 Helper: `path_cost(path)`

Small inline helper (can be a local function inside `_compute_k_shortest_paths`):

```python
def path_cost(path):
    return sum(e.getLength() / max(e.getSpeed(), 0.1) for e in path)
```

Used for ordering candidates in the heap `B`.

---

## 6. Edge Cases

| Case | Handling |
|---|---|
| `from_edge == to_edge` | `_dijkstra` returns `[from_edge]` immediately. Single-edge route is valid. |
| Fewer than k paths exist | Yen's outer loop breaks early when `B` is empty. Routes padded with `None`, mask set to 0 for those slots. |
| `from_edge` or `to_edge` not in network | Caught by `net.getEdge()` exception at line 938–941, returns `[None]*k, [0]*k`. No change needed here. |
| Graph with no path (disconnected subgraph) | `_dijkstra` returns `None`. Yen's step 1 triggers the early return. |
| Very short routes (1 or 2 edges) | Spur loop has no interior spur nodes; Yen's produces only 1 path. Remaining slots padded. This is correct — there is genuinely only one path on a nearly-direct connection. |
| Internal junction edges in path | Filtered out when converting `Edge` objects to ID strings: `if not e.getID().startswith(':')`. |
| `forbidden_edges` blocks all paths | `_dijkstra` returns `None` for that spur; that spur contributes no candidate. Yen's continues to next spur. |

---

## 7. Performance Considerations

For a 4×4 grid with 32 agents:
- Network size: ~96 directed edges (4×4 grid = 16 blocks × 4 directions, minus boundary edges)
- Episode decision periods: 1000 steps / 10 second decision period = 100 calls to `_generate_route_candidates()` per episode
- Per call: 32 agents × `_compute_k_shortest_paths()` = 32 Yen's invocations per decision period
- Yen's inner work: k=4 paths, spur iterations bounded by path length (~5–10 edges), each spur runs `_dijkstra` on a 96-edge graph

Dijkstra on 96 edges is sub-millisecond. Total overhead per decision period: ~32 × 4 × 10 Dijkstra calls = ~1280 calls, each negligible. This is dominated by TraCI communication cost and will not add measurable wall time.

---

## 8. File Change Summary

| Method | Change type | Lines (approx) |
|---|---|---|
| `_compute_k_shortest_paths()` | Replace body entirely | 928–986 |
| `_dijkstra()` | New method, add after above | ~987–1020 |

No other files are changed. No new imports are needed — `heapq` is stdlib. `sumolib` is already imported.

---

## 9. Verification Steps

After implementation, run a single short episode in verbose mode and confirm:

1. `route_candidates[agent_id]` contains 4 distinct lists of edge IDs (or fewer if the graph genuinely has fewer paths) — not 3 `None` entries.
2. `route_masks[agent_id]` has more than one `1` entry for agents with multiple reachable routes.
3. No exception is raised for edge cases: agent on internal edge, agent at destination, inactive agent.
4. Wall time per episode does not increase measurably (confirm with timing logs).

---

## 10. Relation to 500k Smoke Test

This fix must land **before** the 500k smoke test. The smoke test's purpose is to determine whether reward curves flatten early enough to justify cutting the 2M-step training budget. That signal is only valid if the action space reflects the intended 4-route diversity. Running the smoke test with a broken action space (effectively 1 valid action) would yield faster apparent convergence that does not generalise to the real training setup.

Once this fix is in and the 500k smoke test is run:
- If reward curves flatten before ~400k steps → investigate cutting t_max to 1M
- If curves are still rising at 500k → proceed with 2M-step budget as planned
