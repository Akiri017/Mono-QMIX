# Action Space Topology Findings

**Date:** 2026-03-31  
**Context:** Debug run during 500k smoke test preparation, after implementing Yen's k-shortest paths algorithm.

---

## 1. Finding

After replacing the stub in `_compute_k_shortest_paths()` with a proper Yen's algorithm implementation, a debug print was added to log route candidates and action masks for the first 3 active agents at episode step 0. The output was:

```
Agent 0 | mask=[1, 1, 0, 0] | from=-E2 to=E2  | routes=[['-E2', 'E5'], ['-E2', '-E1'], None, None]
Agent 1 | mask=[1, 0, 0, 0] | from=E1  to=E9  | routes=[['E1',  'E5'], None,            None, None]
Agent 2 | mask=[1, 0, 0, 0] | from=E0  to=E1  | routes=[['E0',  'E1'], None,            None, None]
```

**Yen's algorithm is working correctly.** Agent 0 has a longer cross-grid trip and Yen's found 2 genuinely distinct paths. Agents 1 and 2 returned only 1 valid route — not because of an algorithm failure, but because their OD pairs are short and the 4×4 grid network does not contain additional topologically distinct paths for those origins and destinations.

Agent 2 (`E0 → E1`) is an adjacent-edge trip. There is exactly one edge connecting them. No alternative path exists regardless of algorithm.

---

## 2. Root Cause

The 4×4 SUMO grid is a small, sparse network. Most OD pairs — particularly short ones — have only 1 or 2 topologically distinct paths. Setting `n_actions=4` assumed 4 meaningful route alternatives exist per agent, which the network cannot provide for the majority of agents.

The consequence: most agents have a mask of `[1, 0, 0, 0]`, leaving them with only one valid action regardless of policy. This caused QMIX, noop, and greedy_shortest to produce exactly identical evaluation results — all three policies degenerate to the same behavior when only one action is valid.

This also explains why the random policy produced different results: it likely samples from all 4 action indices regardless of mask validity, and applying a `None` route has different side effects than action 0.

---

## 3. Why Lowering the Deduplication Threshold Is Not the Fix

An alternative approach — lowering `_deduplicate_routes()` threshold from 0.85 to ~0.5 — would allow routes with high edge overlap to survive as distinct actions. This was considered and rejected.

Routes sharing 50–80% of their edges are not meaningfully different from a routing perspective. An agent choosing between near-identical paths does not gain real routing diversity. The action space would appear richer on paper but the learning signal would be the same. This would misrepresent the environment's actual capabilities and is not defensible as a design choice.

---

## 4. Decision: Reduce `n_actions` to 2

**Planned change:** Set `n_actions: 2` in `pymarl/src/config/envs/sumo_grid4x4.yaml`.

**Rationale:**

- Reflects ground truth — the network genuinely supports 1–2 distinct paths per OD pair for most agents.
- A binary choice (keep current route / take alternative) is still a meaningful learning problem. QMIX must learn *when* to reroute across 32 coordinated agents under congestion — this is the thesis contribution, not the number of route options.
- Honest action masking: agents with 1 path get mask `[1, 0]`, agents with 2 paths get mask `[1, 1]`. No inflated slots.
- The thesis comparison (QMIX vs noop vs greedy vs random) remains valid. A binary rerouting decision is standard in traffic control literature.

**What is preserved:**
- Agents with 2 genuine paths (like Agent 0, `-E2 → E2`) still choose between them.
- Agents with 1 path (like Agents 1 and 2) correctly have only 1 valid action.
- The `action_noop_as_keep_route` logic still applies — action 0 is always "keep current route."

---

## 5. Files to Change

| File | Parameter | Current | Target |
|---|---|---|---|
| `pymarl/src/config/envs/sumo_grid4x4.yaml` | `n_actions` | 4 | 2 |

No changes to `sumo_grid_reroute.py` are needed. The Yen's implementation already handles `k=2` correctly — it will return up to 2 paths and mask accordingly.

---

## 6. Deferred Until After Documentation

This change has not been made yet. It will be applied before the 500k smoke test run.

After the change, re-run the 2000-step debug verification to confirm:
- Masks are `[1, 0]` or `[1, 1]` (no more `[1, 0, 0, 0]`)
- QMIX and noop no longer produce identical results (agents with 2 valid paths now have a real choice)
- The debug print can then be removed

---

## 7. Dijkstra Performance Issue and Deferred Fix

**Observed:** The current `_dijkstra()` implementation carries the full path as a list inside each heap entry (`path + [next_edge]` on every push). This is O(n²) in time and memory per call. On the 4×4 grid, with ~51,000 Dijkstra calls per episode (100 decision steps × 32 agents × ~16 Yen's spur calls), this caused a single training episode to take approximately **70 minutes**.

**Deferred fix — predecessor dict approach:**

Replace path-in-heap with a `dist` dict and `prev` dict. Reconstruct the path once at the end by backtracking through `prev`. Heap entries become `(cost, tie, edge)` instead of `(cost, tie, full_path_list)`, eliminating the O(n²) copying.

```python
dist = {from_edge: 0.0}
prev = {from_edge: None}
heap = [(0.0, 0, from_edge)]

while heap:
    cost, _, current = heapq.heappop(heap)
    if current == to_edge:
        path, node = [], to_edge
        while node is not None:
            path.append(node)
            node = prev[node]
        return path[::-1]
    if cost > dist.get(current, float('inf')):
        continue
    for next_edge in current.getOutgoing():
        if next_edge in forbidden_edges:
            continue
        new_cost = cost + next_edge.getLength() / max(next_edge.getSpeed(), 0.1)
        if new_cost < dist.get(next_edge, float('inf')):
            dist[next_edge] = new_cost
            prev[next_edge] = current
            heapq.heappush(heap, (new_cost, tie, next_edge))
```

**Tradeoff:** The predecessor dict tracks only one path per node — the cheapest. For Yen's algorithm with small k and a small grid, this is fine. On a larger graph (e.g., BGC OSM) with k≥4, suboptimal partial paths discarded by the predecessor dict could cause Yen's to miss valid candidates. At that scale, the correct solution is either:

1. **Cache routes at episode start** — compute once per OD pair at `reset()`, not every decision step. Eliminates ~99% of Dijkstra calls regardless of algorithm.
2. **Use `networkx.shortest_simple_paths()`** — battle-tested Yen's implementation. Build a NetworkX graph once from sumolib at load time and query it at runtime.
3. **Precompute offline** — run k-shortest paths for all OD pairs before training and load from disk.

**This is a known scalability gap.** It is documented here as a pre-BGC migration task, not a 4×4 grid blocker (since the 4×4 prototype will use caching or `n_actions=2` to limit call volume).
