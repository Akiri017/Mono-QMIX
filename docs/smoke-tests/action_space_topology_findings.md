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

## 6. ~~Deferred Until After Documentation~~ — DONE (2026-04-02)

- `n_actions: 2` applied to `pymarl/src/config/envs/sumo_grid4x4.yaml` (both `env_args` and top-level PyMARL block).
- 2000-step debug verification run confirmed masks are `[1, 0]` or `[1, 1]` — no more `[1, 0, 0, 0]`.
- Debug print removed from `_generate_route_candidates()`.

---

## 7. ~~Dijkstra Performance Issue and Deferred Fix~~ — DONE (2026-04-02)

**Original issue:** `_dijkstra()` carried the full path list in every heap entry (`path + [next_edge]` on every push) — O(n²) per call. With ~51,000 calls per episode this caused ~70 min/episode.

**Fixes applied to `pymarl/src/envs/sumo_grid_reroute.py`:**

**Fix 1 — Predecessor dict Dijkstra:** Replaced path-in-heap with `dist` + `prev` dicts. Heap entries are now `(cost, tie, edge)`. Path reconstructed once at the end by backtracking through `prev`. O(n²) → O(n log n) per call.

**Fix 2 — OD-pair cache (`_yen_cache`):** Added `self._yen_cache: dict` keyed on `(from_edge_id, to_edge_id)`. On a cache hit, Yen's algorithm is skipped entirely. Cache is cleared at `reset()`. Since cost metric is static (edge length), results are deterministic per OD pair and safe to reuse within an episode.

**Tradeoff (still applies for BGC migration):** The predecessor dict tracks only the cheapest path per node. On a larger graph (e.g., BGC OSM) with k≥4, suboptimal partial paths discarded by `prev` could cause Yen's to miss valid candidates. At that scale, the correct solution is either:

1. **Use `networkx.shortest_simple_paths()`** — battle-tested Yen's implementation.
2. **Precompute offline** — run k-shortest paths for all OD pairs before training and load from disk.

These remain pre-BGC migration tasks. For the 4×4 prototype, Fix 1 + Fix 2 are sufficient.
