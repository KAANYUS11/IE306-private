# Assignment 2 — Change Report

## Bugs (spec violations / logical errors)

**1. Coordination offset off by 4 s (§1.2)**
The original offset initialisation ran WE_GREEN for `offset` (20 s) then appended a 4 s amber, making B's NS_GREEN start at t = 24 s instead of t = 20 s. Fixed by running WE_GREEN for `offset − amber = 16 s` followed by the 4 s amber so NS_GREEN begins at exactly t = 20 s.

**2. Emergency preemption restarted the cycle from scratch instead of the interrupted point (§1.7)**
`_preemption_sequence` called `_run_cycle()` after the emergency, which always began with NS_GREEN and, for Intersection B, re-executed the offset initialisation block — adding a ~24 s drift on every preemption event. Fixed by adding `_run_normal_cycle()` (the repeating loop, no offset) and `_resume_from_interrupted()`, which serves out the remaining time of the interrupted phase (AMBER_NS / WE_GREEN / AMBER_WE) before handing off to the normal loop.

**3. Recovery time was always a constant 4 s (§3 KPI 6)**
`recovery_times.append(env.now − _emergency_end)` was called right after the 4 s post-emergency amber, so it always equalled 4 regardless of how long the cycle was out of phase. Moved the recording to the end of `_resume_from_interrupted`, where it now measures the full time from emergency-green end to normal-cycle resumption.

**4. `compute_time_average` dropped the final interval (§3 KPI 4)**
The loop ran `range(len(log) − 1)`, silently omitting the interval from the last log entry to `sim_time`. Added explicit accumulation of the final interval after the loop.

**5. Queue length counted the vehicle being served (§3 KPI 2)**
`ns_leave()` / `we_leave()` were called after service completion, so the "queue" counter always included the vehicle actively crossing the intersection. Moved both calls to fire when service begins (after server acquisition), so the count reflects only vehicles waiting.

**6. Single NS service RNG shared across both intersections (§2.1)**
One stream (`service_rng_ns`) was used for intersection A and B service times in sequence, violating the one-stream-per-source requirement. Split into two independent streams (`service_rng_ns_A`, `service_rng_ns_B`) propagated through the full call chain.

**7. Single turning RNG shared between A and B decisions (§2.1 stream contamination)**
A vehicle turning at A draws 1 value from `turn_rng`; one going straight draws 2. This caused B-turning decisions to shift depending on A's outcomes. Split into `turn_rng_A` and `turn_rng_B` so the streams are fully independent.

---

## Better Practices

**8. `link._traverse()` called directly instead of public API**
`southbound_vehicle_process` bypassed encapsulation by calling the private `link._traverse(vehicle)`. Changed to `yield link.enter(vehicle)`.

**9. `base_seed = 1000` buried in a function default**
The base seed was hidden inside `run_experiment`'s signature with no module-level visibility. Promoted to a top-level constant `BASE_SEED = 1000` so it can be found and changed in one place.

**10. Hardcoded `offset=20` in main block**
The coordination offset φ = L/vf was written as the literal `20` in the scenario calls. Added `'phi': 20` to PARAMS and replaced both occurrences with `PARAMS['phi']`.

**11. Hardcoded `n_reps=20` and `warmup=900` in function defaults**
Several function signatures duplicated values already in PARAMS, creating a silent divergence risk if PARAMS is changed. Updated all defaults to `PARAMS['n_reps']` and `PARAMS['warmup']`.

**12. Stale comment on queue tracking**
The `Intersection` class comment said "counts ALL waiting vehicles (for green + server)" which became incorrect after fix #5. Updated to reflect the new semantics (waiting only, excludes vehicle in service).

**13. Unused imports**
`field` (from `dataclasses`) and `Optional` (from `typing`) were imported but never referenced. Removed.
