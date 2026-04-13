# Assignment 2 — Change Report (Round 2)

This file documents the second round of fixes applied to `assignment.py`.

## 1) Through-delay KPI now includes all S->N through vehicles

### Problem
`avg_delay_through` was computed from `through_delay` only (cars/buses), while emergency through vehicles were tracked separately and excluded from the main S->N through delay KPI.

### Fix
- Added `through_delay_all` collection at S->N exit for all through vehicles (car, bus, emergency).
- Updated KPI computation to use `through_delay_all` for `avg_delay_through`.
- Kept `avg_delay_emergency` as a separate Scenario 3 diagnostic KPI.
- Added backward-compatible fallback in KPI computation for old stats dictionaries.

### Impact
The S->N through-delay KPI now matches the assignment wording more directly by representing all through traffic classes.

## 2) Recovery-time anchor corrected to "end of emergency service"

### Problem
Recovery time was anchored at the end of fixed emergency-green duration (`emergency_green = 10s`), not necessarily the actual emergency service completion time.

### Fix
- Added controller method: `note_emergency_service_complete(t_done)`.
- Called it when an emergency vehicle finishes intersection service.
- During preemption, controller records the latest emergency service completion timestamp.
- Recovery-time anchor now uses this recorded timestamp; if none is recorded, it falls back to emergency-green end time.

### Impact
`avg_recovery_time` now aligns with the assignment definition: time from emergency service completion until regular cycle point recovery.

## 3) Warm-up filter switched from exit-time to vehicle-entry-time for delay samples

### Problem
Delay samples were included when `env.now > warmup`, meaning vehicles that entered during warm-up but exited later were counted.

### Fix
- Updated vehicle-based delay sampling conditions to use `vehicle.created_at >= warmup` in:
  - W/E delay recording
  - S->N turned-at-A delay recording
  - S->N through/turned-at-B delay recording
- Throughput storage now records `(arrival_time, exit_time)` tuples for robust warm-up filtering.
- Throughput KPI updated to read both tuple-based (new) and scalar-time (legacy) formats.

### Impact
Warm-up handling is now consistent with entry-based filtering for vehicle-level delay metrics, reducing warm-up contamination.

## Verification
- Executed full `assignment.py` run (all 3 scenarios, 20 replications each) after edits.
- Run completed successfully with no runtime errors.
