"""
IE 306 — Assignment 2: Coordinated Traffic Corridor Simulation
Two-intersection corridor with signal control, bus priority,
emergency preemption, and downstream blocking.
"""

import simpy
import numpy as np
from dataclasses import dataclass
from collections import defaultdict

# ============================================================
# 1. PARAMETERS & REPRODUCIBILITY
# ============================================================

# Base seed for all replications.  Change this single constant to reproduce
# or vary the entire experiment.  Replication r uses SeedSequence(BASE_SEED + r*100).
BASE_SEED = 1000

PARAMS = {
    # Geometry
    'L': 300,            # link length (m)
    'vf': 15,            # free-flow speed (m/s)
    'kj': 133,           # jam density (veh/km)
    'Nmax': 40,          # link capacity = kj * L/1000

    # Signal timing (seconds)
    'C': 90,             # cycle length
    'g_NS': 45,          # N/S green duration
    'amber': 4,          # amber/all-red clearance
    'g_WE': 37,          # W/E green duration
    'phi': 20,           # coordination offset = L/vf = 300/15 (s)

    # Arrivals
    'lambda_S': 900/3600,    # S entry rate (veh/s)
    'lambda_WE': 400/3600,   # W/E rate per intersection (veh/s)

    # Vehicle mix (south entry)
    'p_car': 0.85,
    'p_bus': 0.10,
    'p_emergency': 0.05,

    # Turning proportions (northbound)
    'turn_A': 0.30,      # fraction turning E at A
    'turn_B': 0.20,      # fraction turning E at B

    # Service
    'mu_s': 2.0,         # mean intersection service time (s)

    # Emergency
    'emergency_green': 10,   # min green for emergency (s)
    'emergency_clearance': 4, # all-red before emergency green (s)

    # Simulation
    'sim_time': 7200,    # total run time (s)
    'warmup': 900,       # warmup discard (s)
    'n_reps': 20,        # replications
}

# Priority values (lower = higher priority in SimPy)
PRIORITY_EMERGENCY = 0
PRIORITY_BUS = 1
PRIORITY_CAR = 2
PRIORITY_WE_CAR = 2   # W/E traffic = normal car priority


# ============================================================
# 2. DATA CLASSES
# ============================================================

@dataclass
class Vehicle:
    id: int
    vtype: str           # 'car', 'bus', 'emergency'
    direction: str       # 'NS' or 'WE'
    created_at: float    # env.now at creation
    arrival_at_A: float = 0.0
    service_start_A: float = 0.0
    departure_A: float = 0.0
    arrival_at_B: float = 0.0
    service_start_B: float = 0.0
    departure_B: float = 0.0
    turned_at: str = ''  # 'A', 'B', or '' (through)
    exited: bool = False
    service_time_A: float = 0.0
    service_time_B: float = 0.0

    @property
    def priority(self):
        if self.vtype == 'emergency':
            return PRIORITY_EMERGENCY
        elif self.vtype == 'bus':
            return PRIORITY_BUS
        return PRIORITY_CAR


# ============================================================
# 3. SIGNAL CONTROLLER
# ============================================================

class SignalController:
    """
    Two-phase fixed-time signal controller for one intersection.
    
    Phases: NS_GREEN -> AMBER -> WE_GREEN -> AMBER -> repeat
    
    Supports emergency preemption:
    - Interrupts current phase
    - Inserts 4s all-red clearance
    - Activates NS green for at least 10s
    - Resumes cycle from interruption point
    """

    # Phase constants
    NS_GREEN = 'NS_GREEN'
    AMBER_NS = 'AMBER_NS'   # amber after NS green
    WE_GREEN = 'WE_GREEN'
    AMBER_WE = 'AMBER_WE'   # amber after WE green
    PREEMPT_CLEARANCE = 'PREEMPT_CLEARANCE'
    PREEMPT_GREEN = 'PREEMPT_GREEN'

    def __init__(self, env, name, params, offset=0, stats=None):
        self.env = env
        self.name = name
        self.p = params
        self.offset = offset
        self.stats = stats

        # Current state
        self.phase = None
        self.ns_green_on = False
        self.we_green_on = False
        self.amber_on = False  # all-red clearance active

        # Events that vehicles wait on
        self.ns_green_event = env.event()   # fired when NS goes green
        self.we_green_event = env.event()   # fired when WE goes green

        # Preemption tracking
        self.preempting = False
        self.cycle_process = None
        self.interrupted_phase = None
        self.interrupted_remaining = 0
        self.preemption_count = 0
        self.recovery_times = []

        # Tracks the scheduled end-time of the current phase (for interrupted_remaining)
        self._phase_end_time = 0

        # Start the cycle
        self.cycle_process = env.process(self._run_cycle())

    def _set_phase(self, phase):
        """Update phase and green/amber flags."""
        self.phase = phase
        self.ns_green_on = (phase == self.NS_GREEN or phase == self.PREEMPT_GREEN)
        self.we_green_on = (phase == self.WE_GREEN)
        self.amber_on = (phase in (self.AMBER_NS, self.AMBER_WE, self.PREEMPT_CLEARANCE))

    def _fire_ns_green(self):
        """Notify all NS vehicles waiting for green."""
        if self.ns_green_event.triggered:
            self.ns_green_event = self.env.event()
        if not self.ns_green_event.triggered:
            self.ns_green_event.succeed()

    def _fire_we_green(self):
        """Notify all WE vehicles waiting for green."""
        if self.we_green_event.triggered:
            self.we_green_event = self.env.event()
        if not self.we_green_event.triggered:
            self.we_green_event.succeed()

    def _reset_ns_event(self):
        """Create a fresh event for NS vehicles to wait on."""
        if self.ns_green_event.triggered:
            self.ns_green_event = self.env.event()

    def _reset_we_event(self):
        """Create a fresh event for WE vehicles to wait on."""
        if self.we_green_event.triggered:
            self.we_green_event = self.env.event()

    def _run_cycle(self):
        """Main signal cycle loop — handles the one-time offset, then delegates to
        _run_normal_cycle.  The offset logic must NOT be re-entered after a
        preemption; post-preemption restarts use _run_normal_cycle directly."""
        if self.offset > 0:
            # WE green for (offset − amber) seconds so that, after the amber,
            # NS_GREEN starts exactly at t = offset (matching the green-wave offset φ).
            we_init = self.offset - self.p['amber']
            self._set_phase(self.WE_GREEN)
            self._fire_we_green()
            self._phase_end_time = self.env.now + we_init
            try:
                yield self.env.timeout(we_init)
            except simpy.Interrupt:
                return
            self._reset_we_event()

            self._set_phase(self.AMBER_WE)
            self._phase_end_time = self.env.now + self.p['amber']
            try:
                yield self.env.timeout(self.p['amber'])
            except simpy.Interrupt:
                return

        yield from self._run_normal_cycle()

    def _run_normal_cycle(self):
        """Repeating NS/amber/WE/amber cycle with phase-end-time tracking.
        Used both by _run_cycle (normal start) and after preemption resume."""
        while True:
            # --- NS GREEN ---
            self._set_phase(self.NS_GREEN)
            self._fire_ns_green()
            self._phase_end_time = self.env.now + self.p['g_NS']
            try:
                yield self.env.timeout(self.p['g_NS'])
            except simpy.Interrupt:
                return
            self._reset_ns_event()

            # --- AMBER (after NS) ---
            self._set_phase(self.AMBER_NS)
            self._phase_end_time = self.env.now + self.p['amber']
            try:
                yield self.env.timeout(self.p['amber'])
            except simpy.Interrupt:
                return

            # --- WE GREEN ---
            self._set_phase(self.WE_GREEN)
            self._fire_we_green()
            self._phase_end_time = self.env.now + self.p['g_WE']
            try:
                yield self.env.timeout(self.p['g_WE'])
            except simpy.Interrupt:
                return
            self._reset_we_event()

            # --- AMBER (after WE) ---
            self._set_phase(self.AMBER_WE)
            self._phase_end_time = self.env.now + self.p['amber']
            try:
                yield self.env.timeout(self.p['amber'])
            except simpy.Interrupt:
                return

    def request_preemption(self):
        """
        Called by an emergency vehicle. Returns a process the
        vehicle should yield on (or None if NS already green).
        """
        if self.ns_green_on and not self.preempting:
            # NS already green — no preemption needed
            return None

        if self.preempting:
            # Already preempting — just wait for the existing preempt green
            return None

        # Need to preempt!
        self.preempting = True
        self.preemption_count += 1

        # Record where we are in the cycle and how much time remains in the phase
        self.interrupted_phase = self.phase
        self.interrupted_remaining = max(0.0, self._phase_end_time - self.env.now)

        # Interrupt the current cycle process
        if self.cycle_process and self.cycle_process.is_alive:
            self.cycle_process.interrupt()

        # Start preemption sequence
        preempt_proc = self.env.process(self._preemption_sequence())
        return preempt_proc

    def _preemption_sequence(self):
        """Execute preemption: clearance -> emergency green -> amber -> resume from
        interrupted point.  Does NOT re-execute the offset initialisation."""
        self._reset_ns_event()
        self._reset_we_event()

        # 1. All-red clearance (4s)
        self._set_phase(self.PREEMPT_CLEARANCE)
        yield self.env.timeout(self.p['emergency_clearance'])

        # 2. NS green for emergency (at least 10s)
        self._set_phase(self.PREEMPT_GREEN)
        self._fire_ns_green()
        yield self.env.timeout(self.p['emergency_green'])
        self._reset_ns_event()
        _emergency_end = self.env.now

        # 3. Amber transition back
        self._set_phase(self.AMBER_NS)
        yield self.env.timeout(self.p['amber'])

        self.preempting = False

        # 4. Complete the interrupted phase's remaining time, then restart the
        #    normal cycle.  Recovery time is recorded inside _resume_from_interrupted
        #    once the cycle is truly back on its regular track.
        yield from self._resume_from_interrupted(_emergency_end)

    def _resume_from_interrupted(self, emergency_end):
        """Serve out the remaining time of the phase that was interrupted, continue
        with any subsequent phases in the current half-cycle, then hand off to the
        normal repeating cycle.  Recovery time is measured from end-of-emergency
        green to the point where the normal cycle resumes."""
        phase = self.interrupted_phase
        remaining = self.interrupted_remaining

        if phase == self.AMBER_NS:
            # Finish remaining amber, then run WE_GREEN + AMBER_WE
            if remaining > 0:
                self._set_phase(self.AMBER_NS)
                self._phase_end_time = self.env.now + remaining
                try:
                    yield self.env.timeout(remaining)
                except simpy.Interrupt:
                    return

            self._set_phase(self.WE_GREEN)
            self._fire_we_green()
            self._phase_end_time = self.env.now + self.p['g_WE']
            try:
                yield self.env.timeout(self.p['g_WE'])
            except simpy.Interrupt:
                return
            self._reset_we_event()

            self._set_phase(self.AMBER_WE)
            self._phase_end_time = self.env.now + self.p['amber']
            try:
                yield self.env.timeout(self.p['amber'])
            except simpy.Interrupt:
                return

        elif phase == self.WE_GREEN:
            # Finish remaining WE green, then AMBER_WE
            if remaining > 0:
                self._set_phase(self.WE_GREEN)
                self._fire_we_green()
                self._phase_end_time = self.env.now + remaining
                try:
                    yield self.env.timeout(remaining)
                except simpy.Interrupt:
                    return
                self._reset_we_event()

            self._set_phase(self.AMBER_WE)
            self._phase_end_time = self.env.now + self.p['amber']
            try:
                yield self.env.timeout(self.p['amber'])
            except simpy.Interrupt:
                return

        elif phase == self.AMBER_WE:
            # Finish remaining amber, then fall straight into normal cycle
            if remaining > 0:
                self._set_phase(self.AMBER_WE)
                self._phase_end_time = self.env.now + remaining
                try:
                    yield self.env.timeout(remaining)
                except simpy.Interrupt:
                    return

        # Recovery time: from end of emergency green to when the normal cycle resumes
        self.recovery_times.append(self.env.now - emergency_end)

        # Restart the repeating cycle — no offset block
        self.cycle_process = self.env.process(self._run_normal_cycle())

    def is_ns_green(self):
        return self.ns_green_on

    def is_we_green(self):
        return self.we_green_on

    def is_amber(self):
        return self.amber_on

    def wait_for_ns_green(self):
        """Return an event that fires when NS goes green."""
        if self.ns_green_on:
            # Already green — return a pre-triggered event
            evt = self.env.event()
            evt.succeed()
            return evt
        # Reset if needed and return
        if self.ns_green_event.triggered:
            self.ns_green_event = self.env.event()
        return self.ns_green_event

    def wait_for_we_green(self):
        """Return an event that fires when WE goes green."""
        if self.we_green_on:
            evt = self.env.event()
            evt.succeed()
            return evt
        if self.we_green_event.triggered:
            self.we_green_event = self.env.event()
        return self.we_green_event


# ============================================================
# 4. LINK MODEL
# ============================================================

class Link:
    """
    Link between intersection A and B.
    Finite capacity (Nmax=40), congestion-dependent travel time.
    """

    def __init__(self, env, params, stats=None):
        self.env = env
        self.p = params
        self.n = 0          # current occupancy
        self.Nmax = params['Nmax']
        self.L = params['L']
        self.vf = params['vf']
        self.free_flow_time = self.L / self.vf  # 20s
        self.stats = stats

        # Event fired when a vehicle leaves (slot opens)
        self.slot_available = env.event()

        # Occupancy log for time-average calculation
        self.occupancy_log = []

    def is_full(self):
        return self.n >= self.Nmax

    def travel_time(self):
        """Compute travel time based on current occupancy."""
        if self.n >= self.Nmax:
            return self.free_flow_time  # fallback for emergency
        ratio = self.n / self.Nmax
        return self.free_flow_time / (1 - ratio)

    def enter(self, vehicle):
        """
        Vehicle enters the link. Returns a process to yield on.
        Normal vehicles must check is_full() before calling.
        Emergency vehicles can enter regardless.
        """
        return self.env.process(self._traverse(vehicle))

    def _traverse(self, vehicle):
        """Vehicle traverses the link."""
        # Record entry
        tt = self.travel_time()
        self.n += 1
        self._log_occupancy()

        # Travel
        yield self.env.timeout(tt)

        # Exit link
        self.n -= 1
        self._log_occupancy()

        # Notify anyone waiting for a slot
        if not self.slot_available.triggered:
            self.slot_available.succeed()
        self.slot_available = self.env.event()

    def wait_for_slot(self):
        """Wait until a slot becomes available."""
        if not self.is_full():
            evt = self.env.event()
            evt.succeed()
            return evt
        if self.slot_available.triggered:
            self.slot_available = self.env.event()
        return self.slot_available

    def _log_occupancy(self):
        self.occupancy_log.append((self.env.now, self.n))


# ============================================================
# 5. INTERSECTION (combines signal + service)
# ============================================================

class Intersection:
    """
    One intersection with:
    - A signal controller
    - A PriorityResource for NS approach (capacity=1)
    - A PriorityResource for WE approach (capacity=1)
    - Queue length tracking for both approaches
    """

    def __init__(self, env, name, params, offset=0, stats=None):
        self.env = env
        self.name = name
        self.p = params
        self.stats = stats

        # Signal controller
        self.signal = SignalController(env, name, params, offset, stats)

        # Service resources (single server per approach)
        self.ns_server = simpy.PriorityResource(env, capacity=1)
        self.we_server = simpy.PriorityResource(env, capacity=1)

        # Queue length tracking: vehicles waiting to start service (excludes vehicle
        # actively crossing the intersection — ns_leave/we_leave fires on service start)
        self.ns_waiting = 0
        self.we_waiting = 0
        self.ns_queue_log = []   # list of (time, count)
        self.we_queue_log = []

    def ns_join(self):
        self.ns_waiting += 1
        self.ns_queue_log.append((self.env.now, self.ns_waiting))

    def ns_leave(self):
        self.ns_waiting -= 1
        self.ns_queue_log.append((self.env.now, self.ns_waiting))

    def we_join(self):
        self.we_waiting += 1
        self.we_queue_log.append((self.env.now, self.we_waiting))

    def we_leave(self):
        self.we_waiting -= 1
        self.we_queue_log.append((self.env.now, self.we_waiting))


# ============================================================
# 6. VEHICLE PROCESSES
# ============================================================

def ns_vehicle_at_intersection(env, vehicle, intersection, service_rng,
                               will_turn, link=None,
                               stats=None, warmup=PARAMS['warmup']):
    """
    Process for a N/S vehicle at one intersection.

    Args:
        will_turn: pre-decided bool — True if this vehicle turns east here.
        link:      downstream link; only needed when will_turn is False.

    Steps:
        1. Join queue count
        2. Emergency preemption (if applicable)
        3. Wait for NS green AND (if going straight) link not full
        4. Request server with priority
        5. Cross intersection (service time)
        6. Leave queue count

    Returns: ('straight' | 'turned', service_time)
    """
    is_emergency = vehicle.vtype == 'emergency'

    # --- Join queue ---
    intersection.ns_join()

    # --- Emergency preemption logic ---
    if is_emergency and not intersection.signal.is_ns_green():
        preempt_proc = intersection.signal.request_preemption()
        if preempt_proc is not None:
            yield preempt_proc
            yield env.timeout(0)

    # --- Combined wait: NS green AND (turning OR link has space) ---
    while True:
        # Wait for NS green
        while not intersection.signal.is_ns_green():
            yield intersection.signal.wait_for_ns_green()
            yield env.timeout(0)

        # Straight non-emergency vehicles also need link capacity
        if not will_turn and link is not None and not is_emergency:
            if link.is_full():
                # Blocking: NS is green but link is full
                if stats is not None and env.now > warmup:
                    stats['blocking_events'].append(env.now)
                yield link.wait_for_slot()
                yield env.timeout(0)
                continue  # re-check green after slot opens
        break  # both conditions satisfied

    # --- Request server (priority queue) ---
    req = intersection.ns_server.request(priority=vehicle.priority)
    yield req

    # --- Double-check signal (edge case: phase changed while in server queue) ---
    if not intersection.signal.is_ns_green() and not is_emergency:
        intersection.ns_server.release(req)
        while not intersection.signal.is_ns_green():
            yield intersection.signal.wait_for_ns_green()
            yield env.timeout(0)
        req = intersection.ns_server.request(priority=vehicle.priority)
        yield req

    # --- Leave queue when service begins (vehicle is crossing, no longer waiting) ---
    intersection.ns_leave()

    # --- Service (crossing the intersection) ---
    service_time = service_rng.exponential(PARAMS['mu_s'])
    yield env.timeout(service_time)

    # --- Release server ---
    intersection.ns_server.release(req)

    if will_turn:
        vehicle.turned_at = intersection.name
        return 'turned', service_time

    return 'straight', service_time


def we_vehicle_process(env, vehicle, intersection, service_rng,
                       stats=None, warmup=PARAMS['warmup']):
    """
    Process for a W/E vehicle at one intersection.
    Simpler: wait for WE green, request server, cross, exit.
    """
    arrive_time = env.now

    # Join queue
    intersection.we_join()

    # Wait for WE green
    while not intersection.signal.is_we_green():
        yield intersection.signal.wait_for_we_green()
        yield env.timeout(0)

    # Request server
    req = intersection.we_server.request(priority=PRIORITY_WE_CAR)
    yield req

    # Check still green (edge case)
    if not intersection.signal.is_we_green():
        intersection.we_server.release(req)
        while not intersection.signal.is_we_green():
            yield intersection.signal.wait_for_we_green()
            yield env.timeout(0)
        req = intersection.we_server.request(priority=PRIORITY_WE_CAR)
        yield req

    # Leave queue when service begins (vehicle is crossing, no longer waiting)
    intersection.we_leave()

    # Service
    svc = service_rng.exponential(PARAMS['mu_s'])
    yield env.timeout(svc)

    intersection.we_server.release(req)

    # Record stats
    if stats is not None and env.now > warmup:
        delay = env.now - arrive_time
        stats[f'we_delay_{intersection.name}'].append(delay)
    vehicle.exited = True


def southbound_vehicle_process(env, vehicle, intA, intB, link,
                                service_rng_A, service_rng_B,
                                turn_rng_A, turn_rng_B, stats,
                                warmup=PARAMS['warmup']):
    """
    Full lifecycle of a N/S vehicle entering at A.

    Turning is pre-decided on arrival at each intersection so that the
    downstream blocking check only applies to straight-going vehicles.

    Delay definition: total corridor time minus free-flow link travel time
    (20 s).  Intersection service times are included in the reported delay
    because they may involve signal-induced queuing.  W/E delay uses raw
    time-in-system at the intersection — both approaches are documented in
    the report as the assumed reference for each direction.

    Args:
        service_rng_A: independent RNG stream for NS service at intersection A.
        service_rng_B: independent RNG stream for NS service at intersection B.
        turn_rng_A:    independent RNG stream for turning decision at A.
        turn_rng_B:    independent RNG stream for turning decision at B.
                       Keeping A and B separate prevents stream contamination:
                       a vehicle that turns at A draws 0 values from turn_rng_B,
                       so B decisions are not shifted by A's turning outcomes.
    """
    vehicle.arrival_at_A = env.now

    # Pre-decide turning at A (before joining queue, so blocking check
    # inside ns_vehicle_at_intersection is applied only to through vehicles)
    will_turn_A = (turn_rng_A.random() < PARAMS['turn_A'])

    # --- Intersection A ---
    vehicle.service_start_A = env.now
    result_A, svc_A = yield env.process(
        ns_vehicle_at_intersection(
            env, vehicle, intA, service_rng_A,
            will_turn=will_turn_A,
            link=link,
            stats=stats, warmup=warmup
        )
    )
    vehicle.departure_A = env.now
    vehicle.service_time_A = svc_A

    if result_A == 'turned':
        vehicle.exited = True
        if env.now > warmup and stats is not None:
            stats['turned_A_delay'].append(vehicle.departure_A - vehicle.arrival_at_A)
        return

    # --- Link A -> B ---
    yield link.enter(vehicle)
    vehicle.arrival_at_B = env.now

    # Pre-decide turning at B
    will_turn_B = (turn_rng_B.random() < PARAMS['turn_B'])

    # --- Intersection B ---
    vehicle.service_start_B = env.now
    result_B, svc_B = yield env.process(
        ns_vehicle_at_intersection(
            env, vehicle, intB, service_rng_B,
            will_turn=will_turn_B,
            link=None,
            stats=stats, warmup=warmup
        )
    )
    vehicle.departure_B = env.now
    vehicle.service_time_B = svc_B
    vehicle.exited = True

    # --- Record stats (post-warmup only) ---
    if env.now > warmup and stats is not None:
        # Delay = total time − free-flow link travel time (20 s)
        free_flow_link = PARAMS['L'] / PARAMS['vf']
        delay = (vehicle.departure_B - vehicle.arrival_at_A) - free_flow_link

        if result_B == 'turned':
            stats['turned_B_delay'].append(delay)
        else:
            # Separate emergency delays so they don't dilute car/bus average
            if vehicle.vtype == 'emergency':
                stats['emergency_through_delay'].append(delay)
            else:
                stats['through_delay'].append(delay)
            stats['throughput_exits'].append(env.now)


# ============================================================
# 7. ARRIVAL GENERATORS
# ============================================================

def south_arrival_generator(env, intA, intB, link, arrival_rng,
                             classify_rng, service_rng_A, service_rng_B,
                             turn_rng_A, turn_rng_B,
                             stats, warmup, enable_emergency=True):
    """
    Poisson arrivals from south (λ=900 veh/h).
    Poisson splitting: 85% car, 10% bus, 5% emergency.
    service_rng_A / service_rng_B: independent NS service streams per intersection.
    turn_rng_A / turn_rng_B: separate turning-decision streams to prevent stream
    contamination (vehicles that turn at A never draw from turn_rng_B).
    """
    vid = 0
    while True:
        # Interarrival time
        iat = arrival_rng.exponential(1.0 / PARAMS['lambda_S'])
        yield env.timeout(iat)

        vid += 1

        # Classify vehicle
        r = classify_rng.random()
        if r < PARAMS['p_car']:
            vtype = 'car'
        elif r < PARAMS['p_car'] + PARAMS['p_bus']:
            vtype = 'bus'
        else:
            if enable_emergency:
                vtype = 'emergency'
            else:
                vtype = 'car'  # replace with car when disabled

        vehicle = Vehicle(id=vid, vtype=vtype, direction='NS',
                          created_at=env.now)

        # Launch vehicle process
        env.process(southbound_vehicle_process(
            env, vehicle, intA, intB, link,
            service_rng_A, service_rng_B, turn_rng_A, turn_rng_B, stats, warmup
        ))


def we_arrival_generator(env, intersection, arrival_rng, service_rng,
                          stats, warmup):
    """
    Poisson arrivals from west at one intersection (λ=400 veh/h).
    All regular cars.
    """
    vid = 0
    while True:
        iat = arrival_rng.exponential(1.0 / PARAMS['lambda_WE'])
        yield env.timeout(iat)

        vid += 1
        vehicle = Vehicle(id=vid, vtype='car', direction='WE',
                          created_at=env.now)

        env.process(we_vehicle_process(
            env, vehicle, intersection, service_rng,
            stats, warmup
        ))


# ============================================================
# 8. STATISTICS COLLECTION
# ============================================================

def compute_time_average(log, sim_time, warmup):
    """Time-weighted average from (time, value) log (used for link occupancy)."""
    if not log:
        return 0.0
    total = 0.0
    for i in range(len(log) - 1):
        t_i, v_i = log[i]
        t_next, _ = log[i + 1]
        if t_next <= warmup:
            continue
        t_start = max(t_i, warmup)
        t_end = t_next
        if t_end > t_start:
            total += v_i * (t_end - t_start)
    # Include the final interval from the last log entry to sim_time
    t_last, v_last = log[-1]
    t_start_last = max(t_last, warmup)
    if t_start_last < sim_time:
        total += v_last * (sim_time - t_start_last)
    duration = sim_time - warmup
    return total / duration if duration > 0 else 0.0


def compute_queue_stats(log, sim_time, warmup):
    """
    Compute max and time-weighted average queue length from (time, count) log.
    The last observed value is held until sim_time.
    """
    if not log:
        return 0, 0.0
    # Filter to post-warmup events, keeping the last pre-warmup value as initial
    last_before = 0
    for t, v in log:
        if t <= warmup:
            last_before = v
    relevant = [(max(t, warmup), v) for t, v in log if t >= warmup]
    if not relevant:
        return last_before, float(last_before)

    max_q = max(v for _, v in relevant)

    # Time-weighted average over [warmup, sim_time]
    total = 0.0
    # Interval from warmup to first event
    if relevant[0][0] > warmup:
        total += last_before * (relevant[0][0] - warmup)
    for i in range(len(relevant) - 1):
        t_i, v_i = relevant[i]
        t_next = relevant[i + 1][0]
        total += v_i * (t_next - t_i)
    # Last interval to sim_time
    total += relevant[-1][1] * (sim_time - relevant[-1][0])

    duration = sim_time - warmup
    avg_q = total / duration if duration > 0 else 0.0
    return max_q, avg_q


def compute_kpis(stats, link, intA, intB, sim_time, warmup):
    """Compute all KPIs from collected statistics."""
    duration = sim_time - warmup
    kpis = {}

    # 1a. Average delay — through NS traffic (cars and buses only)
    through = stats.get('through_delay', [])
    kpis['avg_delay_through'] = np.mean(through) if through else 0.0

    # 1b. Average delay — W/E traffic
    we_delays = stats.get('we_delay_A', []) + stats.get('we_delay_B', [])
    kpis['avg_delay_WE'] = np.mean(we_delays) if we_delays else 0.0

    # 1c. Average delay — emergency vehicles (Scenario 3)
    emg = stats.get('emergency_through_delay', [])
    kpis['avg_delay_emergency'] = np.mean(emg) if emg else 0.0

    # 2. Queue lengths (max and time-average) per approach
    for label, log in [('ns_A', intA.ns_queue_log),
                       ('we_A', intA.we_queue_log),
                       ('ns_B', intB.ns_queue_log),
                       ('we_B', intB.we_queue_log)]:
        max_q, avg_q = compute_queue_stats(log, sim_time, warmup)
        kpis[f'max_queue_{label}'] = max_q
        kpis[f'avg_queue_{label}'] = avg_q

    # 3. Throughput (vehicles/hour exiting north at B)
    exits = [t for t in stats.get('throughput_exits', []) if t > warmup]
    kpis['throughput_per_hour'] = len(exits) / (duration / 3600) if duration > 0 else 0

    # 4. Average link occupancy
    kpis['avg_link_occupancy'] = compute_time_average(
        link.occupancy_log, sim_time, warmup
    )

    # 5. Downstream blocking frequency
    #    (fraction of cycles in which ≥1 N/S vehicle at A was blocked)
    n_cycles = duration / PARAMS['C']
    blocking_times = [t for t in stats.get('blocking_events', []) if t > warmup]
    if n_cycles > 0:
        blocked_cycles = set()
        for t in blocking_times:
            cycle_num = int((t - warmup) / PARAMS['C'])
            blocked_cycles.add(cycle_num)
        kpis['blocking_frequency'] = len(blocked_cycles) / n_cycles
    else:
        kpis['blocking_frequency'] = 0.0

    kpis['n_served_through'] = len(through)
    kpis['n_served_WE'] = len(we_delays)

    return kpis


# ============================================================
# 9. SINGLE REPLICATION
# ============================================================

def run_replication(seed, offset=0, enable_emergency=True):
    """
    Run one replication of the traffic corridor simulation.
    
    Args:
        seed: base seed for RNG streams
        offset: signal offset for intersection B (0 or 20)
        enable_emergency: whether emergency vehicles are enabled
    
    Returns:
        dict of KPIs for this replication
    """
    # --- Create separate RNG streams (one per independent random source, §2.1) ---
    # Stream index → purpose
    #   0  south arrival inter-arrival times
    #   1  vehicle type classification (car / bus / emergency)
    #   2  NS intersection service times at A
    #   3  NS intersection service times at B
    #   4  turning decisions at A  ← separate from B to avoid stream contamination
    #   5  turning decisions at B    (vehicles that turn at A draw 0 values from this)
    #   6  W/E arrival inter-arrival times at A
    #   7  W/E arrival inter-arrival times at B
    #   8  W/E intersection service times at A
    #   9  W/E intersection service times at B
    ss = np.random.SeedSequence(seed)
    seeds = ss.spawn(10)
    arrival_south_rng  = np.random.default_rng(seeds[0])
    classify_rng       = np.random.default_rng(seeds[1])
    service_rng_ns_A   = np.random.default_rng(seeds[2])
    service_rng_ns_B   = np.random.default_rng(seeds[3])
    turn_rng_A         = np.random.default_rng(seeds[4])
    turn_rng_B         = np.random.default_rng(seeds[5])
    arrival_we_A_rng   = np.random.default_rng(seeds[6])
    arrival_we_B_rng   = np.random.default_rng(seeds[7])
    service_rng_we_A   = np.random.default_rng(seeds[8])
    service_rng_we_B   = np.random.default_rng(seeds[9])

    # --- Create SimPy environment ---
    env = simpy.Environment()

    # --- Statistics container ---
    stats = defaultdict(list)

    # --- Build model components ---
    intA = Intersection(env, 'A', PARAMS, offset=0, stats=stats)
    intB = Intersection(env, 'B', PARAMS, offset=offset, stats=stats)
    link = Link(env, PARAMS, stats=stats)

    # --- Start arrival generators ---
    env.process(south_arrival_generator(
        env, intA, intB, link,
        arrival_south_rng, classify_rng, service_rng_ns_A, service_rng_ns_B,
        turn_rng_A, turn_rng_B, stats, PARAMS['warmup'], enable_emergency
    ))

    env.process(we_arrival_generator(
        env, intA, arrival_we_A_rng, service_rng_we_A,
        stats, PARAMS['warmup']
    ))

    env.process(we_arrival_generator(
        env, intB, arrival_we_B_rng, service_rng_we_B,
        stats, PARAMS['warmup']
    ))

    # --- Run simulation ---
    env.run(until=PARAMS['sim_time'])

    # --- Compute KPIs ---
    kpis = compute_kpis(stats, link, intA, intB, PARAMS['sim_time'], PARAMS['warmup'])

    # Add preemption stats from signal controllers
    kpis['preemptions_A'] = intA.signal.preemption_count
    kpis['preemptions_B'] = intB.signal.preemption_count
    kpis['preemptions_total'] = (intA.signal.preemption_count +
                                  intB.signal.preemption_count)
    recovery = intA.signal.recovery_times + intB.signal.recovery_times
    kpis['avg_recovery_time'] = np.mean(recovery) if recovery else 0.0

    return kpis


# ============================================================
# 10. EXPERIMENT RUNNER
# ============================================================

def run_experiment(scenario_name, offset, enable_emergency,
                   n_reps=PARAMS['n_reps'], base_seed=BASE_SEED):
    """Run all replications for one scenario."""
    print(f"\n{'='*60}")
    print(f"Scenario: {scenario_name}")
    print(f"Offset={offset}s, Emergency={'ON' if enable_emergency else 'OFF'}")
    print(f"Replications: {n_reps}")
    print(f"{'='*60}")

    all_kpis = defaultdict(list)

    for rep in range(n_reps):
        seed = base_seed + rep * 100
        kpis = run_replication(seed, offset, enable_emergency)
        for k, v in kpis.items():
            all_kpis[k].append(v)
        if (rep + 1) % 5 == 0:
            print(f"  Completed {rep+1}/{n_reps} replications...")

    # Compute point estimates and 95% CIs
    results = {}
    for key, values in all_kpis.items():
        arr = np.array(values)
        mean = np.mean(arr)
        if n_reps > 1:
            se = np.std(arr, ddof=1) / np.sqrt(n_reps)
            # t-critical for 95% CI with n-1 df
            from scipy.stats import t as t_dist
            t_crit = t_dist.ppf(0.975, n_reps - 1)
            half_width = t_crit * se
        else:
            half_width = 0
        results[key] = {
            'mean': mean,
            'ci_low': mean - half_width,
            'ci_high': mean + half_width,
            'half_width': half_width
        }

    # Print results
    print(f"\n--- Results: {scenario_name} ---")
    for key, val in results.items():
        print(f"  {key:30s}: {val['mean']:10.2f}  "
              f"[{val['ci_low']:.2f}, {val['ci_high']:.2f}]")

    return results


# ============================================================
# 11. MAIN — RUN ALL SCENARIOS
# ============================================================

if __name__ == '__main__':
    print("IE 306 — Assignment 2: Traffic Corridor Simulation")
    print("=" * 60)

    # Scenario 1: Uncoordinated, no emergencies
    results_1 = run_experiment(
        "1: Uncoordinated, no emergencies",
        offset=0, enable_emergency=False
    )

    # Scenario 2: Coordinated, no emergencies
    results_2 = run_experiment(
        "2: Coordinated, no emergencies",
        offset=PARAMS['phi'], enable_emergency=False
    )

    # Scenario 3: Coordinated, with emergencies
    results_3 = run_experiment(
        "3: Coordinated, with emergencies",
        offset=PARAMS['phi'], enable_emergency=True
    )

    print("\n\nDone! All three scenarios completed.")