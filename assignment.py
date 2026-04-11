"""
IE 306 — Assignment 2: Coordinated Traffic Corridor Simulation
Two-intersection corridor with signal control, bus priority,
emergency preemption, and downstream blocking.
"""

import simpy
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional

# ============================================================
# 1. PARAMETERS
# ============================================================

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
        """Main signal cycle loop."""
        # Initial offset delay (for coordination)
        if self.offset > 0:
            # During offset, WE gets green first
            self._set_phase(self.WE_GREEN)
            self._fire_we_green()
            try:
                yield self.env.timeout(self.offset)
            except simpy.Interrupt:
                return  # preemption during offset

            self._reset_we_event()

            # Amber after the partial WE phase
            self._set_phase(self.AMBER_WE)
            try:
                yield self.env.timeout(self.p['amber'])
            except simpy.Interrupt:
                return

        while True:
            # --- NS GREEN ---
            self._set_phase(self.NS_GREEN)
            self._fire_ns_green()
            try:
                yield self.env.timeout(self.p['g_NS'])
            except simpy.Interrupt:
                return  # preemption takes over

            self._reset_ns_event()

            # --- AMBER (after NS) ---
            self._set_phase(self.AMBER_NS)
            try:
                yield self.env.timeout(self.p['amber'])
            except simpy.Interrupt:
                return

            # --- WE GREEN ---
            self._set_phase(self.WE_GREEN)
            self._fire_we_green()
            try:
                yield self.env.timeout(self.p['g_WE'])
            except simpy.Interrupt:
                return

            self._reset_we_event()

            # --- AMBER (after WE) ---
            self._set_phase(self.AMBER_WE)
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

        # Record where we are in the cycle
        self.interrupted_phase = self.phase
        # Calculate remaining time in current phase
        # (we'll approximate by saving phase identity for resume)

        # Interrupt the current cycle process
        if self.cycle_process and self.cycle_process.is_alive:
            self.cycle_process.interrupt()

        # Start preemption sequence
        preempt_proc = self.env.process(self._preemption_sequence())
        return preempt_proc

    def _preemption_sequence(self):
        """Execute preemption: clearance -> emergency green -> resume."""
        preempt_start = self.env.now

        # Reset all green events
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

        # 3. Resume normal cycle
        # Amber transition back
        self._set_phase(self.AMBER_NS)
        yield self.env.timeout(self.p['amber'])

        recovery_time = self.env.now - preempt_start
        self.recovery_times.append(recovery_time)

        self.preempting = False

        # Restart the normal cycle from the beginning
        # (simplified: restart from NS green)
        self.cycle_process = self.env.process(self._run_cycle())

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


# ============================================================
# 6. VEHICLE PROCESSES
# ============================================================

def ns_vehicle_at_intersection(env, vehicle, intersection, service_rng,
                               turn_rng, turn_prob, link=None,
                               link_entry=True, stats=None, warmup=900):
    """
    Process for a N/S vehicle at one intersection.
    
    Steps:
    1. Wait for NS green
    2. Request server with priority
    3. Check downstream blocking (if going straight to link)
    4. Cross intersection (service time)
    5. Decide: turn E or go straight
    
    Returns: 'straight' or 'turned'
    """
    is_emergency = vehicle.vtype == 'emergency'

    # --- Emergency preemption logic ---
    if is_emergency and not intersection.signal.is_ns_green():
        preempt_proc = intersection.signal.request_preemption()
        if preempt_proc is not None:
            yield preempt_proc  # wait for clearance + green
            # After preemption, NS is green now
            # Small delay to let the phase settle
            yield env.timeout(0)

    # --- Wait for NS green ---
    while not intersection.signal.is_ns_green():
        yield intersection.signal.wait_for_ns_green()
        yield env.timeout(0)  # let phase events settle

    # --- If going straight and link exists, check blocking ---
    if link_entry and link is not None and not is_emergency:
        # Must wait for BOTH: NS green AND link not full
        while intersection.signal.is_ns_green() and link.is_full():
            # Wait for a slot to open, but re-check green after
            yield link.wait_for_slot()
            yield env.timeout(0)

        # Re-check NS green (might have turned red while waiting for slot)
        while not intersection.signal.is_ns_green():
            yield intersection.signal.wait_for_ns_green()
            yield env.timeout(0)

        # Track blocking
        if stats is not None and link.is_full() and env.now > warmup:
            stats['blocking_events'].append(env.now)

    # --- Request server (priority queue) ---
    req = intersection.ns_server.request(priority=vehicle.priority)
    yield req

    # --- Double-check signal is still green ---
    # (In edge cases, signal might have changed during queue wait)
    if not intersection.signal.is_ns_green() and not is_emergency:
        intersection.ns_server.release(req)
        # Need to re-wait — recursive-like via loop
        # Release and re-enter the queue next green
        yield intersection.signal.wait_for_ns_green()
        yield env.timeout(0)
        req = intersection.ns_server.request(priority=vehicle.priority)
        yield req

    # --- Service (crossing the intersection) ---
    service_time = service_rng.exponential(PARAMS['mu_s'])
    yield env.timeout(service_time)

    # --- Release server ---
    intersection.ns_server.release(req)

    # --- Turning decision ---
    if turn_rng.random() < turn_prob:
        vehicle.turned_at = intersection.name
        return 'turned'
    
    return 'straight'


def we_vehicle_process(env, vehicle, intersection, service_rng,
                       stats=None, warmup=900):
    """
    Process for a W/E vehicle at one intersection.
    Simpler: wait for WE green, request server, cross, exit.
    """
    arrive_time = env.now

    # Wait for WE green
    while not intersection.signal.is_we_green():
        yield intersection.signal.wait_for_we_green()
        yield env.timeout(0)

    # Request server
    req = intersection.we_server.request(priority=PRIORITY_WE_CAR)
    yield req

    # Check still green
    if not intersection.signal.is_we_green():
        intersection.we_server.release(req)
        yield intersection.signal.wait_for_we_green()
        yield env.timeout(0)
        req = intersection.we_server.request(priority=PRIORITY_WE_CAR)
        yield req

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
                                service_rng, turn_rng, stats, warmup=900):
    """
    Full lifecycle of a southbound (N/S) vehicle entering at A.
    
    A -> (maybe turn) -> link -> B -> (maybe turn) -> exit north
    """
    vehicle.arrival_at_A = env.now

    # --- Intersection A ---
    vehicle.service_start_A = env.now
    result_A = yield env.process(
        ns_vehicle_at_intersection(
            env, vehicle, intA, service_rng, turn_rng,
            turn_prob=PARAMS['turn_A'],
            link=link, link_entry=True,
            stats=stats, warmup=warmup
        )
    )
    vehicle.departure_A = env.now

    if result_A == 'turned':
        # Vehicle turned E at A — done
        vehicle.exited = True
        if env.now > warmup and stats is not None:
            delay_A = vehicle.departure_A - vehicle.arrival_at_A
            stats['turned_A_delay'].append(delay_A)
        return

    # --- Link A -> B ---
    yield env.process(link._traverse(vehicle))

    vehicle.arrival_at_B = env.now

    # --- Intersection B ---
    vehicle.service_start_B = env.now
    result_B = yield env.process(
        ns_vehicle_at_intersection(
            env, vehicle, intB, service_rng, turn_rng,
            turn_prob=PARAMS['turn_B'],
            link=None, link_entry=False,
            stats=stats, warmup=warmup
        )
    )
    vehicle.departure_B = env.now
    vehicle.exited = True

    # --- Record stats ---
    if env.now > warmup and stats is not None:
        total_delay = vehicle.departure_B - vehicle.arrival_at_A
        if result_B == 'turned':
            stats['turned_B_delay'].append(total_delay)
        else:
            stats['through_delay'].append(total_delay)
            stats['throughput_exits'].append(env.now)


# ============================================================
# 7. ARRIVAL GENERATORS
# ============================================================

def south_arrival_generator(env, intA, intB, link, arrival_rng,
                             classify_rng, service_rng, turn_rng,
                             stats, warmup, enable_emergency=True):
    """
    Poisson arrivals from south (λ=900 veh/h).
    Poisson splitting: 85% car, 10% bus, 5% emergency.
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
            service_rng, turn_rng, stats, warmup
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
    """Compute time-weighted average from (time, value) log."""
    if not log:
        return 0.0
    total = 0.0
    count = 0
    for i in range(len(log) - 1):
        t_i, v_i = log[i]
        t_next, _ = log[i + 1]
        if t_next <= warmup:
            continue
        t_start = max(t_i, warmup)
        t_end = t_next
        if t_end > t_start:
            total += v_i * (t_end - t_start)
            count += 1
    duration = sim_time - warmup
    return total / duration if duration > 0 else 0.0


def compute_kpis(stats, link, sim_time, warmup):
    """Compute all KPIs from collected statistics."""
    duration = sim_time - warmup
    kpis = {}

    # 1. Average delay (through NS traffic)
    if stats['through_delay']:
        kpis['avg_delay_through'] = np.mean(stats['through_delay'])
    else:
        kpis['avg_delay_through'] = 0.0

    # 1b. Average delay (WE traffic)
    we_delays = []
    for key in ['we_delay_A', 'we_delay_B']:
        we_delays.extend(stats.get(key, []))
    kpis['avg_delay_WE'] = np.mean(we_delays) if we_delays else 0.0

    # 2. Queue lengths (we'll use NS server queue)
    # (tracked via resource monitoring — simplified here)

    # 3. Throughput (vehicles/hour exiting north at B)
    exits = [t for t in stats.get('throughput_exits', []) if t > warmup]
    kpis['throughput_per_hour'] = len(exits) / (duration / 3600) if duration > 0 else 0

    # 4. Average link occupancy
    kpis['avg_link_occupancy'] = compute_time_average(
        link.occupancy_log, sim_time, warmup
    )

    # 5. Downstream blocking frequency
    # (fraction of cycles with at least one blocking event)
    n_cycles = duration / PARAMS['C']
    blocking_times = [t for t in stats.get('blocking_events', []) if t > warmup]
    if n_cycles > 0:
        # Count unique cycles with blocking
        blocked_cycles = set()
        for t in blocking_times:
            cycle_num = int((t - warmup) / PARAMS['C'])
            blocked_cycles.add(cycle_num)
        kpis['blocking_frequency'] = len(blocked_cycles) / n_cycles
    else:
        kpis['blocking_frequency'] = 0.0

    # 6. Preemption stats
    kpis['n_served_through'] = len(stats.get('through_delay', []))
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
    # --- Create separate RNG streams ---
    ss = np.random.SeedSequence(seed)
    seeds = ss.spawn(8)
    arrival_south_rng = np.random.default_rng(seeds[0])
    classify_rng = np.random.default_rng(seeds[1])
    service_rng_ns = np.random.default_rng(seeds[2])
    turn_rng = np.random.default_rng(seeds[3])
    arrival_we_A_rng = np.random.default_rng(seeds[4])
    arrival_we_B_rng = np.random.default_rng(seeds[5])
    service_rng_we_A = np.random.default_rng(seeds[6])
    service_rng_we_B = np.random.default_rng(seeds[7])

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
        arrival_south_rng, classify_rng, service_rng_ns, turn_rng,
        stats, PARAMS['warmup'], enable_emergency
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
    kpis = compute_kpis(stats, link, PARAMS['sim_time'], PARAMS['warmup'])

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

def run_experiment(scenario_name, offset, enable_emergency, n_reps=20,
                   base_seed=1000):
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
        offset=20, enable_emergency=False
    )

    # Scenario 3: Coordinated, with emergencies
    results_3 = run_experiment(
        "3: Coordinated, with emergencies",
        offset=20, enable_emergency=True
    )

    print("\n\nDone! All three scenarios completed.")