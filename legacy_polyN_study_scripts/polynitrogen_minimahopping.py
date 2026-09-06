#!/usr/bin/env python3
"""
polynitrogen_minimahopping.py
==============================
Global potential-energy-surface exploration of neutral all-nitrogen (Nn)
molecules using the Minima Hopping algorithm (Goedecker, J. Chem. Phys.
120, 9911, 2004), driven by a MACE machine-learned potential, to identify
thermodynamically metastable structures relative to molecular N2.

SCIENTIFIC OBJECTIVE
---------------------
For each even n in [n_min, n_max], discover as many DISTINCT local minima
of the Nn potential energy surface as possible (not just relax one
combinatorially-built topology to its nearest minimum, as the earlier
graph-enumeration script did). Each minimum found is referenced against
N2 via

    delta_E_per_atom(n) = E(Nn)/n - E(N2)/2          [eV/atom]

Plotting delta_E_per_atom against n (or 1/n) gives the metastability
landscape / convex-hull-style view requested: N2 sits at the origin
(delta_E = 0 by construction) and every other minimum found is, by
definition of being a local minimum distinct from n/2 x N2, a candidate
METASTABLE allotrope -- the deeper local minima at low delta_E being the
more promising synthetic targets.

WHY MINIMA HOPPING (not graph enumeration + single relaxation)
-----------------------------------------------------------------
Relaxing one starting topology to its nearest local minimum (the
approach of polynitrogen_mace.py) is biased by construction: LBFGS
converges to whichever minimum is closest to the start, not to the
minimum that best characterizes the n-atom PES. Minima Hopping instead
alternates short NVE molecular dynamics (which can cross energy barriers
when "hot" enough) with local relaxation, and adaptively raises the MD
temperature whenever it re-discovers an already-known minimum -- this
lets it escape a basin and find genuinely different local minima, rather
than only ever re-confirming the starting topology's own basin.
See the companion methodology manual, Section "Global PES exploration",
for the full derivation and literature comparison (Schonborn/Goedecker/
Oganov 2009; Krummenacher et al. 2024 ASE implementation).

SPIN / PARITY CONSTRAINT
--------------------------
Neutral N has 7 electrons (odd). For Nn neutral, total electron count is
7n. Only EVEN n gives an even electron count, compatible with a
closed-shell singlet ground state -- the implicit assumption of MACE
(trained on DFT singlet/closed-shell reference data with no explicit
spin channel). Odd n necessarily requires open-shell treatment (doublet
minimum), which MACE cannot represent. This script therefore restricts
n to EVEN values only; see manual for the full electron-counting
argument.

FRAGMENTATION CONTROL
-----------------------
Minima Hopping's MD phase explores at increasingly high temperature when
trapped, which can break N-N bonds entirely (real dissociation, or
unphysical artifacts of the MLIP far from its training distribution).
Two independent safeguards are used:
  1. ASE Hookean constraints on the bonds of the STARTING topology,
     which apply a restorative force (without violating energy
     conservation, unlike a hard bond-length cap) once a target distance
     is exceeded -- preventing outright separation of the starting
     molecular graph's bonded pairs during the MD phase.
  2. Post-hoc geometric integrity checking (reused from
     polynitrogen_mace.py) on every accepted minimum, classifying it as
     OK / FRAGMENTED / REARRANGED / BOND_ANOMALY / COLLAPSED.
A minimum classified as REARRANGED is not an error -- it is itself the
scientifically interesting case of a starting topology relaxing into a
DIFFERENT, possibly more stable, topology. Only FRAGMENTED/COLLAPSED
results are operationally suspect (optimizer/MLIP pathology) and are
flagged for exclusion from the convex-hull analysis.

INSTALLATION
-------------
    pip install mace-torch ase networkx numpy scipy spglib --break-system-packages

USAGE
------
    # Quick test: N4 and N6, 2 independent MH runs each, 10 steps per run
    python polynitrogen_minimahopping.py --n_min 4 --n_max 6 \\
        --n_seeds 2 --n_steps 10

    # Production run: N4 to N16, 5 seeds per size, 60 steps per run, GPU
    python polynitrogen_minimahopping.py --n_min 4 --n_max 16 \\
        --n_seeds 5 --n_steps 60 --device cuda \\
        --output_dir ./N4_N16_mh_results

AUTHOR
-------
    Generated for IC2MP/E4 Mediacat -- Universite de Poitiers
"""

import argparse
import csv
import logging
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np

import multiprocessing as mp

try:
    from ase import Atoms
    from ase.constraints import Hookean
    from ase.io import read, write
    from ase.optimize import LBFGS
    from ase.optimize.minimahopping import MinimaHopping
except ImportError:
    sys.exit("ERROR: ASE not found. Run: pip install ase")

try:
    import spglib
    SPGLIB_OK = True
except ImportError:
    warnings.warn("spglib not found -- symmetry detection disabled.")
    SPGLIB_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

NN_BOND_INIT = 1.35  # Angstrom, initial embedding bond length guess

# Structural integrity thresholds (see polynitrogen_mace.py manual section)
BOND_CUTOFF_NN = 1.60          # Angstrom, max distance counted as a covalent bond
MIN_SANE_NN_DISTANCE = 0.90    # Angstrom, below this = numerical collapse
MAX_SANE_NN_DISTANCE = 2.50    # Angstrom, sanity bound

# Energy sanity bound, independent of geometry. Guards against the MLIP
# extrapolating far outside its training distribution (e.g. a compact,
# hypervalent cluster, see manual) and returning a spuriously low total
# energy that would otherwise masquerade as an extremely stable
# "discovery". No physically reasonable Nn local minimum should have
# delta_E_per_atom more negative than this value; N2 itself defines
# delta_E = 0 by construction, and no polynitrogen species is expected
# to be more stable than N2 (the global thermodynamic sink for nitrogen).
# A generous negative margin (rather than exactly 0) avoids flagging
# small numerical/convergence noise around the N2 reference itself.
MIN_SANE_DELTA_E_PER_ATOM = -0.5   # eV/atom

# Upper sanity bound, added after a real production artifact was
# observed: a Minima Hopping "minimum" with delta_E_atom ~ +31 eV,
# nearly two orders of magnitude above any physically meaningful
# polynitrogen candidate (the heuristic screening band used throughout
# this project is 0-3 eV/atom; even a badly strained or hypervalent
# structure should not plausibly exceed a few tens of eV/atom above N2,
# since that would imply near-total disintegration of chemical bonding).
# Such values arise when the geometric integrity check (Section 6/8 of
# the companion manual) classifies a structure as REARRANGED or
# BOND_ANOMALY (both treated as "usable" by merge_polyN_results.py,
# since a topology change is itself scientifically informative) but the
# underlying MACE energy for that specific rearranged geometry is, in
# fact, a wild extrapolation artifact -- i.e. the geometric check did
# not happen to also classify it as COLLAPSED, while the energy itself
# is nonetheless nonsensical. This independent upper bound closes that
# gap: ANY delta_E_atom outside [MIN_SANE_DELTA_E_PER_ATOM,
# MAX_SANE_DELTA_E_PER_ATOM], regardless of which geometric status was
# assigned, is overridden to ENERGY_ANOMALY and excluded from
# downstream hull construction.
MAX_SANE_DELTA_E_PER_ATOM = 5.0    # eV/atom

# Hookean restraint: applied to bonds of the STARTING topology only.
# rt is the distance beyond which the restorative force activates; chosen
# above the longest plausible strained N-N single bond (~1.55 A) but well
# below van der Waals separation, so it only intervenes once a bond is
# genuinely being pulled apart, not during normal vibrational motion.
HOOKEAN_RT = 2.0       # Angstrom
HOOKEAN_K = 8.0        # eV/Angstrom^2, restoring force constant

# Minima Hopping parameters, RECALIBRATED for small molecules.
# ASE defaults (T0=1000 K, Ediff0=0.5 eV) are tuned for solids/clusters
# with many atoms and a correspondingly large heat capacity; for an
# n<20-atom molecule these defaults are far too aggressive (T0=1000K
# alone can dissociate small N-N systems within a few MD steps) and far
# too coarse in energy (0.5 eV is a huge fraction of typical N-N
# conformational energy differences). The values below were validated
# empirically on N4 (see manual) to explore basins without immediate
# runaway heating.
MH_T0 = 300.0           # K, initial MD temperature
MH_EDIFF0 = 0.05        # eV, initial energy acceptance window
MH_MDMIN = 2            # number of minima to pass through during MD before stopping
MH_FMAX = 0.05          # eV/Angstrom, local relaxation force convergence

# Maximum MD temperature allowed before a trajectory is force-terminated,
# via ASE's native MinimaHopping.__call__(maxtemp=...) argument.
#
# WHY THIS IS NECESSARY (discovered during a real production run): the
# adaptive heating rule (T <- beta1*T, beta1=1.1, every time the SAME
# minimum is re-found) has no upper bound in the base algorithm. For a
# small, highly constrained system like N4 -- few accessible basins, and
# a deep, narrow global-minimum-adjacent funnel -- a trajectory that
# keeps re-finding the same minimum can runaway-heat from T0=300 K past
# 5000 K within ~30-40 steps (observed directly: a seed reached
# T=5758 K by step 40). Beyond a few hundred K this is no longer
# physically meaningful nitrogen dynamics at all (way above any
# realistic thermal decomposition regime), AND it silently inflates the
# wall-clock cost per step (increasingly energetic, chaotic MD requires
# more internal integrator substeps to stay numerically stable),
# producing the practically confusing symptom of "the script appears to
# hang" on one particular seed while CPU usage stays at 100% the whole
# time -- the calculation IS progressing, just inside an unphysical and
# wastefully expensive regime.
#
# Capping at 3000 K (an already generous upper bound, ~3x the
# unphysical regime, chosen so genuinely difficult-but-physical barrier
# crossings are not prematurely cut off) ensures no single seed can
# consume disproportionate wall-clock time; a seed that hits the cap is
# logged and simply contributes whatever minima it found before hitting
# it, rather than continuing to "calculate" inside a regime with no
# further chemical meaning.
MH_MAX_TEMP = 3000.0    # K

# Per-seed wall-clock timeout (seconds), enforced EXTERNALLY to ASE's own
# control flow via signal.alarm (Unix/macOS) or a watcher thread (Windows
# fallback, less reliable for interrupting native/C extension calls).
#
# WHY THIS IS NECESSARY (root cause found by direct inspection of ASE's
# source, after a real production run hung indefinitely on a single
# seed despite max_temp being correctly enforced): ASE's internal
# _molecular_dynamics() method runs
#
#     while mincount < self._mdmin:
#         dyn.run(1)
#         ...
#
# with NO upper bound on the number of MD substeps inside this loop.
# mincount only increments when PassedMinimum() detects a specific
# energy-trajectory pattern (two downward points followed by two upward
# points, by default). If the VelocityVerlet trajectory enters a regime
# where this pattern never occurs -- e.g. a near-monotonic energy drift,
# or oscillations with a period that does not match the 4-point window
# -- this inner loop can run FOREVER, completely independent of and
# unreachable by the maxtemp safeguard above (which is only checked
# BETWEEN complete MD phases, never during one). This was observed
# directly: a seed's log froze at exactly "Molecular dynamics: md00001"
# / "Optimization: qn00001" -- i.e. inside the very first MD phase of
# the very first step -- while CPU usage stayed at 100%, for far longer
# than any of the surrounding seeds (which completed in 5-15 s each).
#
# A timeout of 120 s is deliberately generous relative to the ~5-15 s
# typically observed per full MH step (initial optimization + MD phase
# + relaxation) for n<20 systems on consumer hardware (CPU or MPS), so
# genuinely slow-but-converging steps are not prematurely killed, while
# guaranteeing no single seed can block a multi-size production
# campaign indefinitely.
MH_SEED_TIMEOUT = 120    # seconds


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("polyN-MH")


# ─────────────────────────────────────────────────────────────────────────────
# SEED TOPOLOGY GENERATION (reused logic from polynitrogen_mace.py, trimmed)
# ─────────────────────────────────────────────────────────────────────────────
#
# Minima Hopping needs a handful of DIVERSE starting topologies per size n
# (not an exhaustive enumeration -- that combinatorial coverage is the job
# of polynitrogen_mace.py). Here we deliberately generate only a few
# qualitatively different seeds (chain, ring, branched) so that
# independent MH runs start in different basins, maximizing the chance of
# discovering distinct minima within a limited step budget.

def seed_topologies(n, n_seeds, max_degree=3, seed=0):
    """
    Generate n_seeds qualitatively diverse connected graphs on n nodes,
    degree in [1,3], at least one node degree >= 2.

    Strategy: always include the simple chain (path graph) and the
    simple ring (cycle graph) as canonical seeds (when realizable), then
    fill remaining seed slots with random degree-sequence graphs for
    additional diversity (branched/mixed topologies).
    """
    rng = np.random.default_rng(seed)
    seeds = []

    # Canonical seed 1: linear chain
    chain = nx.path_graph(n)
    if max(dict(chain.degree()).values()) >= 2:
        seeds.append(chain)

    # Canonical seed 2: simple ring (needs n >= 3)
    if n >= 3:
        ring = nx.cycle_graph(n)
        seeds.append(ring)

    # Remaining seeds: random degree-sequence graphs for topological diversity
    attempts = 0
    while len(seeds) < n_seeds and attempts < n_seeds * 50:
        attempts += 1
        degs = rng.integers(1, max_degree + 1, size=n).tolist()
        if sum(degs) % 2 != 0:
            idx = degs.index(1) if 1 in degs else 0
            degs[idx] = 2
        if max(degs) < 2:
            degs[0] = 2
        if not nx.is_graphical(degs):
            continue
        try:
            G = nx.random_degree_sequence_graph(
                degs, seed=int(rng.integers(0, 2**31)), tries=5
            )
        except nx.NetworkXError:
            continue
        if not nx.is_connected(G):
            continue
        degs_actual = [G.degree(v) for v in G.nodes()]
        if max(degs_actual) > max_degree or max(degs_actual) < 2:
            continue
        # avoid exact duplicates of already-chosen seeds
        if any(nx.is_isomorphic(G, s) for s in seeds):
            continue
        seeds.append(G)

    return seeds[:n_seeds]


def embed_3d(G, bond_length=NN_BOND_INIT, seed=0):
    """
    Single force-field-free geometric embedding: spring layout rescaled
    to the target bond length. No energy-based relaxation here -- MACE
    (via Minima Hopping's own initial local optimization step) handles
    bringing this to a physically sensible starting geometry. Kept
    deliberately simple since Minima Hopping itself performs the first
    local optimization before any MD.
    """
    n = len(G.nodes())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pos = nx.spring_layout(G, dim=3, seed=seed, k=bond_length * 1.2)
    coords = np.array([pos[i] for i in range(n)], dtype=float)
    edges = list(G.edges())
    if edges:
        mean_d = np.mean([np.linalg.norm(coords[u] - coords[v]) for u, v in edges])
        if mean_d > 1e-8:
            coords *= bond_length / mean_d
    coords -= coords.mean(axis=0)
    return coords


# ─────────────────────────────────────────────────────────────────────────────
# STRUCTURAL INTEGRITY CHECK (reused from polynitrogen_mace.py)
# ─────────────────────────────────────────────────────────────────────────────

def infer_geometric_graph(coords, bond_cutoff=BOND_CUTOFF_NN):
    n = len(coords)
    G_geom = nx.Graph()
    G_geom.add_nodes_from(range(n))
    distances = {}
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(coords[i] - coords[j]))
            distances[(i, j)] = d
            if d <= bond_cutoff:
                G_geom.add_edge(i, j, length=d)
    return G_geom, distances


def check_structural_integrity(coords, G_target, bond_cutoff=BOND_CUTOFF_NN,
                                min_distance=MIN_SANE_NN_DISTANCE,
                                max_physical_degree=3):
    """
    Classify a relaxed/MD-evolved structure relative to its starting
    topology G_target. See polynitrogen_mace.py manual section for full
    derivation. Returns dict with 'status' in
    {OK, FRAGMENTED, REARRANGED, BOND_ANOMALY, COLLAPSED}.

    max_physical_degree guards against a failure mode NOT covered by the
    minimum-distance check alone: an MLIP extrapolating outside its
    training distribution can produce a geometrically "compact but not
    individually too-close" cluster where one atom sits within
    bond_cutoff of four or more neighbors simultaneously -- a degree no
    neutral, non-hypervalent nitrogen atom can physically sustain (max
    sp3 coordination = 3, see Property 2.1 in the methodology manual).
    Such a structure can still report individually plausible pairwise
    distances (e.g. all > MIN_SANE_NN_DISTANCE) while being chemically
    absurd and carrying a spuriously low MACE energy (the MLIP has
    likely left its training distribution). Default threshold of 3 is
    the true chemical maximum for neutral, non-hypervalent nitrogen;
    any geometric degree > 3 is treated as unambiguous collapse.
    """
    n = len(coords)
    G_geom, distances = infer_geometric_graph(coords, bond_cutoff=bond_cutoff)

    n_fragments = nx.number_connected_components(G_geom)
    is_connected = (n_fragments == 1)

    all_d = list(distances.values())
    min_any_distance = min(all_d) if all_d else float("inf")
    max_any_distance = max(all_d) if all_d else 0.0

    geometric_degrees = [G_geom.degree(v) for v in G_geom.nodes()] if n > 0 else [0]
    max_geometric_degree = max(geometric_degrees)
    hypervalent = max_geometric_degree > max_physical_degree

    target_bond_lengths = []
    for (u, v) in G_target.edges():
        i, j = (u, v) if u < v else (v, u)
        target_bond_lengths.append(distances[(i, j)])
    min_target_bond = min(target_bond_lengths) if target_bond_lengths else float("inf")
    max_target_bond = max(target_bond_lengths) if target_bond_lengths else 0.0
    bonds_in_range = all(min_distance <= d <= bond_cutoff for d in target_bond_lengths)

    topology_preserved = is_connected and nx.is_isomorphic(G_geom, G_target)
    collapsed = (min_any_distance < min_distance) or hypervalent

    if collapsed:
        status = "COLLAPSED"
    elif not is_connected:
        status = "FRAGMENTED"
    elif not topology_preserved:
        status = "REARRANGED"
    elif not bonds_in_range:
        status = "BOND_ANOMALY"
    else:
        status = "OK"

    return {
        "status": status,
        "is_connected": is_connected,
        "n_fragments": n_fragments,
        "topology_preserved": topology_preserved,
        "bonds_in_range": bonds_in_range,
        "min_target_bond": min_target_bond,
        "max_target_bond": max_target_bond,
        "min_any_distance": min_any_distance,
        "max_any_distance": max_any_distance,
        "max_geometric_degree": max_geometric_degree,
        "hypervalent": hypervalent,
        "final_graph": G_geom,
    }


def classify_final_topology(G_geom):
    """Lightweight topology label for the geometric graph of a minimum."""
    n = len(G_geom.nodes())
    if n == 0:
        return "empty"
    degs = sorted([G_geom.degree(v) for v in G_geom.nodes()], reverse=True)
    cycles = nx.cycle_basis(G_geom)
    n_rings = len(cycles)
    ring_sizes = sorted(len(c) for c in cycles)
    if not nx.is_connected(G_geom):
        comps = [len(c) for c in nx.connected_components(G_geom)]
        return f"fragmented-{'+'.join(map(str, sorted(comps)))}"
    if n_rings == 0:
        return "chain" if max(degs) <= 2 else "branched-tree"
    if n_rings == 1:
        return f"ring-{ring_sizes[0]}" if all(d == 2 for d in degs) else f"ring-{ring_sizes[0]}-subst"
    return f"polycyclic-{'-'.join(map(str, ring_sizes))}"


# ─────────────────────────────────────────────────────────────────────────────
# SYMMETRY (optional, reused pattern)
# ─────────────────────────────────────────────────────────────────────────────

def get_symmetry(atoms, symprec=0.1):
    if not SPGLIB_OK:
        return {"spacegroup": "N/A", "number": -1}
    a = atoms.copy()
    a.set_pbc(True)
    cell_data = (a.cell[:], a.get_scaled_positions(), a.get_atomic_numbers())
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sym = spglib.get_symmetry_dataset(cell_data, symprec=symprec)
        if sym is None:
            return {"spacegroup": "P1", "number": 1}
        try:
            return {"spacegroup": sym.international, "number": sym.number}
        except AttributeError:
            return {"spacegroup": sym["international"], "number": sym["number"]}
    except Exception:
        return {"spacegroup": "P1", "number": 1}


# ─────────────────────────────────────────────────────────────────────────────
# MACE CALCULATOR LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_mace_calculator(model_type="mace_off", model_size="small",
                          device="cpu", dtype="float64"):
    try:
        from mace.calculators import mace_off, mace_mp
    except ImportError:
        log.error("MACE not found. Install with: pip install mace-torch")
        return None

    # PyTorch's MPS backend (Apple Silicon GPU, e.g. M1-M5) does not
    # support float64 at all: a float64 tensor allocation either raises
    # a runtime error or is silently kept on CPU depending on the
    # PyTorch version, defeating the purpose of requesting --device mps.
    # Auto-correct to float32 with an explicit warning rather than
    # letting the user hit a cryptic downstream error or silently get
    # CPU-speed performance while believing MPS is active.
    if device == "mps" and dtype == "float64":
        log.warning(
            "device='mps' (Apple Silicon GPU) does not support float64. "
            "Automatically switching to dtype='float32' for this run. "
            "Energies/forces will differ from a float64/CPU run at the "
            "~1e-3 to 1e-4 eV level -- negligible for structure screening "
            "and well below the energy_tol used for minima deduplication, "
            "but worth knowing if comparing exact numbers across machines."
        )
        dtype = "float32"

    log.info(f"Loading MACE calculator: {model_type}/{model_size} on {device} ({dtype})")
    common_kw = dict(device=device, default_dtype=dtype)

    if model_type == "mace_off":
        return mace_off(model=model_size, **common_kw)
    elif model_type == "mace_mp":
        return mace_mp(model=model_size, dispersion=False, **common_kw)
    else:
        log.error(f"Unknown model_type: {model_type}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# MINIMA HOPPING RUN (single seed)
# ─────────────────────────────────────────────────────────────────────────────

class SeedTimeoutError(Exception):
    """Raised (in the parent process) when a single Minima Hopping seed
    exceeds MH_SEED_TIMEOUT and had to be forcibly killed."""
    pass


def _mh_seed_worker(G_seed_edges, n, model_type, model_size, device, dtype,
                     n_steps, seed_idx, output_subdir_str, T0, Ediff0,
                     fmax, mdmin, hookean_rt, hookean_k, max_temp,
                     rng_seed, allow_resume):
    """
    Worker function executed in a SEPARATE PROCESS (see
    run_minima_hopping_single_seed for why). Receives only plain,
    picklable arguments (no live MACE calculator object, no networkx
    Graph -- just its edge list) and reconstructs everything locally,
    since objects holding PyTorch state are not reliably shareable
    across the fork boundary in all environments.

    Minima are written incrementally to minima_traj_path by ASE's own
    MinimaHopping as the run progresses (this is ASE's native behavior,
    unrelated to our multiprocessing wrapper) -- so even if this worker
    is killed mid-run by the parent's timeout enforcement, whatever
    minima were found before the kill remain safely readable from disk
    by the parent afterward.
    """
    G_seed = nx.Graph()
    G_seed.add_nodes_from(range(n))
    G_seed.add_edges_from(G_seed_edges)

    calc = load_mace_calculator(model_type, model_size, device, dtype)

    output_subdir = Path(output_subdir_str)
    coords = embed_3d(G_seed, seed=rng_seed)
    atoms = Atoms("N" * n, positions=coords)
    atoms.calc = calc

    constraints = [
        Hookean(a1=int(u), a2=int(v), rt=hookean_rt, k=hookean_k)
        for u, v in G_seed.edges()
    ]
    atoms.set_constraint(constraints)

    minima_traj_path = output_subdir / f"minima_seed{seed_idx}.traj"
    log_path = output_subdir / f"hop_seed{seed_idx}.log"

    if not allow_resume:
        for stale_path in (minima_traj_path, log_path):
            if stale_path.exists():
                stale_path.unlink()

    opt = MinimaHopping(
        atoms, T0=T0, Ediff0=Ediff0, mdmin=mdmin, fmax=fmax,
        minima_traj=str(minima_traj_path), logfile=str(log_path),
    )
    try:
        opt(totalsteps=n_steps, maxtemp=max_temp)
    except Exception:
        pass  # any failure here just means fewer minima get recorded;
              # the parent process reads back whatever IS on disk regardless


def run_minima_hopping_single_seed(G_seed, n, model_type, model_size,
                                    device, dtype, n_steps, seed_idx,
                                    output_subdir, T0=MH_T0, Ediff0=MH_EDIFF0,
                                    fmax=MH_FMAX, mdmin=MH_MDMIN,
                                    hookean_rt=HOOKEAN_RT, hookean_k=HOOKEAN_K,
                                    max_temp=MH_MAX_TEMP, rng_seed=0,
                                    allow_resume=False,
                                    timeout_seconds=MH_SEED_TIMEOUT):
    """
    Run one independent Minima Hopping trajectory starting from one seed
    topology, in a SEPARATE OS PROCESS with a hard wall-clock timeout
    enforced by the parent. Returns list of dicts, one per DISTINCT
    minimum found (after geometric deduplication), each with energy,
    geometry, and structural integrity diagnosis.

    WHY A SEPARATE PROCESS RATHER THAN A SIGNAL-BASED TIMEOUT
    ---------------------------------------------------------------
    An earlier implementation used signal.alarm() (SIGALRM) to interrupt
    a stuck seed from within the same process. This was found, by direct
    testing, to be UNRELIABLE once MACE/PyTorch calls are involved: a
    tight loop of repeated calculator.get_potential_energy() calls can
    prevent the Python signal handler from ever being invoked, even
    after the alarm has long since fired (PyTorch's C/native code
    appears to hold the GIL, or otherwise defer signal delivery, across
    these calls in a way that defeats SIGALRM). This was verified
    directly: a deliberately infinite MACE-calling loop with a 3 s
    SIGALRM timeout ran for 50+ calls and 2+ seconds past the deadline
    with no exception ever raised.

    multiprocessing.Process, by contrast, can always be forcibly
    terminated by the PARENT process via SIGTERM/SIGKILL at the OS
    level, completely independent of whatever the CHILD process's
    Python interpreter is doing internally. This was verified to
    reliably kill a stuck MACE-calling subprocess within ~0.1 s of the
    parent's join(timeout=...) expiring. This is the only mechanism
    found to give a REAL guarantee against indefinite hangs.

    allow_resume : bool
        ASE's MinimaHopping silently RESUMES from an existing
        minima_seed{idx}.traj file if one is already present at the
        target path. This is a deliberate ASE feature for legitimately
        continuing an interrupted long campaign, but it has a sharp
        edge discovered in production use: if a PREVIOUS run (e.g. one
        that hit a now-fixed bug, or used different settings) left a
        .traj file at the same output_dir, a later run will silently
        inherit that file's internal state -- including whatever
        pathologically high temperature an old run may have reached --
        rather than starting the fresh trajectory the user intended.
        Default False: any pre-existing minima_seed{idx}.traj /
        hop_seed{idx}.log found is deleted before starting (inside the
        worker process). Set True only to deliberately resume a
        previous, intentionally-paused campaign with IDENTICAL settings.
    """
    output_subdir.mkdir(parents=True, exist_ok=True)
    minima_traj_path = output_subdir / f"minima_seed{seed_idx}.traj"
    log_path = output_subdir / f"hop_seed{seed_idx}.log"

    if not allow_resume and (minima_traj_path.exists() or log_path.exists()):
        log.warning(
            f"    Found pre-existing file(s) for seed {seed_idx} from a "
            f"previous run at {output_subdir}. They will be deleted "
            f"inside the worker process to guarantee a FRESH trajectory "
            f"with the current settings (T0={T0}, max_temp={max_temp}), "
            "rather than silently resuming a possibly stale/buggy prior "
            "run. Pass allow_resume=True to intentionally resume instead."
        )

    # Use 'spawn' rather than 'fork' for the worker process, regardless
    # of platform. This was changed after a real production failure on
    # macOS: 'fork' duplicates the ENTIRE parent process memory image,
    # including any PyTorch/MACE internal state, thread pools, or BLAS
    # locks already initialized in the parent at the time of forking.
    # On macOS in particular (where Python's own default start method
    # was changed away from 'fork' specifically because of this class of
    # instability with multi-threaded libraries), this caused every
    # single worker to silently fail at startup -- the subprocess
    # appeared to run (no exception surfaced to the parent) but
    # terminated near-instantly without ever reaching the MD loop,
    # observed directly as "0 minima recorded" across 30 consecutive
    # seeds in under 4 seconds total (an impossible wall-clock budget
    # for genuine MACE relaxation work). 'spawn' instead starts a fresh,
    # clean Python interpreter for the worker with none of the parent's
    # potentially-corrupted in-process state, at the cost of a slightly
    # higher per-worker startup latency (re-importing modules) -- a
    # trade firmly worth making for correctness. 'spawn' is available
    # and behaves identically on Linux, macOS, and Windows.
    ctx = mp.get_context("spawn")
    process = ctx.Process(
        target=_mh_seed_worker,
        args=(list(G_seed.edges()), n, model_type, model_size, device, dtype,
              n_steps, seed_idx, str(output_subdir), T0, Ediff0,
              fmax, mdmin, hookean_rt, hookean_k, max_temp,
              rng_seed, allow_resume),
    )
    process.start()
    process.join(timeout=timeout_seconds)

    if process.is_alive():
        log.warning(
            f"    MH seed {seed_idx} HARD-TIMED-OUT after {timeout_seconds} s "
            "and was forcibly terminated. This typically means ASE's "
            "internal MD substep loop never satisfied its minimum-passing "
            "criterion within the time budget (a known ASE limitation "
            "with no built-in upper bound -- see MH_SEED_TIMEOUT "
            "docstring). Any minima already written to disk before the "
            "kill are still read and kept below. Consider a different "
            "--seed for this size if this recurs frequently, or increase "
            "--seed_timeout if your hardware is simply slower than the "
            "default assumes."
        )
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join()

    # Read back all minima found in this run (whether the worker
    # completed normally or was killed mid-run -- ASE writes minima to
    # disk incrementally, so partial results survive a forced kill).
    # Re-attach a calculator in THIS process to recompute energies,
    # since a calculator object cannot reliably cross the fork boundary
    # back from a terminated child.
    results = []
    if not minima_traj_path.exists():
        return results

    try:
        found = read(minima_traj_path, index=":")
    except Exception as exc:
        log.warning(f"    Could not read {minima_traj_path}: {exc}")
        return results

    calc = load_mace_calculator(model_type, model_size, device, dtype)
    for m_idx, m_atoms in enumerate(found):
        m_atoms.calc = calc
        try:
            energy = m_atoms.get_potential_energy()
        except Exception:
            continue
        integrity = check_structural_integrity(m_atoms.get_positions(), G_seed)
        results.append({
            "atoms": m_atoms,
            "energy_eV": energy,
            "seed_idx": seed_idx,
            "minimum_idx_in_run": m_idx,
            "integrity": integrity,
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# DEDUPLICATION ACROSS ALL RUNS (for a fixed n)
# ─────────────────────────────────────────────────────────────────────────────

def deduplicate_minima(all_minima, energy_tol=1e-3):
    """
    Merge minima that are the SAME physical structure found multiple
    times (e.g. by different seeds, or by atom-relabeling symmetry, as
    observed empirically on N4 -- see manual). Two minima are considered
    duplicates if their final geometric graphs are isomorphic AND their
    energies agree within energy_tol (eV).

    Returns a deduplicated list, keeping the lowest-energy representative
    of each duplicate group.
    """
    unique = []
    for m in all_minima:
        g_final = m["integrity"]["final_graph"]
        is_dup = False
        for u in unique:
            g_other = u["integrity"]["final_graph"]
            if abs(m["energy_eV"] - u["energy_eV"]) < energy_tol and \
               nx.is_isomorphic(g_final, g_other):
                is_dup = True
                if m["energy_eV"] < u["energy_eV"]:
                    u.update(m)
                break
        if not is_dup:
            unique.append(m)
    return unique


# ─────────────────────────────────────────────────────────────────────────────
# N2 REFERENCE ENERGY
# ─────────────────────────────────────────────────────────────────────────────

def compute_n2_reference(calc, fmax=0.01):
    """Relax an isolated N2 molecule and return E(N2)/2, eV/atom."""
    atoms = Atoms("N2", positions=[[0, 0, 0], [0, 0, 1.098]])
    atoms.calc = calc
    opt = LBFGS(atoms, logfile=None)
    opt.run(fmax=fmax, steps=200)
    e_total = atoms.get_potential_energy()
    return e_total / 2.0


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT WRITERS
# ─────────────────────────────────────────────────────────────────────────────

def write_xyz(atoms, filepath, comment=""):
    n = len(atoms)
    with open(filepath, "w") as f:
        f.write(f"{n}\n{comment}\n")
        for sym, pos in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
            f.write(f"{sym}  {pos[0]:.8f}  {pos[1]:.8f}  {pos[2]:.8f}\n")


def write_cif(atoms, filepath, cell_size=30.0):
    import tempfile, os
    a = atoms.copy()
    a.set_cell([cell_size] * 3)
    a.center()
    a.set_pbc(True)
    with tempfile.NamedTemporaryFile(suffix=".cif", delete=False, mode="w") as tmp:
        tmp_path = tmp.name
    write(tmp_path, a, format="cif")
    os.replace(tmp_path, filepath)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(n_min=4, n_max=12, n_seeds=4, n_steps=30,
                  model_type="mace_off", model_size="small",
                  device="cpu", dtype="float64",
                  output_dir="polyN_MH_results", seed=42,
                  max_temp=MH_MAX_TEMP, seed_timeout=MH_SEED_TIMEOUT,
                  n_workers=1):

    if n_min % 2 != 0:
        log.warning(f"n_min={n_min} is odd; bumping to {n_min + 1} "
                    "(even n required for closed-shell singlet, see manual).")
        n_min += 1
    if n_max % 2 != 0:
        n_max -= 1

    out = Path(output_dir)
    pre_existing_traj_files = list(out.glob("mh_runs/**/*.traj")) if out.exists() else []
    if pre_existing_traj_files:
        log.warning(
            f"output_dir '{output_dir}' already contains "
            f"{len(pre_existing_traj_files)} .traj file(s) from a previous "
            "run. Each individual seed will detect and delete its own "
            "stale file before starting fresh (see run_minima_hopping_"
            "single_seed docstring) -- this is safe, but if you intended "
            "to keep the previous run's results untouched, stop now and "
            "choose a different --output_dir."
        )

    dirs = {
        "xyz": out / "xyz",
        "cif": out / "cif",
        "runs": out / "mh_runs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    calc = load_mace_calculator(model_type, model_size, device, dtype)
    if calc is None:
        sys.exit(1)

    log.info("Computing N2 reference energy...")
    e_n2_per_atom = compute_n2_reference(calc)
    log.info(f"  E(N2)/atom = {e_n2_per_atom:.6f} eV")

    csv_path = out / "minima_results.csv"
    fieldnames = [
        "id", "n_atoms", "seed_idx", "minimum_idx_in_run",
        "final_topology_label", "energy_eV", "energy_eV_per_atom",
        "delta_E_eV_per_atom_vs_N2",
        "integrity_status", "n_fragments", "topology_preserved",
        "min_any_distance_A", "max_any_distance_A",
        "spacegroup", "xyz_file", "cif_file",
    ]
    csv_file = open(csv_path, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    mol_id = 0
    hull_data = []  # (n, delta_E_per_atom) for every accepted minimum, for convex-hull use

    for n in range(n_min, n_max + 1, 2):
        log.info("=" * 60)
        log.info(f"N{n}: generating {n_seeds} seed topologies...")
        seeds = seed_topologies(n, n_seeds, seed=seed + n)
        log.info(f"N{n}: {len(seeds)} seeds (chain/ring/random mix)")

        n_subdir = dirs["runs"] / f"N{n}"
        n_subdir.mkdir(parents=True, exist_ok=True)

        all_minima_this_n = []

        if n_workers <= 1:
            # Sequential path (default, n_workers=1): identical behavior
            # to all previously validated runs in this project. Kept as
            # a separate code path (rather than always going through the
            # pool with max_workers=1) so the well-tested sequential
            # logic remains untouched and trivially comparable.
            for s_idx, G_seed in enumerate(seeds):
                log.info(f"  seed {s_idx+1}/{len(seeds)}: running Minima Hopping "
                          f"({n_steps} steps, capped at {max_temp:.0f} K, "
                          f"hard timeout {seed_timeout}s)...")
                results = run_minima_hopping_single_seed(
                    G_seed, n, model_type, model_size, device, dtype,
                    n_steps, s_idx, n_subdir,
                    max_temp=max_temp, timeout_seconds=seed_timeout,
                    rng_seed=seed + n * 100 + s_idx,
                )
                log.info(f"    -> {len(results)} minima recorded in this run")
                all_minima_this_n.extend(results)
        else:
            # Parallel path: launch up to n_workers seeds concurrently
            # via a ProcessPoolExecutor.
            #
            # IMPORTANT -- two independent layers of multiprocessing are
            # at play here, and it is essential to understand both before
            # choosing --n_workers on a shared/multi-user machine:
            #   (1) THIS pool: n_workers parent-level processes, each
            #       running run_minima_hopping_single_seed() for one
            #       seed concurrently.
            #   (2) INSIDE each of those, run_minima_hopping_single_seed()
            #       itself spawns ONE MORE child process (see that
            #       function's docstring) purely to enforce the hard
            #       --seed_timeout via a killable subprocess (the
            #       SIGALRM-based approach was found unreliable with
            #       PyTorch/MACE -- see the methodology manual).
            # At any instant, this can therefore launch up to roughly
            # 2 x n_workers OS processes (n_workers "pool" processes,
            # each having spawned its own short-lived timeout-enforcement
            # child). Set --n_workers conservatively relative to the
            # number of physical cores available (e.g. n_workers <=
            # physical_cores / 2 as a starting point), and lower it
            # further if running on a shared cluster node where other
            # jobs may be competing for the same cores.
            pool_ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=pool_ctx) as executor:
                futures = {
                    executor.submit(
                        run_minima_hopping_single_seed,
                        G_seed, n, model_type, model_size, device, dtype,
                        n_steps, s_idx, n_subdir,
                        max_temp, seed_timeout, seed + n * 100 + s_idx,
                    ): s_idx
                    for s_idx, G_seed in enumerate(seeds)
                }
                log.info(f"  Dispatched {len(futures)} seeds to a pool of "
                          f"{n_workers} workers...")
                for future in as_completed(futures):
                    s_idx = futures[future]
                    try:
                        results = future.result()
                    except Exception as exc:
                        log.warning(f"    seed {s_idx}: pool worker raised "
                                  f"{type(exc).__name__}: {exc}")
                        results = []
                    log.info(f"  seed {s_idx+1}/{len(seeds)} finished -> "
                              f"{len(results)} minima recorded")
                    all_minima_this_n.extend(results)

        unique_minima = deduplicate_minima(all_minima_this_n)
        log.info(f"N{n}: {len(all_minima_this_n)} total minima found, "
                  f"{len(unique_minima)} unique after deduplication")

        # sort by energy ascending
        unique_minima.sort(key=lambda m: m["energy_eV"])

        for rank, m in enumerate(unique_minima):
            mol_id += 1
            integ = m["integrity"]
            label = classify_final_topology(integ["final_graph"])
            e_per_atom = m["energy_eV"] / n
            delta_e = e_per_atom - e_n2_per_atom

            # Independent energy sanity override: even a geometrically
            # "OK" structure (no detected hypervalence/collapse/fragmentation)
            # can carry a spuriously low MACE energy if the model
            # extrapolates poorly for a borderline-unusual but not
            # outright pathological geometry. This check is therefore
            # applied regardless of the geometric integrity status, and
            # takes precedence for hull-inclusion purposes. Symmetric
            # upper bound added after a real +31 eV/atom artifact was
            # observed in production (see MAX_SANE_DELTA_E_PER_ATOM
            # docstring for the full account).
            if delta_e < MIN_SANE_DELTA_E_PER_ATOM or delta_e > MAX_SANE_DELTA_E_PER_ATOM:
                integ["status"] = "ENERGY_ANOMALY"

            if integ["status"] in ("OK", "REARRANGED", "BOND_ANOMALY"):
                hull_data.append((n, delta_e))

            sym_info = get_symmetry(m["atoms"])
            stem = f"N{n}_rank{rank:03d}_{label}_seed{m['seed_idx']}"

            comment = (
                f"formula=N{n} topology={label} integrity={integ['status']} "
                f"energy_eV={m['energy_eV']:.6f} delta_E_eV_atom={delta_e:.4f} "
                f"sg={sym_info['spacegroup']}"
            )
            xyz_path = dirs["xyz"] / f"{stem}.xyz"
            cif_path = dirs["cif"] / f"{stem}.cif"
            write_xyz(m["atoms"], xyz_path, comment=comment)
            write_cif(m["atoms"], cif_path)

            writer.writerow({
                "id": mol_id,
                "n_atoms": n,
                "seed_idx": m["seed_idx"],
                "minimum_idx_in_run": m["minimum_idx_in_run"],
                "final_topology_label": label,
                "energy_eV": f"{m['energy_eV']:.6f}",
                "energy_eV_per_atom": f"{e_per_atom:.6f}",
                "delta_E_eV_per_atom_vs_N2": f"{delta_e:.4f}",
                "integrity_status": integ["status"],
                "n_fragments": integ["n_fragments"],
                "topology_preserved": integ["topology_preserved"],
                "min_any_distance_A": f"{integ['min_any_distance']:.4f}",
                "max_any_distance_A": f"{integ['max_any_distance']:.4f}",
                "spacegroup": sym_info["spacegroup"],
                "xyz_file": str(xyz_path.relative_to(out)),
                "cif_file": str(cif_path.relative_to(out)),
            })
        csv_file.flush()

    csv_file.close()

    # ── Convex hull summary (simple lower-envelope in delta_E vs n) ───────
    hull_path = out / "convex_hull_summary.csv"
    with open(hull_path, "w", newline="") as f:
        hw = csv.writer(f)
        hw.writerow(["n_atoms", "min_delta_E_eV_per_atom_found"])
        by_n = {}
        for n, de in hull_data:
            by_n.setdefault(n, []).append(de)
        for n in sorted(by_n):
            hw.writerow([n, f"{min(by_n[n]):.4f}"])

    log.info("=" * 60)
    log.info(f"DONE. Results in '{output_dir}/'")
    log.info(f"  minima_results.csv      : {csv_path}")
    log.info(f"  convex_hull_summary.csv : {hull_path}")
    return csv_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--n_min", type=int, default=4,
                   help="Minimum N count (rounded up to even). Default 4.")
    p.add_argument("--n_max", type=int, default=12,
                   help="Maximum N count (rounded down to even). Default 12.")
    p.add_argument("--n_seeds", type=int, default=4,
                   help="Independent MH runs (diverse starting topologies) per n. Default 4.")
    p.add_argument("--n_steps", type=int, default=30,
                   help="Minima Hopping steps per run. Default 30.")
    p.add_argument("--max_temp", type=float, default=MH_MAX_TEMP,
                   help=f"Maximum MD temperature (K) before a seed's trajectory "
                        f"is force-terminated, to prevent runaway adaptive "
                        f"heating from consuming disproportionate wall-clock "
                        f"time on a single stuck seed (see module constants "
                        f"for the full explanation). Default {MH_MAX_TEMP:.0f}.")
    p.add_argument("--seed_timeout", type=int, default=MH_SEED_TIMEOUT,
                   help=f"Hard wall-clock timeout (seconds) per seed, "
                        f"enforced by running each seed in a separate OS "
                        f"process that the parent forcibly terminates if "
                        f"this limit is exceeded. Protects against ASE's "
                        f"internal MD substep loop having no upper bound, "
                        f"which can cause a seed to hang indefinitely "
                        f"independent of --max_temp (see MH_SEED_TIMEOUT "
                        f"docstring for the full mechanism discovered in "
                        f"production use). Each worker reloads its own "
                        f"MACE model (a few seconds, via 'spawn' for "
                        f"cross-platform reliability -- see ctx = "
                        f"mp.get_context comment), so avoid setting this "
                        f"below ~15-20s or workers may be killed before "
                        f"they finish loading. Default {MH_SEED_TIMEOUT}.")
    p.add_argument("--n_workers", type=int, default=1,
                   help="Number of seeds to run CONCURRENTLY per size, via "
                        "a process pool (default 1 = fully sequential, "
                        "identical behavior to all previously validated "
                        "runs). Each worker may itself spawn one additional "
                        "short-lived child process for hard-timeout "
                        "enforcement, so the actual peak OS process count "
                        "can reach roughly 2x this value -- set "
                        "conservatively relative to available physical "
                        "cores (e.g. n_workers <= physical_cores/2 as a "
                        "starting point), and lower further on a shared "
                        "cluster node. See module docstring for the full "
                        "rationale and a worked example for a 10-core "
                        "Apple M5.")
    p.add_argument("--model", type=str, default="mace_off",
                   choices=["mace_off", "mace_mp"])
    p.add_argument("--model_size", type=str, default="small",
                   help="MACE model size: small/medium/large. Default small "
                        "(fast, appropriate for the large number of MD/relax "
                        "calls in Minima Hopping).")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--dtype", type=str, default="float64",
                   choices=["float32", "float64"])
    p.add_argument("--output_dir", type=str, default="polyN_MH_results")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    log.info("polynitrogen_minimahopping.py -- Global PES exploration of Nn")
    log.info(f"  N range  : N{args.n_min} to N{args.n_max} (even only)")
    log.info(f"  seeds/run: {args.n_seeds}, steps/run: {args.n_steps}, "
             f"max_temp: {args.max_temp:.0f} K, seed_timeout: {args.seed_timeout}s")
    log.info(f"  parallel : n_workers={args.n_workers} "
             f"({'sequential' if args.n_workers <= 1 else 'concurrent pool'})")
    log.info(f"  MACE     : {args.model}/{args.model_size} on {args.device}")
    log.info(f"  output   : {args.output_dir}/")

    run_pipeline(
        n_min=args.n_min, n_max=args.n_max,
        n_seeds=args.n_seeds, n_steps=args.n_steps,
        model_type=args.model, model_size=args.model_size,
        device=args.device, dtype=args.dtype,
        output_dir=args.output_dir, seed=args.seed,
        max_temp=args.max_temp, seed_timeout=args.seed_timeout,
        n_workers=args.n_workers,
    )
