#!/usr/bin/env python3
"""
polynitrogen_airss.py
======================
AIRSS-style ("ab initio random structure searching", Pickard & Needs,
J. Phys.: Condens. Matter 23, 053201, 2011) random generation of neutral
all-nitrogen (Nn) molecules, with VARIABLE n ("variable composition"
search in AIRSS terminology), followed by MACE relaxation and the same
structural-integrity / metastability bookkeeping used in the companion
scripts.

RELATION TO THE OTHER TWO SCRIPTS IN THIS PROJECT
----------------------------------------------------
  polynitrogen_mace.py            : combinatorial graph enumeration,
                                     one relaxation per topology.
                                     Strength: systematic coverage of
                                     small-n topology space.
  polynitrogen_minimahopping.py   : global PES exploration via Minima
                                     Hopping from a few diverse seeds per
                                     fixed n. Strength: basin-hopping
                                     depth at a GIVEN size.
  polynitrogen_airss.py (this)    : large-N stochastic sampling with n
                                     ITSELF drawn randomly each trial
                                     (AIRSS "variable composition" idea,
                                     Pickard & Needs Section on variable
                                     stoichiometry searching), each trial
                                     independent and embarrassingly
                                     parallel. Strength: breadth across
                                     MANY sizes simultaneously, with no
                                     topological bias from a starting
                                     graph at all (purely geometric,
                                     MINSEP-constrained random placement)
                                     -- directly suited to building the
                                     n-vs-delta_E "hull" view requested.

THE GENERATION ALGORITHM (buildcell-equivalent for an isolated molecule)
---------------------------------------------------------------------------
AIRSS's buildcell code (Pickard & Needs) builds sensible random PERIODIC
structures by placing atoms subject to a minimum pairwise separation
(MINSEP) at a target density/volume-per-atom. There is no periodic cell
here -- Nn is an isolated molecule -- so "density" has no direct meaning.
Instead, this script uses a sequential anchor-growth procedure that is
the natural molecular analogue:
  1. Place the first atom at the origin.
  2. To place each subsequent atom, pick a uniformly random ALREADY-PLACED
     atom as an anchor, and propose a new position at a random distance
     in [MINSEP, BOND_GUESS_MAX] from it, in a random direction.
  3. Reject the proposal if it would violate MINSEP against any already-
     placed atom, OR if it would push the candidate's own coordination
     number above MAX_DEGREE, OR if it would push any EXISTING atom's
     coordination number above MAX_DEGREE (checked explicitly for every
     atom that would become a new neighbor -- see manual for why a
     candidate-only check is insufficient, discovered empirically).
  4. Repeat until n atoms are placed or max_attempts is exceeded.
This keeps the growing cluster connected and chemically admissible
(degree <= 3, Property 2.1 of the companion methodology manual) purely
through geometric construction, with NO reference to any pre-built
connectivity graph -- in contrast to polynitrogen_mace.py and
polynitrogen_minimahopping.py, which both start from an explicit graph.
This is the key qualitative difference: graph-free, fully random
sampling, matching the spirit of AIRSS's "we choose an ensemble of
random starting structures" philosophy.

VARIABLE COMPOSITION (variable n)
-------------------------------------
Each trial independently draws n ~ Uniform{n_min, n_min+2, ..., n_max}
(even values only, per the spin/parity argument in the companion
manual), then generates and relaxes one random structure at that n. This
directly produces a population of (n, energy) pairs spanning the whole
requested size range in a single, fully parallel-friendly campaign --
exactly AIRSS's "user can specify an arbitrarily complex composition
space over which AIRSS can sample" idea (Pickard & Needs), adapted to
the single-element Nn case where the "composition space" is simply the
choice of n.

EXPECTED HIGH FAILURE RATE -- READ BEFORE INTERPRETING RESULTS
-------------------------------------------------------------------
Unlike polynitrogen_mace.py and polynitrogen_minimahopping.py, this
script imposes NO topological bias whatsoever: candidates are pure
geometric random placements (MINSEP + degree-constrained only), with no
underlying connectivity graph guiding the search toward a known-sensible
topology. Empirically (see companion manual section for this script),
the majority of such unbiased random starting geometries are found NOT
to be local minima of any cohesive n-atom Nn molecule at all -- LBFGS
relaxation under MACE instead drives them toward dissociated states
(most commonly n/2 x N2, sometimes other fragment combinations), which
are correctly flagged FRAGMENTED by the post-relaxation integrity check.
This is the expected, physically correct behavior of unbiased sampling
applied to a system whose global minimum (N2) is reached by breaking
essentially any other connectivity apart -- not a bug in the generator
or relaxation code. A low OK-fraction is itself the scientific message
of this script: it quantifies how "easy" or "hard" it is to land in a
genuinely cohesive Nn basin by chance alone, complementing the
topology-biased searches of the other two scripts. Expect to need
substantially more trials than seeds/steps in the other scripts to
accumulate a comparable number of usable (OK-status) candidates.

INSTALLATION
-------------
    pip install mace-torch ase networkx numpy scipy spglib --break-system-packages

USAGE
------
    # Quick test: 50 trials, n in {4,...,12}
    python polynitrogen_airss.py --n_min 4 --n_max 12 --n_trials 50

    # Production campaign: 2000 trials, n in {4,...,20}, GPU
    python polynitrogen_airss.py --n_min 4 --n_max 20 --n_trials 2000 \\
        --device cuda --output_dir ./airss_N4_N20

AUTHOR
-------
    Generated for IC2MP/E4 Mediacat -- Universite de Poitiers
"""

import argparse
import csv
import logging
import os
import sys
import tempfile
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import multiprocessing as mp
import networkx as nx
import numpy as np

try:
    from ase import Atoms
    from ase.io import write
    from ase.optimize import LBFGS
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

# Generation parameters (AIRSS buildcell equivalent, see module docstring)
MINSEP = 1.05               # Angstrom, minimum allowed N-N distance at generation
                             # (chosen just below the shortest plausible bond,
                             # the N#N triple bond at ~1.10 A, so legitimate
                             # short triple-bond-like contacts are not rejected
                             # outright, while true atomic overlap is excluded)
BOND_GUESS_MAX = 1.65       # Angstrom, max distance when proposing a new
                             # neighbor position (also used as the bond-
                             # perception cutoff during generation-time degree
                             # checking; matches BOND_CUTOFF_NN of the
                             # companion scripts)
MAX_DEGREE = 3              # chemical maximum for neutral N (Property 2.1)
MAX_GEN_ATTEMPTS = 20000    # per-trial cap on placement attempts

# Structural integrity thresholds (shared convention across all three scripts)
BOND_CUTOFF_NN = 1.60
MIN_SANE_NN_DISTANCE = 0.90
MIN_SANE_DELTA_E_PER_ATOM = -0.5   # eV/atom, see companion manual

# Metastability screening threshold for inclusion in the summary "hull"
# (loosely informed by the LEGO-xtal precedent of 0.5-1.0 eV/atom for an
# analogous covalent-allotrope screening problem; kept generous here
# since this script is meant for broad first-pass screening, not final
# candidate selection)
DELTA_E_SCREEN_THRESHOLD = 3.0     # eV/atom


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("polyN-AIRSS")


# ─────────────────────────────────────────────────────────────────────────────
# RANDOM SENSIBLE STRUCTURE GENERATION (buildcell-equivalent, molecular)
# ─────────────────────────────────────────────────────────────────────────────

def generate_random_sensible_molecule(n_atoms, minsep=MINSEP,
                                       bond_guess_max=BOND_GUESS_MAX,
                                       max_degree=MAX_DEGREE,
                                       max_attempts=MAX_GEN_ATTEMPTS, rng=None):
    """
    Generate one random, chemically-sensible Nn geometry via sequential
    anchor-growth, enforcing MINSEP and MAX_DEGREE throughout (see module
    docstring for the full algorithm and its AIRSS lineage).

    Returns (coords, success, n_attempts). If success is False, the
    constraints could not be jointly satisfied within max_attempts and
    the partial coords array (length < n_atoms) should be discarded.
    """
    if rng is None:
        rng = np.random.default_rng()

    coords = np.zeros((n_atoms, 3))
    coords[0] = [0.0, 0.0, 0.0]
    placed = 1
    attempts = 0

    while placed < n_atoms and attempts < max_attempts:
        attempts += 1

        anchor_idx = rng.integers(0, placed)
        anchor = coords[anchor_idx]
        direction = rng.normal(size=3)
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            continue
        direction /= norm
        dist = rng.uniform(minsep, bond_guess_max)
        candidate = anchor + direction * dist

        all_dists = np.linalg.norm(coords[:placed] - candidate, axis=1)

        # MINSEP: hard rejection of any overlap
        if np.any(all_dists < minsep):
            continue

        # Candidate's own coordination number must be in [1, max_degree]
        # (>=1 ensures it stays connected to the growing molecule; the
        # upper bound enforces Property 2.1 of the companion manual)
        candidate_degree = int(np.sum(all_dists < bond_guess_max))
        if candidate_degree > max_degree or candidate_degree == 0:
            continue

        # Critically, also verify that accepting this candidate would not
        # push any EXISTING atom's degree above max_degree. A candidate-
        # only check is insufficient: discovered empirically that without
        # this loop, final degree sequences could reach 6 even when each
        # individual placement appeared locally valid (see manual).
        would_be_neighbor = all_dists < bond_guess_max
        ok = True
        for j in range(placed):
            if would_be_neighbor[j]:
                dj = np.linalg.norm(coords[:placed] - coords[j], axis=1)
                dj[j] = np.inf
                existing_degree_j = int(np.sum(dj < bond_guess_max))
                if existing_degree_j + 1 > max_degree:
                    ok = False
                    break
        if not ok:
            continue

        coords[placed] = candidate
        placed += 1

    success = (placed == n_atoms)
    coords -= coords[:placed].mean(axis=0) if placed > 0 else 0
    return coords[:placed], success, attempts


# ─────────────────────────────────────────────────────────────────────────────
# STRUCTURAL INTEGRITY CHECK (shared convention, simplified: no "target"
# topology here since AIRSS generation has no pre-built graph to compare
# against -- only post-relaxation geometric sanity is checked)
# ─────────────────────────────────────────────────────────────────────────────

def infer_geometric_graph(coords, bond_cutoff=BOND_CUTOFF_NN):
    n = len(coords)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    distances = {}
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(coords[i] - coords[j]))
            distances[(i, j)] = d
            if d <= bond_cutoff:
                G.add_edge(i, j)
    return G, distances


def check_post_relaxation_sanity(coords, bond_cutoff=BOND_CUTOFF_NN,
                                  min_distance=MIN_SANE_NN_DISTANCE,
                                  max_physical_degree=MAX_DEGREE):
    """
    Post-relaxation geometric sanity check, adapted from the integrity
    classifiers of the companion scripts but WITHOUT a target-topology
    comparison (since AIRSS trials have no a priori intended graph --
    the relaxed structure IS the result, whatever connectivity it ends
    up with). Classifies into:
        OK          : single connected component, all distances and
                      degrees physically sane
        FRAGMENTED  : split into >=2 disconnected pieces during relaxation
        COLLAPSED   : a pairwise distance below min_distance, or an atom
                      with geometric degree > max_physical_degree
    """
    n = len(coords)
    G, distances = infer_geometric_graph(coords, bond_cutoff=bond_cutoff)
    n_fragments = nx.number_connected_components(G)
    is_connected = (n_fragments == 1)

    all_d = list(distances.values())
    min_any_distance = min(all_d) if all_d else float("inf")
    max_any_distance = max(all_d) if all_d else 0.0

    degrees = [G.degree(v) for v in G.nodes()] if n > 0 else [0]
    max_geometric_degree = max(degrees)
    hypervalent = max_geometric_degree > max_physical_degree
    collapsed = (min_any_distance < min_distance) or hypervalent

    if collapsed:
        status = "COLLAPSED"
    elif not is_connected:
        status = "FRAGMENTED"
    else:
        status = "OK"

    return {
        "status": status,
        "is_connected": is_connected,
        "n_fragments": n_fragments,
        "min_any_distance": min_any_distance,
        "max_any_distance": max_any_distance,
        "max_geometric_degree": max_geometric_degree,
        "final_graph": G,
    }


def classify_topology(G):
    n = len(G.nodes())
    if n == 0:
        return "empty"
    degs = sorted([G.degree(v) for v in G.nodes()], reverse=True)
    if not nx.is_connected(G):
        comps = sorted(len(c) for c in nx.connected_components(G))
        return f"fragmented-{'+'.join(map(str, comps))}"
    cycles = nx.cycle_basis(G)
    n_rings = len(cycles)
    ring_sizes = sorted(len(c) for c in cycles)
    if n_rings == 0:
        return "chain" if max(degs) <= 2 else "branched-tree"
    if n_rings == 1:
        return f"ring-{ring_sizes[0]}" if all(d == 2 for d in degs) else f"ring-{ring_sizes[0]}-subst"
    return f"polycyclic-{'-'.join(map(str, ring_sizes))}"


# ─────────────────────────────────────────────────────────────────────────────
# SYMMETRY
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
# MACE CALCULATOR
# ─────────────────────────────────────────────────────────────────────────────

def load_mace_calculator(model_type="mace_off", model_size="small",
                          device="cpu", dtype="float64"):
    try:
        from mace.calculators import mace_off, mace_mp
    except ImportError:
        log.error("MACE not found. Install with: pip install mace-torch")
        return None

    # MPS (Apple Silicon GPU) has no float64 support; auto-correct rather
    # than fail downstream or silently fall back to CPU speed.
    if device == "mps" and dtype == "float64":
        log.warning(
            "device='mps' does not support float64 -- switching to float32 "
            "automatically. See companion manual for the precision caveat."
        )
        dtype = "float32"

    log.info(f"Loading MACE calculator: {model_type}/{model_size} on {device} ({dtype})")
    common_kw = dict(device=device, default_dtype=dtype)
    if model_type == "mace_off":
        return mace_off(model=model_size, **common_kw)
    elif model_type == "mace_mp":
        return mace_mp(model=model_size, dispersion=False, **common_kw)
    log.error(f"Unknown model_type: {model_type}")
    return None


def compute_n2_reference(calc, fmax=0.01):
    atoms = Atoms("N2", positions=[[0, 0, 0], [0, 0, 1.098]])
    atoms.calc = calc
    opt = LBFGS(atoms, logfile=None)
    opt.run(fmax=fmax, steps=200)
    return atoms.get_potential_energy() / 2.0


def relax_with_mace(atoms, calc, fmax=0.05, steps=300, check_every=10,
                     bond_cutoff=BOND_CUTOFF_NN):
    """
    Relax with LBFGS, but periodically check (every `check_every` steps)
    whether the structure has already fragmented into >=2 disconnected
    pieces. If so, stop early rather than continuing to full convergence:
    a fragmented trajectory will only continue separating under MACE's
    gradient (the true minimum from that point IS the dissociated state),
    so further optimizer steps waste compute without changing the
    qualitative outcome. This was found necessary empirically: a sizeable
    fraction of purely-randomly-generated Nn starting geometries are NOT
    local minima of any cohesive n-atom molecule at all, and instead lie
    in the basin of n/2 x N2 (or other fragment combinations) -- this is
    expected physics for unbiased random sampling (see manual), not a
    bug, but it means most of the relaxation trajectory after the
    fragmentation point is computationally wasted for our purposes.
    """
    atoms = atoms.copy()
    atoms.calc = calc
    opt = LBFGS(atoms, logfile=None)

    try:
        remaining = steps
        while remaining > 0:
            this_chunk = min(check_every, remaining)
            converged = opt.run(fmax=fmax, steps=this_chunk)
            remaining -= this_chunk

            G, _ = infer_geometric_graph(atoms.get_positions(), bond_cutoff=bond_cutoff)
            if nx.number_connected_components(G) > 1:
                # Already fragmented: one more short relaxation burst to
                # let the fragments settle into clean local minima
                # individually (so reported energies are physically
                # meaningful), then stop.
                opt.run(fmax=fmax, steps=min(20, remaining if remaining > 0 else 20))
                break
            if converged:
                break
        energy = atoms.get_potential_energy()
    except Exception as exc:
        log.warning(f"    Relaxation failed: {exc}")
        return atoms, None, False

    return atoms, energy, True


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

def _airss_trial_worker(n, trial_idx, model_type, model_size, device, dtype,
                         minsep, bond_guess_max, fmax, max_opt_steps,
                         rng_seed):
    """
    Run ONE complete AIRSS trial (random generation + MACE relaxation +
    integrity check) in a fully self-contained way, suitable for
    execution in a separate OS process via ProcessPoolExecutor.

    WHY THIS IS A SEPARATE FUNCTION RATHER THAN REUSING THE MAIN LOOP
    -----------------------------------------------------------------
    The sequential version of this script loads ONE MACE calculator
    once in the main process and reuses it for every trial -- cheap and
    simple, but a loaded MACE calculator (a live PyTorch model object)
    cannot be reliably shared across a process pool: PyTorch model
    objects are not designed to be pickled and shared this way, and the
    minima-hopping script in this same project hit exactly this class
    of problem when using 'fork' to share parent-process PyTorch state
    (see polynitrogen_minimahopping.py's manual section on the
    fork-vs-spawn macOS failure). The robust fix, applied consistently
    across this project, is the same here: each worker loads its OWN
    MACE calculator from scratch. This costs a few seconds of
    per-trial overhead (model loading) compared to the sequential
    version's one-time cost, which is the trade made for safe
    parallelism -- exactly mirroring the cost/robustness trade-off
    already documented for Minima Hopping's parallel seeds.

    Returns a plain dict of results (no live Atoms/calculator objects
    cross the process boundary in their original form beyond what is
    explicitly serialized here), or None if generation/relaxation failed.
    """
    rng = np.random.default_rng(rng_seed)
    coords, gen_ok, gen_attempts = generate_random_sensible_molecule(
        n, minsep=minsep, bond_guess_max=bond_guess_max, rng=rng,
    )
    if not gen_ok:
        return {"status": "GEN_FAILED", "n": n, "trial_idx": trial_idx,
                "gen_attempts": gen_attempts}

    calc = load_mace_calculator(model_type, model_size, device, dtype)
    atoms = Atoms("N" * n, positions=coords)
    final_atoms, energy, converged = relax_with_mace(
        atoms, calc, fmax=fmax, steps=max_opt_steps
    )
    if energy is None:
        return {"status": "RELAX_FAILED", "n": n, "trial_idx": trial_idx,
                "gen_attempts": gen_attempts}

    integ = check_post_relaxation_sanity(final_atoms.get_positions())
    sym_info = get_symmetry(final_atoms)

    return {
        "status": "OK",
        "n": n,
        "trial_idx": trial_idx,
        "gen_attempts": gen_attempts,
        "energy_eV": energy,
        "positions": final_atoms.get_positions(),
        "symbols": final_atoms.get_chemical_symbols(),
        "integrity_status": integ["status"],
        "n_fragments": integ["n_fragments"],
        "max_geometric_degree": integ["max_geometric_degree"],
        "min_any_distance": integ["min_any_distance"],
        "max_any_distance": integ["max_any_distance"],
        "final_graph": integ["final_graph"],
        "spacegroup": sym_info["spacegroup"],
    }


def run_pipeline(n_min=4, n_max=20, n_trials=200,
                  model_type="mace_off", model_size="small",
                  device="cpu", dtype="float64",
                  minsep=MINSEP, bond_guess_max=BOND_GUESS_MAX,
                  fmax=0.05, max_opt_steps=300,
                  output_dir="polyN_AIRSS_results", seed=42,
                  n_workers=1):

    if n_min % 2 != 0:
        n_min += 1
    if n_max % 2 != 0:
        n_max -= 1
    sizes = list(range(n_min, n_max + 1, 2))

    out = Path(output_dir)
    dirs = {"xyz": out / "xyz", "cif": out / "cif"}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    calc = load_mace_calculator(model_type, model_size, device, dtype)
    if calc is None:
        sys.exit(1)

    log.info("Computing N2 reference energy...")
    e_n2_per_atom = compute_n2_reference(calc)
    log.info(f"  E(N2)/atom = {e_n2_per_atom:.6f} eV")

    csv_path = out / "airss_results.csv"
    fieldnames = [
        "id", "trial_idx", "n_atoms", "topology_label",
        "gen_attempts", "gen_success",
        "energy_eV", "energy_eV_per_atom", "delta_E_eV_per_atom_vs_N2",
        "is_screening_candidate",
        "integrity_status", "n_fragments", "max_geometric_degree",
        "min_any_distance_A", "max_any_distance_A",
        "spacegroup", "xyz_file", "cif_file",
    ]
    csv_file = open(csv_path, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    rng = np.random.default_rng(seed)
    mol_id = 0
    n_gen_failures = 0
    n_relax_failures = 0
    t_start = time.time()

    log.info(f"Starting AIRSS-style campaign: {n_trials} trials, "
             f"n in {{{n_min}..{n_max} (even)}}, n_workers={n_workers}")

    def _write_result(trial, n, gen_attempts, energy, e_per_atom, delta_e,
                       integ_status, n_fragments, max_geom_degree,
                       min_any_d, max_any_d, label, spacegroup, atoms):
        """Shared result-writing logic for both the sequential and
        parallel paths, so the two stay byte-for-byte consistent in
        what they write to results.csv / xyz / cif."""
        nonlocal mol_id
        if delta_e < MIN_SANE_DELTA_E_PER_ATOM:
            integ_status = "ENERGY_ANOMALY"
        is_candidate = (
            integ_status == "OK" and 0.0 <= delta_e < DELTA_E_SCREEN_THRESHOLD
        )
        mol_id += 1
        stem = f"trial{trial:05d}_N{n}_{label}"
        comment = (
            f"formula=N{n} topology={label} integrity={integ_status} "
            f"energy_eV={energy:.6f} delta_E_eV_atom={delta_e:.4f} "
            f"sg={spacegroup}"
        )
        xyz_path = dirs["xyz"] / f"{stem}.xyz"
        cif_path = dirs["cif"] / f"{stem}.cif"
        write_xyz(atoms, xyz_path, comment=comment)
        write_cif(atoms, cif_path)
        writer.writerow({
            "id": mol_id, "trial_idx": trial, "n_atoms": n,
            "topology_label": label, "gen_attempts": gen_attempts,
            "gen_success": True,
            "energy_eV": f"{energy:.6f}",
            "energy_eV_per_atom": f"{e_per_atom:.6f}",
            "delta_E_eV_per_atom_vs_N2": f"{delta_e:.4f}",
            "is_screening_candidate": is_candidate,
            "integrity_status": integ_status,
            "n_fragments": n_fragments,
            "max_geometric_degree": max_geom_degree,
            "min_any_distance_A": f"{min_any_d:.4f}",
            "max_any_distance_A": f"{max_any_d:.4f}",
            "spacegroup": spacegroup,
            "xyz_file": str(xyz_path.relative_to(out)),
            "cif_file": str(cif_path.relative_to(out)),
        })

    if n_workers <= 1:
        # Sequential path (default, n_workers=1): identical to every
        # previously validated run of this script. The single MACE
        # calculator loaded above is reused for all trials.
        for trial in range(n_trials):
            n = int(rng.choice(sizes))
            coords, gen_ok, gen_attempts = generate_random_sensible_molecule(
                n, minsep=minsep, bond_guess_max=bond_guess_max, rng=rng,
            )
            if not gen_ok:
                n_gen_failures += 1
                log.warning(f"  trial {trial}: generation FAILED for n={n} "
                            f"after {gen_attempts} attempts (skipped)")
                continue

            atoms = Atoms("N" * n, positions=coords)
            final_atoms, energy, converged = relax_with_mace(
                atoms, calc, fmax=fmax, steps=max_opt_steps
            )
            if energy is None:
                n_relax_failures += 1
                continue

            integ = check_post_relaxation_sanity(final_atoms.get_positions())
            label = classify_topology(integ["final_graph"])
            e_per_atom = energy / n
            delta_e = e_per_atom - e_n2_per_atom
            sym_info = get_symmetry(final_atoms)

            _write_result(
                trial, n, gen_attempts, energy, e_per_atom, delta_e,
                integ["status"], integ["n_fragments"],
                integ["max_geometric_degree"], integ["min_any_distance"],
                integ["max_any_distance"], label, sym_info["spacegroup"],
                final_atoms,
            )

            if (trial + 1) % max(1, n_trials // 10) == 0:
                csv_file.flush()
                log.info(f"  trial {trial+1}/{n_trials} done")

    else:
        # Parallel path: each trial runs in its own process (own MACE
        # calculator reloaded from scratch -- see _airss_trial_worker
        # docstring for why this is necessary rather than sharing the
        # already-loaded 'calc' object across the pool). As with
        # polynitrogen_minimahopping.py's --n_workers, 'spawn' is used
        # unconditionally for cross-platform reliability (a 'fork'-based
        # pool was found to silently fail every worker on macOS when
        # PyTorch state was already initialized in the parent -- see
        # that script's manual section for the full account).
        pool_ctx = mp.get_context("spawn")
        trial_sizes = [int(rng.choice(sizes)) for _ in range(n_trials)]
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=pool_ctx) as executor:
            futures = {
                executor.submit(
                    _airss_trial_worker, trial_sizes[trial], trial,
                    model_type, model_size, device, dtype,
                    minsep, bond_guess_max, fmax, max_opt_steps,
                    seed + trial,
                ): trial
                for trial in range(n_trials)
            }
            log.info(f"  Dispatched {len(futures)} trials to a pool of "
                      f"{n_workers} workers...")
            n_done = 0
            for future in as_completed(futures):
                trial = futures[future]
                n_done += 1
                try:
                    result = future.result()
                except Exception as exc:
                    log.warning(f"    trial {trial}: pool worker raised "
                              f"{type(exc).__name__}: {exc}")
                    n_relax_failures += 1
                    continue

                if result["status"] == "GEN_FAILED":
                    n_gen_failures += 1
                    log.warning(f"  trial {trial}: generation FAILED for "
                                f"n={result['n']} after "
                                f"{result['gen_attempts']} attempts (skipped)")
                    continue
                if result["status"] == "RELAX_FAILED":
                    n_relax_failures += 1
                    continue

                n = result["n"]
                label = classify_topology(result["final_graph"])
                e_per_atom = result["energy_eV"] / n
                delta_e = e_per_atom - e_n2_per_atom
                atoms = Atoms(result["symbols"], positions=result["positions"])

                _write_result(
                    trial, n, result["gen_attempts"], result["energy_eV"],
                    e_per_atom, delta_e, result["integrity_status"],
                    result["n_fragments"], result["max_geometric_degree"],
                    result["min_any_distance"], result["max_any_distance"],
                    label, result["spacegroup"], atoms,
                )

                if n_done % max(1, n_trials // 10) == 0:
                    csv_file.flush()
                    log.info(f"  trial {n_done}/{n_trials} done")

    csv_file.close()

    # ── Hull-style summary: best (lowest) delta_E per n among OK structures ──
    hull_path = out / "airss_hull_summary.csv"
    by_n = {}
    status_counts = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            status_counts[row["integrity_status"]] = status_counts.get(
                row["integrity_status"], 0) + 1
            if row["integrity_status"] != "OK":
                continue
            n = int(row["n_atoms"])
            de = float(row["delta_E_eV_per_atom_vs_N2"])
            by_n.setdefault(n, []).append(de)
    with open(hull_path, "w", newline="") as f:
        hw = csv.writer(f)
        hw.writerow(["n_atoms", "n_OK_structures_found", "min_delta_E_eV_per_atom"])
        for n in sorted(by_n):
            hw.writerow([n, len(by_n[n]), f"{min(by_n[n]):.4f}"])

    elapsed_total = time.time() - t_start
    n_evaluated = sum(status_counts.values())
    log.info("=" * 60)
    log.info(f"DONE. {mol_id} structures written in {elapsed_total:.1f}s "
             f"({n_gen_failures} generation failures, "
             f"{n_relax_failures} relaxation failures)")
    log.info("Post-relaxation integrity breakdown (this quantifies how "
             "often unbiased random sampling lands in a cohesive basin):")
    for status in ("OK", "FRAGMENTED", "COLLAPSED", "ENERGY_ANOMALY"):
        count = status_counts.get(status, 0)
        pct = 100.0 * count / n_evaluated if n_evaluated else 0.0
        log.info(f"    {status:14s}: {count:5d}  ({pct:5.1f}%)")
    log.info(f"  airss_results.csv      : {csv_path}")
    log.info(f"  airss_hull_summary.csv : {hull_path}")
    return csv_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--n_min", type=int, default=4)
    p.add_argument("--n_max", type=int, default=20)
    p.add_argument("--n_trials", type=int, default=200,
                   help="Total independent random trials across all sizes. Default 200.")
    p.add_argument("--minsep", type=float, default=MINSEP,
                   help=f"Minimum N-N separation at generation, Angstrom. Default {MINSEP}.")
    p.add_argument("--bond_guess_max", type=float, default=BOND_GUESS_MAX,
                   help=f"Max proposal distance from anchor, Angstrom. Default {BOND_GUESS_MAX}.")
    p.add_argument("--model", type=str, default="mace_off", choices=["mace_off", "mace_mp"])
    p.add_argument("--model_size", type=str, default="small")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    p.add_argument("--fmax", type=float, default=0.05)
    p.add_argument("--max_steps", type=int, default=300)
    p.add_argument("--output_dir", type=str, default="polyN_AIRSS_results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_workers", type=int, default=1,
                   help="Number of trials to run CONCURRENTLY via a process "
                        "pool (default 1 = fully sequential, identical "
                        "behavior to all previously validated runs). Each "
                        "worker reloads its own MACE model from scratch "
                        "(a few seconds overhead per trial, the cost of "
                        "safe cross-process parallelism -- see "
                        "_airss_trial_worker docstring). Set conservatively "
                        "relative to available physical cores; uses "
                        "'spawn' unconditionally for cross-platform "
                        "reliability (see polynitrogen_minimahopping.py's "
                        "manual section on the fork-vs-spawn macOS issue "
                        "for why).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    log.info("polynitrogen_airss.py -- AIRSS-style variable-n random screening")
    log.info(f"  N range  : N{args.n_min} to N{args.n_max} (even only)")
    log.info(f"  trials   : {args.n_trials}")
    log.info(f"  parallel : n_workers={args.n_workers} "
             f"({'sequential' if args.n_workers <= 1 else 'concurrent pool'})")
    log.info(f"  MACE     : {args.model}/{args.model_size} on {args.device}")
    log.info(f"  output   : {args.output_dir}/")

    run_pipeline(
        n_min=args.n_min, n_max=args.n_max, n_trials=args.n_trials,
        model_type=args.model, model_size=args.model_size,
        device=args.device, dtype=args.dtype,
        minsep=args.minsep, bond_guess_max=args.bond_guess_max,
        fmax=args.fmax, max_opt_steps=args.max_steps,
        output_dir=args.output_dir, seed=args.seed,
        n_workers=args.n_workers,
    )
