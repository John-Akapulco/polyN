#!/usr/bin/env python3
"""
polynitrogen_charged_explore.py
===================================
Multi-size exploration of charged polynitrogen anions (Nn^-, n odd) using
SCC-DFTB as the energy/force engine, with TWO selectable exploration
strategies mirroring the rest of this project:

    --mode airss   : unbiased random structure generation (one or more
                      independent trials per size), analogous to
                      polynitrogen_airss.py
    --mode mh       : global PES exploration via Minima Hopping from
                      diverse random seeds per size, analogous to
                      polynitrogen_minimahopping.py

WHY ANIONS SPECIFICALLY, AND WHY ODD n
------------------------------------------
Nn^- has 7n+1 electrons. For a closed-shell singlet (even electron
count, representable by the same restricted-SCC-DFTB Hamiltonian used
for the validated N5+/N5- calculation in this project):
    7n + 1 even  <=>  7n odd  <=>  n odd
This is the exact mirror image of the neutral-species constraint (even
n required there); for the ANION family, n must be ODD. This is
verified consistent with the request (N3-, N5-, N7-, ...) and with the
already-validated N5- result (ring-5, SCC-DFTB, integrity OK).

NO LITERATURE-INFORMED SEED BIAS (BY EXPLICIT USER REQUEST)
-----------------------------------------------------------------
Unlike polynitrogen_charged_dftb.py (which seeded N5- directly as a
planar ring informed by the known cyclo-pentazolate structure), this
script deliberately uses ONLY unbiased generation -- the same
MINSEP-constrained random anchor-growth algorithm validated in
polynitrogen_airss.py, and the same diverse chain/ring/random seed
mixture validated in polynitrogen_minimahopping.py -- for every size,
including n=5. This is a deliberate methodological choice: it avoids
the risk of a confirmation-bias-style result where the "discovery" of
the ring-5 pentazolate topology is trivial because it was handed to the
optimizer as the starting point rather than found by unbiased search.

WHY DFTB+ (NOT MACE) FOR THIS SCRIPT
-----------------------------------------
MACE-OFF23 was verified directly (see companion conversation and the
methodology manual) to be completely insensitive to declared atomic
charge -- energies for a "charged" vs neutral Nn geometry were
IDENTICAL to machine precision. SCC-DFTB, by construction, solves its
Hamiltonian self-consistently with respect to the declared total
charge, making it genuinely charge-aware. This was validated end-to-end
on N5+/N5- in this project (DFTB+ N2 bond length within 10 mA of the
experimental reference; N5+ relaxed to a chain, N5- to a planar ring,
matching the qualitative topology of the experimentally characterized
pentazenium cation and cyclo-pentazolate anion respectively).

PERFORMANCE NOTE: DFTB+ IS SLOWER PER CALL THAN MACE
---------------------------------------------------------
Unlike MACE (an in-memory PyTorch model), DFTB+ is a FileIOCalculator:
every energy/force evaluation writes input files to disk, launches the
external dftb+ binary as a subprocess, and parses its output files.
This is substantially slower per call than MACE (typically by 1-2
orders of magnitude for a small molecule), and is why --n_trials /
--n_steps defaults in this script are set lower than their MACE-based
counterparts. Budget your campaigns accordingly.

PARALLEL-SAFETY NOTE: each worker (whether an AIRSS trial running in
the main process, or a Minima Hopping seed running in its own
subprocess) is given its own working directory / DFTB+ label prefix,
so concurrent DFTB+ runs never overwrite each other's input/output
files on disk.

INSTALLATION
-------------
    conda install -c conda-forge dftbplus
    # Download mio-1-1 Slater-Koster files from
    # https://github.com/dftbparams/mio/releases (verified working
    # mirror; download the .tar.xz asset and extract it).

USAGE
------
    # AIRSS-style: 20 random trials per odd size from 3 to 7
    python polynitrogen_charged_explore.py --slako_dir ~/dftb_params/mio-1-1/ \\
        --mode airss --n_min 3 --n_max 7 --n_trials 20 --charge -1

    # Minima Hopping: 3 seeds x 15 steps per odd size from 3 to 7
    python polynitrogen_charged_explore.py --slako_dir ~/dftb_params/mio-1-1/ \\
        --mode mh --n_min 3 --n_max 7 --n_seeds 3 --n_steps 15 --charge -1

AUTHOR
-------
    Generated for IC2MP/E4 Mediacat -- Universite de Poitiers
"""

import argparse
import csv
import logging
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
from pathlib import Path

from scipy.optimize import minimize

import networkx as nx
import numpy as np

try:
    from ase import Atoms
    from ase.io import read, write
    from ase.optimize import LBFGS
except ImportError:
    sys.exit("ERROR: ASE not found. Run: pip install ase")

try:
    from ase.calculators.dftb import Dftb
except ImportError:
    sys.exit("ERROR: ASE's DFTB+ calculator module not found.")

try:
    from ase.constraints import Hookean
    from ase.optimize.minimahopping import MinimaHopping
    HAVE_MH = True
except ImportError:
    HAVE_MH = False

try:
    import spglib
    SPGLIB_OK = True
except ImportError:
    SPGLIB_OK = False


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("polyN-charged-explore")


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

BOND_CUTOFF_NN = 1.60
MIN_SANE_NN_DISTANCE = 0.90
MAX_PHYSICAL_DEGREE = 3

# MINSEP is set higher here (1.15 A) than in the MACE-based AIRSS
# script (1.05 A there). Reason, discovered empirically in production:
# DFTB+ must explicitly diagonalize a tight-binding Hamiltonian/overlap
# matrix built from tabulated Slater-Koster two-center integrals
# (Section "SCC-DFTB engine" of the companion manual); a starting
# geometry with two atoms closer than ~1.1 A (i.e. at or below the
# physical N#N triple-bond length) sits in a regime the Slater-Koster
# tables were not meant to extrapolate into, and was observed to
# trigger genuine Fortran-level numerical failures in the dftb+ binary
# itself (IEEE_OVERFLOW/DIVIDE_BY_ZERO, "ERROR STOP") at a rate of
# ~17% over a 300-trial production campaign. MACE, being a neural
# network rather than an explicit linear-algebra diagonalization, has
# no equivalent failure mode -- it always returns SOME (possibly very
# high) energy rather than crashing, which is why MINSEP could safely
# be set lower there. Raising MINSEP here trades a small amount of
# coverage of very short/strained starting contacts for a meaningfully
# lower DFTB+ crash rate; combined with the retry logic in
# run_airss_mode (below), the net effect should be both fewer crashes
# and no loss of usable trials (a crashed geometry carries no
# information anyway).
MINSEP = 1.15
BOND_GUESS_MAX = 1.65

# Classical force-field constants for graph-seeded geometry pre-relaxation.
# Transplanted verbatim from polynitrogen_mace.py (where they were validated
# empirically), applied here to improve the quality of MH seed geometries
# before they are handed to DFTB+: a graph-embedded + FF-pre-relaxed geometry
# starts much closer to a local minimum of the target topology than a raw
# spring-layout placement, which reduces the probability that DFTB+'s SCC
# solver encounters a numerically pathological starting configuration.
# The target bond length 1.35 A is intermediate between N-N single (1.45 A)
# and double (1.25 A) bonds, appropriate as a generic starting point for any
# polynitrogen topology. Angle targets: 120 deg for 2-coordinate N (sp2-like,
# e.g. in chains), 109.5 deg for 3-coordinate N (sp3-like, branching nodes).
NN_BOND_FF = 1.35      # Angstrom, target bond length for FF pre-relaxation
FF_K_BOND  = 200.0     # eV/A^2, harmonic bond stretching constant
FF_K_ANGLE = 100.0     # eV/rad^2, harmonic angle bending constant
FF_K_REP   = 0.5       # eV, non-bonded repulsion prefactor
FF_REP_EXP = 9         # exponent for r^(-FF_REP_EXP) repulsion
TARGET_ANGLES_FF = {2: 120.0, 3: 109.5}  # degrees by coordination number

MH_T0 = 300.0
MH_EDIFF0 = 0.05
MH_MDMIN = 2
MH_FMAX = 0.05
MH_MAX_TEMP = 3000.0
MH_SEED_TIMEOUT = 240

DELTA_E_SCREEN_THRESHOLD = 3.0


# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY GENERATION (unbiased, reused/adapted from polynitrogen_airss.py
# and polynitrogen_minimahopping.py -- NO literature-informed seeding)
# ─────────────────────────────────────────────────────────────────────────────

def ff_energy_charged(flat_coords, G, bond_length=NN_BOND_FF):
    """
    Classical force-field energy for graph-seeded geometry pre-relaxation,
    identical in functional form to polynitrogen_mace.py's ff_energy().
    The suffix '_charged' distinguishes it from any future neutral-species
    version imported into the same namespace; functionally, the FF is
    charge-agnostic (it only knows about the graph topology, not the
    electronic state) -- the charge affects bond lengths/angles at the
    DFT/DFTB level, not at this pre-relaxation stage.

    Terms:
      - Harmonic bond stretching: k_bond * (d - d0)^2 for each graph edge
      - Harmonic angle bending:   k_angle * (cos(theta) - cos(theta_0))^2
        for each pair of bonds sharing a common atom
      - Repulsive non-bonded:     k_rep * (2*d0/r)^exp for non-bonded pairs
    """
    n = len(G.nodes())
    xyz = flat_coords.reshape(n, 3)
    edges = list(G.edges())
    bonded = set(edges) | {(v, u) for u, v in edges}
    E = 0.0
    for u, v in edges:
        d = np.linalg.norm(xyz[u] - xyz[v])
        E += FF_K_BOND * (d - bond_length) ** 2
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in bonded and (j, i) not in bonded:
                d = max(np.linalg.norm(xyz[i] - xyz[j]), 0.3)
                E += FF_K_REP * (2.0 * bond_length / d) ** FF_REP_EXP
    for node in G.nodes():
        nb = list(G.neighbors(node))
        deg = len(nb)
        if deg < 2:
            continue
        target_angle = np.radians(TARGET_ANGLES_FF.get(deg, 109.5))
        for i in range(len(nb)):
            for j in range(i + 1, len(nb)):
                v1 = xyz[nb[i]] - xyz[node]
                v2 = xyz[nb[j]] - xyz[node]
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 < 1e-8 or n2 < 1e-8:
                    continue
                cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
                E += FF_K_ANGLE * (cos_a - np.cos(target_angle)) ** 2
    return E


def embed_graph_3d_ff(G, bond_length=NN_BOND_FF, n_conformers=3, seed=0):
    """
    Generate a 3D embedding of graph G pre-relaxed with the classical FF,
    returning the best conformer (lowest FF energy) as a numpy coordinate
    array. This is the graph-seeded + FF-pre-relaxed analogue of the simple
    spring-layout embedding used before this improvement was added.

    WHY THIS IS BETTER THAN RAW SPRING-LAYOUT FOR DFTB+ (motivation added
    following a direct user suggestion, validated by the scientific argument
    below):
      - A raw spring-layout placement has correct graph topology but arbitrary
        bond angles and lengths (the spring constant is uniform, with no
        chemical knowledge of angle preferences).
      - DFTB+ diagonalizes an explicit tight-binding Hamiltonian at EVERY
        geometry evaluation: even a single step from a chemically unreasonable
        geometry (e.g. a 60-degree angle at a nominally sp2 nitrogen) can
        produce a numerically ill-conditioned overlap matrix, triggering the
        IEEE_OVERFLOW / ERROR STOP failures observed in production.
      - A FF-pre-relaxed geometry has bond lengths near NN_BOND_FF (~1.35 A)
        and angles near 109.5/120 deg, firmly within the Slater-Koster
        parameterization range, dramatically reducing the probability of a
        first-step numerical failure.
      - Additionally, starting closer to a local minimum of the TARGET
        TOPOLOGY means MH's first local relaxation step is more likely to
        stay within that topological basin rather than immediately crossing
        into a fragmentation basin -- directly addressing the user's insight
        that graph-theoretic seed generation should reduce fragmentation rates.

    n_conformers: number of independently perturbed starting points to try;
    the lowest-FF-energy result is returned (best of n_conformers).
    """
    import warnings
    n = len(G.nodes())
    rng = np.random.default_rng(seed)
    best_coords, best_energy = None, np.inf

    for ci in range(n_conformers):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pos = nx.spring_layout(G, dim=3, seed=ci, k=bond_length * 1.2)
        init = np.array([pos[i] for i in range(n)], dtype=float)
        edges = list(G.edges())
        if edges:
            mean_d = np.mean([np.linalg.norm(init[u] - init[v])
                              for u, v in edges])
            if mean_d > 1e-8:
                init *= bond_length / mean_d
        init += rng.normal(0, 0.15 * (ci + 0.5), size=(n, 3))
        result = minimize(
            ff_energy_charged, init.flatten(),
            args=(G, bond_length),
            method="L-BFGS-B",
            options={"maxiter": 10000, "ftol": 1e-14, "gtol": 1e-10},
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_coords = result.x.reshape(n, 3)

    best_coords -= best_coords.mean(axis=0)
    return best_coords


def generate_random_sensible_molecule(n, minsep=MINSEP, bond_guess_max=BOND_GUESS_MAX,
                                       max_degree=MAX_PHYSICAL_DEGREE,
                                       max_attempts=20000, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    coords = np.zeros((n, 3))
    coords[0] = [0.0, 0.0, 0.0]
    placed = 1
    attempts = 0
    while placed < n and attempts < max_attempts:
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
        if np.any(all_dists < minsep):
            continue
        candidate_degree = int(np.sum(all_dists < bond_guess_max))
        if candidate_degree > max_degree or candidate_degree == 0:
            continue
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
    success = (placed == n)
    coords -= coords[:placed].mean(axis=0) if placed > 0 else 0
    return coords[:placed], success


def seed_topologies_for_mh(n, n_seeds, max_degree=MAX_PHYSICAL_DEGREE, seed=0):
    rng = np.random.default_rng(seed)
    seeds = []
    chain = nx.path_graph(n)
    if max(dict(chain.degree()).values()) >= 2:
        seeds.append(chain)
    if n >= 3:
        seeds.append(nx.cycle_graph(n))
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
        if any(nx.is_isomorphic(G, s) for s in seeds):
            continue
        seeds.append(G)
    return seeds[:n_seeds]


def embed_graph_3d(G, bond_length=1.35, seed=0):
    import warnings
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
# STRUCTURAL INTEGRITY CHECK
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


def check_structural_integrity(coords, bond_cutoff=BOND_CUTOFF_NN,
                                min_distance=MIN_SANE_NN_DISTANCE,
                                max_physical_degree=MAX_PHYSICAL_DEGREE):
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
        "status": status, "is_connected": is_connected,
        "n_fragments": n_fragments, "min_any_distance": min_any_distance,
        "max_any_distance": max_any_distance,
        "max_geometric_degree": max_geometric_degree, "final_graph": G,
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
# DFTB+ CALCULATOR
# ─────────────────────────────────────────────────────────────────────────────

def check_dftb_available():
    if shutil.which("dftb+") is None:
        sys.exit(
            "ERROR: the 'dftb+' executable was not found on your PATH.\n"
            "Install it first, e.g.: conda install -c conda-forge dftbplus\n"
            "and download Slater-Koster files (e.g. from "
            "https://github.com/dftbparams/mio/releases)."
        )


def load_dftb_calculator(slako_dir, charge=0, label="dftb_work/calc",
                          scc_tolerance=1e-7, max_scc_iterations=200):
    slako_path = str(Path(slako_dir).expanduser().resolve()) + "/"
    if not Path(slako_dir).expanduser().exists():
        sys.exit(f"ERROR: --slako_dir '{slako_dir}' does not exist.")
    Path(label).parent.mkdir(parents=True, exist_ok=True)
    calc = Dftb(
        label=label,
        Hamiltonian_="DFTB",
        Hamiltonian_SCC="Yes",
        Hamiltonian_SCCTolerance=scc_tolerance,
        Hamiltonian_MaxSCCIterations=max_scc_iterations,
        Hamiltonian_MaxAngularMomentum_="",
        Hamiltonian_MaxAngularMomentum_N="p",
        Hamiltonian_Charge=charge,
        slako_dir=slako_path,
    )
    return calc


def relax_with_dftb(atoms, calc, fmax=0.05, steps=300, check_every=10,
                     bond_cutoff=BOND_CUTOFF_NN):
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
                opt.run(fmax=fmax, steps=min(20, remaining if remaining > 0 else 20))
                break
            if converged:
                break
        energy = atoms.get_potential_energy()
    except Exception as exc:
        log.warning(f"    DFTB+ relaxation failed: {exc}")
        return atoms, None, False
    return atoms, energy, True


def sanity_check_n2(slako_dir):
    log.info("Running N2 sanity check with the current DFTB+ setup...")
    calc = load_dftb_calculator(slako_dir, charge=0, label="dftb_work/n2_sanity")
    atoms = Atoms("N2", positions=[[0, 0, 0], [0, 0, 1.10]])
    atoms.calc = calc
    opt = LBFGS(atoms, logfile=None)
    try:
        opt.run(fmax=0.01, steps=200)
        d = atoms.get_distance(0, 1)
        e = atoms.get_potential_energy()
        log.info(f"  N2 (DFTB+): d(N-N) = {d:.4f} A "
                 f"(experimental reference: 1.0977 A), E = {e:.4f} eV")
        if abs(d - 1.0977) > 0.05:
            log.warning("  N2 bond length deviates notably from experiment "
                       "-- check your DFTB+/Slako installation.")
        return True, e / 2
    except Exception as exc:
        log.error(f"  N2 sanity check FAILED: {exc}")
        return False, None


# ─────────────────────────────────────────────────────────────────────────────
# AIRSS MODE
# ─────────────────────────────────────────────────────────────────────────────

def compute_size_weights(sizes, power=2):
    """
    Compute trial-count sampling weights proportional to
    (3n - 6)^power, where (3n - 6) is the number of internal
    (vibrational/conformational) degrees of freedom of a non-linear
    N-atom molecule after removing the 3 translational and 3
    rotational external degrees of freedom (standard result from
    molecular degrees-of-freedom counting; for a linear molecule the
    count is 3n-5, but 3n-6 is used uniformly here as the generic
    non-linear case, consistent with how this quantity is normally
    reported).

    RATIONALE (raised directly by the user): the AIRSS-style generator
    was previously sampling n UNIFORMLY across the requested size
    range, giving every size an equal number of trials regardless of
    how large its underlying conformational space actually is. Since
    the volume of accessible conformational space grows with the
    number of internal degrees of freedom -- at least linearly, and
    plausibly faster, since each additional degree of freedom adds a
    further dimension to search -- a FIXED trial budget per size
    systematically under-samples large n relative to small n. This
    function lets that budget instead grow with (3n-6)^power, so
    larger (higher-dimensional) sizes receive proportionally more
    trials than smaller ones, in an attempt to maintain comparable
    SAMPLING DENSITY (trials per unit of conformational space) rather
    than comparable trial COUNT across sizes.

    power=2 (the default requested) makes this comparison even more
    aggressive than a linear (3n-6) weighting: e.g. for n=3 vs n=7,
    3n-6 = 3 vs 15 (5x), but (3n-6)^2 = 9 vs 225 (25x) -- meaning N7
    would receive roughly 25 times as many trials as N3 in this
    scheme, for the same total trial budget. This is a deliberate,
    aggressive choice favoring depth of coverage at large n over
    breadth of coverage at small n; power=1 (linear) is available as a
    gentler alternative by passing power=1 if this proves too
    aggressive for n_max much larger than n_min in practice.

    Returns a list of weights, same length and order as `sizes`,
    normalized to sum to 1.0 (a valid probability distribution for
    np.random.Generator.choice's p= argument).
    """
    dof = np.array([max(3 * n - 6, 1) for n in sizes], dtype=float)
    weights = dof ** power
    weights /= weights.sum()
    return weights


def _dftb_relax_worker(coords, n, charge, slako_dir, fmax, max_steps,
                        label, result_path):
    """
    Run ONE DFTB+ relaxation in a SEPARATE PROCESS, writing the result
    to disk, so the parent can enforce a hard wall-clock timeout by
    killing this process if it never returns.

    WHY THIS WAS NECESSARY (discovered in production)
    -------------------------------------------------------
    relax_with_dftb() (the in-process version used directly by
    run_airss_mode before this fix) has NO timeout mechanism at all.
    Unlike the DFTB+ "ERROR STOP" crashes already handled by the retry
    logic above (which fail fast, within seconds), a DFTB+ self-
    consistent-charge cycle that fails to converge cleanly can instead
    hang indefinitely inside the external dftb+ subprocess -- neither
    crashing nor returning -- with no signal back to the calling ASE
    LBFGS loop. This was observed directly in a production run: trial
    169 finished normally, then the NEXT log line (trial 171's crash
    report) appeared 12 minutes later, with no intervening output --
    a single trial silently consumed essentially the entire campaign's
    remaining time budget. This mirrors, almost exactly, the unbounded-
    internal-loop failure mode already discovered and fixed for
    polynitrogen_minimahopping.py's MD phase (see that script's manual
    section) -- the same root cause (an external numerical solver with
    no convergence guarantee) recurring in a different part of this
    project, now fixed the same proven way: isolate the risky call in
    its own OS process and let the parent enforce a hard kill.

    Why a separate process rather than another in-process mechanism: a
    signal-based (SIGALRM) timeout was already tested for the
    MACE/Minima-Hopping case and found unreliable once a tight,
    repeated-call loop is involved (the interpreter can defer signal
    delivery indefinitely). A subprocess.run()-based timeout on the
    dftb+ binary itself was considered but rejected: ASE's Dftb
    FileIOCalculator launches the binary internally without exposing a
    timeout parameter, so the only externally controllable boundary is
    the whole Python call into ASE, hence the same multiprocessing
    pattern used for Minima Hopping is applied here too.

    Writes a small numpy .npz file (result_path) containing the
    relaxed positions and energy on success; writes nothing if the
    relaxation raised an exception (the parent treats a missing file,
    whether from a clean failure or a hard kill, identically).
    """
    atoms = Atoms("N" * n, positions=coords)
    calc = load_dftb_calculator(slako_dir, charge=charge, label=label)
    try:
        final_atoms, energy, converged = relax_with_dftb(
            atoms, calc, fmax=fmax, steps=max_steps
        )
    except Exception:
        return
    if energy is None:
        return
    np.savez(result_path, positions=final_atoms.get_positions(), energy=energy)


def relax_with_dftb_timeout(coords, n, charge, slako_dir, fmax, max_steps,
                             label, timeout_seconds):
    """
    Parent-side wrapper: runs _dftb_relax_worker in a subprocess (via
    'spawn', for the same cross-platform-reliability reasons documented
    extensively in polynitrogen_minimahopping.py's manual section --
    'fork' was found to silently break every worker on macOS once
    PyTorch/native-library state is involved; DFTB+'s ASE wrapper is
    less directly implicated, but the same caution is applied
    uniformly across this project rather than re-litigated per script),
    and forcibly kills it if timeout_seconds elapses without a result.

    Returns (positions, energy) on success, or (None, None) on any
    failure (crash, exception, or hard timeout) -- the caller cannot
    distinguish these cases from this return value alone, which is
    fine here since run_airss_mode already retries on any failure type.
    """
    result_path = Path(label).parent / "result.npz"
    if result_path.exists():
        result_path.unlink()

    ctx = mp.get_context("spawn")
    process = ctx.Process(
        target=_dftb_relax_worker,
        args=(coords, n, charge, slako_dir, fmax, max_steps, label,
              str(result_path)),
    )
    process.start()
    process.join(timeout=timeout_seconds)
    if process.is_alive():
        log.warning(f"    DFTB+ relaxation HARD-TIMED-OUT after "
                   f"{timeout_seconds}s and was forcibly terminated "
                   f"(see relax_with_dftb_timeout docstring -- this is "
                   f"the DFTB+ analogue of the unbounded-MD-loop issue "
                   f"already fixed for Minima Hopping).")
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join()

    if not result_path.exists():
        return None, None
    data = np.load(result_path)
    return data["positions"], float(data["energy"])


def run_airss_mode(n_min, n_max, n_trials, charge, slako_dir, fmax, max_steps,
                    seed, out, writer, csv_file, e_n2_per_atom,
                    max_dftb_retries=2, size_weight_power=2,
                    dftb_timeout=120):
    """
    max_dftb_retries : int
        If the dftb+ binary itself crashes (Fortran-level numerical
        failure, e.g. IEEE_OVERFLOW/DIVIDE_BY_ZERO/"ERROR STOP" --
        observed in production at ~17% of trials before MINSEP was
        raised, see MINSEP docstring), a crashed geometry carries no
        usable information. Rather than simply discarding that trial
        slot, generate a FRESH random geometry and retry, up to this
        many times, before giving up on the slot. This keeps the
        effective number of evaluated trials closer to --n_trials
        despite the underlying DFTB+ crash rate, at a modest extra
        cost (a few wasted dftb+ invocations per crash).
    size_weight_power : float
        Exponent applied to (3n-6) when computing per-size trial
        sampling weights (see compute_size_weights docstring). Default
        2 (quadratic): larger sizes receive disproportionately more
        trials, reflecting their larger conformational space. Set to 0
        to recover the original uniform-over-sizes sampling.
    dftb_timeout : int
        Hard wall-clock timeout (seconds) per DFTB+ relaxation attempt,
        enforced by running it in a separate process that the parent
        forcibly kills if exceeded (see relax_with_dftb_timeout
        docstring for the full rationale -- this was added after a
        production run showed a single trial silently consuming 12
        minutes with no error and no progress). Default 120s, generous
        relative to the few-seconds-to-tens-of-seconds typical cost of
        a successful small-Nn relaxation observed in production.
    """
    sizes = list(range(n_min, n_max + 1, 2))
    size_weights = compute_size_weights(sizes, power=size_weight_power)
    rng = np.random.default_rng(seed)
    mol_id = 0
    status_counts = {}
    n_dftb_crashes = 0
    n_dftb_timeouts = 0

    log.info(f"AIRSS-style campaign: {n_trials} trials, "
             f"n in {{{n_min}..{n_max} (odd)}}, charge={charge:+d}")
    log.info("  Per-size trial weighting: (3n-6)^{:.0f}".format(size_weight_power))
    for n, w in zip(sizes, size_weights):
        log.info(f"    N{n}: weight={w:.3f}  "
                 f"(expected ~{w*n_trials:.0f}/{n_trials} trials)")
    log.info(f"  Per-attempt DFTB+ hard timeout: {dftb_timeout}s")

    for trial in range(n_trials):
        n = int(rng.choice(sizes, p=size_weights))

        energy = None
        final_atoms = None
        for attempt in range(max_dftb_retries + 1):
            coords, gen_ok = generate_random_sensible_molecule(n, rng=rng)
            if not gen_ok:
                log.warning(f"  trial {trial}: generation incomplete for "
                           f"n={n}, skipped")
                break

            label = f"dftb_work/airss_t{trial:05d}_a{attempt}/calc"
            positions, energy = relax_with_dftb_timeout(
                coords, n, charge, slako_dir, fmax, max_steps,
                label, dftb_timeout,
            )
            if energy is not None:
                final_atoms = Atoms("N" * n, positions=positions)
                break
            n_dftb_crashes += 1
            if attempt < max_dftb_retries:
                log.info(f"    trial {trial}: DFTB+ crashed/timed-out on "
                         f"attempt {attempt+1}/{max_dftb_retries+1}, "
                         f"retrying with a fresh geometry...")

        if energy is None:
            continue

        integ = check_structural_integrity(final_atoms.get_positions())
        label = classify_topology(integ["final_graph"])
        e_per_atom = energy / n
        delta_e = e_per_atom - e_n2_per_atom if e_n2_per_atom is not None else None
        status_counts[integ["status"]] = status_counts.get(integ["status"], 0) + 1

        mol_id += 1
        stem = f"airss_trial{trial:05d}_N{n}{'+' if charge>0 else '-'}_{label}"
        comment = (
            f"n={n} charge={charge:+d} topology={label} "
            f"integrity={integ['status']} energy_eV={energy:.6f} "
            f"method=SCC-DFTB(mio-1-1)"
        )
        xyz_path = out / "xyz" / f"{stem}.xyz"
        cif_path = out / "cif" / f"{stem}.cif"
        write_xyz(final_atoms, xyz_path, comment=comment)
        write_cif(final_atoms, cif_path)

        writer.writerow({
            "mode": "airss", "n_atoms": n, "charge": charge,
            "topology_label": label, "energy_eV": f"{energy:.6f}",
            "energy_eV_per_atom": f"{e_per_atom:.6f}",
            "delta_E_eV_per_atom_vs_N2": f"{delta_e:.4f}" if delta_e is not None else "",
            "integrity_status": integ["status"],
            "xyz_file": str(xyz_path.relative_to(out)),
            "cif_file": str(cif_path.relative_to(out)),
        })
        if (trial + 1) % max(1, n_trials // 10) == 0:
            csv_file.flush()
            log.info(f"  trial {trial+1}/{n_trials} done")

    log.info("AIRSS integrity breakdown: " +
             ", ".join(f"{k}={v}" for k, v in status_counts.items()))
    log.info(f"DFTB+ crashes (Fortran-level numerical failures, including "
             f"successfully-retried ones): {n_dftb_crashes} total across "
             f"{n_trials} trial slots")


# ─────────────────────────────────────────────────────────────────────────────
# MINIMA HOPPING MODE
# ─────────────────────────────────────────────────────────────────────────────

def _mh_seed_worker(G_seed_edges, n, charge, slako_dir, n_steps, seed_idx,
                     work_dir_str, T0, Ediff0, fmax, mdmin, max_temp, rng_seed):
    G_seed = nx.Graph()
    G_seed.add_nodes_from(range(n))
    G_seed.add_edges_from(G_seed_edges)

    work_dir = Path(work_dir_str)
    work_dir.mkdir(parents=True, exist_ok=True)
    calc = load_dftb_calculator(slako_dir, charge=charge,
                                label=str(work_dir / "calc"))

    # Use graph-seeded + FF-pre-relaxed embedding instead of raw spring
    # layout. The FF pre-relaxation brings bond lengths to ~1.35 A and
    # angles to chemically reasonable values (109.5/120 deg) before DFTB+
    # ever sees the geometry, reducing the probability of a first-step
    # Hamiltonian ill-conditioning (IEEE_OVERFLOW / ERROR STOP) and keeping
    # the starting point closer to the topological basin of the seed graph
    # -- directly implementing the user's suggestion to use graph theory
    # to generate better initial structures (see embed_graph_3d_ff docstring
    # for the full scientific argument).
    coords = embed_graph_3d_ff(G_seed, seed=rng_seed)
    atoms = Atoms("N" * n, positions=coords)
    atoms.calc = calc

    constraints = [
        Hookean(a1=int(u), a2=int(v), rt=2.0, k=8.0)
        for u, v in G_seed.edges()
    ]
    atoms.set_constraint(constraints)

    minima_traj_path = work_dir / f"minima_seed{seed_idx}.traj"
    log_path = work_dir / f"hop_seed{seed_idx}.log"
    for stale in (minima_traj_path, log_path):
        if stale.exists():
            stale.unlink()

    opt = MinimaHopping(
        atoms, T0=T0, Ediff0=Ediff0, mdmin=mdmin, fmax=fmax,
        minima_traj=str(minima_traj_path), logfile=str(log_path),
    )
    try:
        opt(totalsteps=n_steps, maxtemp=max_temp)
    except Exception:
        pass


def run_mh_seed(G_seed, n, charge, slako_dir, n_steps, seed_idx, work_dir,
                 timeout_seconds, rng_seed):
    ctx = mp.get_context("spawn")
    process = ctx.Process(
        target=_mh_seed_worker,
        args=(list(G_seed.edges()), n, charge, slako_dir, n_steps, seed_idx,
              str(work_dir), MH_T0, MH_EDIFF0, MH_FMAX, MH_MDMIN,
              MH_MAX_TEMP, rng_seed),
    )
    process.start()
    process.join(timeout=timeout_seconds)
    if process.is_alive():
        log.warning(f"    MH seed {seed_idx} HARD-TIMED-OUT after "
                   f"{timeout_seconds}s, forcibly terminated.")
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join()

    minima_traj_path = work_dir / f"minima_seed{seed_idx}.traj"
    results = []
    if not minima_traj_path.exists():
        return results
    try:
        found = read(minima_traj_path, index=":")
    except Exception as exc:
        log.warning(f"    Could not read {minima_traj_path}: {exc}")
        return results

    calc = load_dftb_calculator(slako_dir, charge=charge,
                                label=str(work_dir / "reread"))
    for m_idx, m_atoms in enumerate(found):
        m_atoms.calc = calc
        try:
            energy = m_atoms.get_potential_energy()
        except Exception:
            continue
        integ = check_structural_integrity(m_atoms.get_positions())
        results.append({
            "atoms": m_atoms, "energy_eV": energy,
            "seed_idx": seed_idx, "minimum_idx_in_run": m_idx,
            "integrity": integ,
        })
    return results


def deduplicate_minima(all_minima, energy_tol=1e-3):
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


def run_mh_mode(n_min, n_max, n_seeds, n_steps, charge, slako_dir,
                 seed_timeout, seed, out, writer, csv_file, e_n2_per_atom):
    sizes = list(range(n_min, n_max + 1, 2))

    for n in sizes:
        log.info("=" * 50)
        log.info(f"N{n}{'+' if charge>0 else '-'}: generating {n_seeds} seed topologies...")
        seeds = seed_topologies_for_mh(n, n_seeds, seed=seed + n)
        log.info(f"  {len(seeds)} seeds (chain/ring/random mix)")

        n_work_dir = out / "mh_runs" / f"N{n}"
        all_minima_this_n = []
        for s_idx, G_seed in enumerate(seeds):
            log.info(f"  seed {s_idx+1}/{len(seeds)}: running MH "
                     f"({n_steps} steps, timeout {seed_timeout}s)...")
            results = run_mh_seed(
                G_seed, n, charge, slako_dir, n_steps, s_idx,
                n_work_dir / f"seed{s_idx}", seed_timeout,
                rng_seed=seed + n * 100 + s_idx,
            )
            log.info(f"    -> {len(results)} minima recorded")
            all_minima_this_n.extend(results)

        unique_minima = deduplicate_minima(all_minima_this_n)
        log.info(f"N{n}: {len(all_minima_this_n)} total, "
                 f"{len(unique_minima)} unique after dedup")
        unique_minima.sort(key=lambda m: m["energy_eV"])

        for rank, m in enumerate(unique_minima):
            integ = m["integrity"]
            label = classify_topology(integ["final_graph"])
            e_per_atom = m["energy_eV"] / n
            delta_e = e_per_atom - e_n2_per_atom if e_n2_per_atom is not None else None

            stem = f"mh_N{n}{'+' if charge>0 else '-'}_rank{rank:03d}_{label}_seed{m['seed_idx']}"
            comment = (
                f"n={n} charge={charge:+d} topology={label} "
                f"integrity={integ['status']} energy_eV={m['energy_eV']:.6f} "
                f"method=SCC-DFTB(mio-1-1)"
            )
            xyz_path = out / "xyz" / f"{stem}.xyz"
            cif_path = out / "cif" / f"{stem}.cif"
            write_xyz(m["atoms"], xyz_path, comment=comment)
            write_cif(m["atoms"], cif_path)

            writer.writerow({
                "mode": "mh", "n_atoms": n, "charge": charge,
                "topology_label": label, "energy_eV": f"{m['energy_eV']:.6f}",
                "energy_eV_per_atom": f"{e_per_atom:.6f}",
                "delta_E_eV_per_atom_vs_N2": f"{delta_e:.4f}" if delta_e is not None else "",
                "integrity_status": integ["status"],
                "xyz_file": str(xyz_path.relative_to(out)),
                "cif_file": str(cif_path.relative_to(out)),
            })
        csv_file.flush()


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT WRITERS
# ─────────────────────────────────────────────────────────────────────────────

def write_xyz(atoms, filepath, comment=""):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    n = len(atoms)
    with open(filepath, "w") as f:
        f.write(f"{n}\n{comment}\n")
        for sym, pos in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
            f.write(f"{sym}  {pos[0]:.8f}  {pos[1]:.8f}  {pos[2]:.8f}\n")


def write_cif(atoms, filepath, cell_size=30.0):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    a = atoms.copy()
    a.set_cell([cell_size] * 3)
    a.center()
    a.set_pbc(True)
    with tempfile.NamedTemporaryFile(suffix=".cif", delete=False, mode="w") as tmp:
        tmp_path = tmp.name
    write(tmp_path, a, format="cif")
    os.replace(tmp_path, filepath)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mode", type=str, required=True, choices=["airss", "mh"])
    p.add_argument("--slako_dir", type=str, required=True)
    p.add_argument("--charge", type=int, default=-1,
                   help="Total charge in units of e. Default -1 (anion). "
                        "Use +1 for cations (then n should be ODD for the "
                        "same even-electron-count reasoning, mirrored).")
    p.add_argument("--n_min", type=int, default=3)
    p.add_argument("--n_max", type=int, default=7)
    p.add_argument("--n_trials", type=int, default=20,
                   help="(airss mode) total random trials across all sizes")
    p.add_argument("--size_weight_power", type=float, default=2.0,
                   help="(airss mode) exponent applied to (3n-6) -- the "
                        "number of internal molecular degrees of freedom "
                        "-- when distributing --n_trials across sizes. "
                        "Default 2.0 (quadratic): larger n receive "
                        "disproportionately more trials, reflecting their "
                        "larger conformational space (e.g. for n_min=3, "
                        "n_max=7, N7 gets ~25x as many trials as N3 with "
                        "the default power=2, vs 5x with power=1 linear, "
                        "vs equal counts with power=0 = old uniform "
                        "behavior). See compute_size_weights docstring "
                        "for the full rationale.")
    p.add_argument("--dftb_timeout", type=int, default=120,
                   help="(airss mode) hard wall-clock timeout in seconds "
                        "per DFTB+ relaxation attempt, enforced via a "
                        "killable subprocess. Protects against a DFTB+ "
                        "SCC cycle hanging indefinitely without crashing "
                        "or returning -- observed in production to "
                        "silently consume 12+ minutes on a single trial "
                        "with no error message (see "
                        "relax_with_dftb_timeout docstring). Default 120.")
    p.add_argument("--n_seeds", type=int, default=3,
                   help="(mh mode) independent MH runs per size")
    p.add_argument("--n_steps", type=int, default=15,
                   help="(mh mode) MH steps per run")
    p.add_argument("--seed_timeout", type=int, default=MH_SEED_TIMEOUT)
    p.add_argument("--fmax", type=float, default=0.05)
    p.add_argument("--max_steps", type=int, default=300)
    p.add_argument("--output_dir", type=str, default="charged_explore_results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip_sanity_check", action="store_true")
    args = p.parse_args()

    if args.mode == "mh" and not HAVE_MH:
        sys.exit("ERROR: ASE's MinimaHopping/Hookean modules not available.")

    check_dftb_available()

    n_min, n_max = args.n_min, args.n_max
    if n_min % 2 == 0:
        n_min += 1
        log.warning(f"--n_min bumped to {n_min} (must be odd for charge "
                   f"{args.charge:+d} to give an even electron count)")
    if n_max % 2 == 0:
        n_max -= 1

    out = Path(args.output_dir)
    for sub in ("xyz", "cif", "mh_runs"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    e_n2_per_atom = None
    if not args.skip_sanity_check:
        ok, e_n2_per_atom = sanity_check_n2(args.slako_dir)
        if not ok:
            sys.exit(1)

    csv_path = out / "charged_explore_results.csv"
    fieldnames = [
        "mode", "n_atoms", "charge", "topology_label", "energy_eV",
        "energy_eV_per_atom", "delta_E_eV_per_atom_vs_N2",
        "integrity_status", "xyz_file", "cif_file",
    ]
    csv_file = open(csv_path, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    log.info(f"Mode: {args.mode} | charge: {args.charge:+d} | "
             f"n: {n_min}..{n_max} (odd)")

    if args.mode == "airss":
        run_airss_mode(
            n_min, n_max, args.n_trials, args.charge, args.slako_dir,
            args.fmax, args.max_steps, args.seed, out, writer, csv_file,
            e_n2_per_atom, size_weight_power=args.size_weight_power,
            dftb_timeout=args.dftb_timeout,
        )
    else:
        run_mh_mode(
            n_min, n_max, args.n_seeds, args.n_steps, args.charge,
            args.slako_dir, args.seed_timeout, args.seed, out, writer,
            csv_file, e_n2_per_atom,
        )

    csv_file.close()
    log.info("=" * 60)
    log.info(f"DONE. Results in '{args.output_dir}/'")
    log.info(f"  {csv_path}")


if __name__ == "__main__":
    main()
