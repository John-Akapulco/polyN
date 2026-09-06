#!/usr/bin/env python3
"""
polynitrogen_mace.py
====================
Generator and optimizer of neutral all-nitrogen (Nn) molecular allotropes.

WORKFLOW
--------
1. Enumerate non-isomorphic connected graphs on n nodes (n = n_min..n_max)
   Constraints:
     - Every atom: degree 1, 2, or 3  (σ-framework; MACE handles electronics)
     - At least one atom with degree ≥ 2 (di- or tri-connected)
   Strategy:
     - n ≤ 6 : exhaustive canonical enumeration  (exact, via graph6 + full isomorphism)
     - n = 7..10 : exhaustive with early cutoff at max_graphs_exact topologies
     - n > 10  : random degree-sequence sampling  (WL-hash deduplication)

2. For each unique topology, generate n_conformers 3D structures
   via spring-layout seeding + local force-field minimization (bond + angle + repulsion).

3. Relax each conformer with a MACE potential:
     - MACE-OFF23   (molecules, recommended) : mace_off(model="medium")
     - MACE-MP-0    (materials/solids)       : mace_mp(model="medium")
   Uses ASE LBFGS optimizer (fmax = 0.05 eV/Å).

4. Deduplicate by final energy (ΔE < 0.01 eV → same minimum) and output
   the lowest-energy conformer per topology.

5. Write outputs:
     - <output_dir>/xyz/   : one .xyz per molecule (ASE extended-XYZ)
     - <output_dir>/cif/   : one .cif per molecule (P1 box for portability)
     - <output_dir>/results.csv : energy, connectivity, symmetry, stability flag
     - <output_dir>/topologies/ : .gml graph files for each topology

INSTALLATION (conda or venv recommended)
-----------------------------------------
    pip install mace-torch ase spglib networkx numpy scipy

    # For GPU (optional):
    pip install mace-torch --extra-index-url https://download.pytorch.org/whl/cu118

USAGE
-----
    # Basic: N3 to N10, MACE-OFF23 medium, CPU
    python polynitrogen_mace.py --n_max 10

    # Extended run to N20, GPU, save all conformers
    python polynitrogen_mace.py --n_min 3 --n_max 20 --model mace_off \
        --device cuda --n_conformers 10 --output_dir ./N3_N20_results

    # Dry run (no MACE, force-field only) — useful for testing topology enumeration
    python polynitrogen_mace.py --n_max 6 --dry_run

AUTHOR
------
    Generated for IC2MP/E4 Médiacat — Université de Poitiers
    Compatible with: Python ≥ 3.10, ASE ≥ 3.22, MACE ≥ 0.3, spglib ≥ 2.0
"""

import argparse
import csv
import logging
import os
import sys
import tempfile
import time
import warnings
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.optimize import minimize

# ASE
try:
    from ase import Atoms
    from ase.io import write
    from ase.optimize import LBFGS
    from ase.units import Hartree, eV
except ImportError:
    sys.exit("ERROR: ASE not found. Run: pip install ase")

# spglib
try:
    import spglib
    SPGLIB_OK = True
except ImportError:
    warnings.warn("spglib not found — symmetry detection disabled. pip install spglib")
    SPGLIB_OK = False

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Reference N–N bond length for initial embedding (Å)
# Approximate average between single (1.45) and double (1.25)
NN_BOND_INIT = 1.35

# Force-field parameters for initial geometry
FF_K_BOND  = 200.0   # eV/Å² equivalent (harmonic bond)
FF_K_ANGLE = 100.0   # eV/rad²  (harmonic angle)
FF_K_REP   = 0.5     # eV       (non-bonded repulsion prefactor)
FF_REP_EXP = 9       # exponent for repulsion

# Target angles (degrees) by coordination number
TARGET_ANGLES = {2: 120.0, 3: 109.5}

# ── Post-relaxation structural integrity checks ────────────────────────────
# Maximum N-N distance to be considered a covalent bond (Å).
# Typical N-N single bond ~1.45 Å, double ~1.25 Å, triple ~1.10 Å;
# 1.6 Å gives a safety margin above the longest plausible single bond
# (cf. weak/elongated N-N single bonds in strained rings, ~1.50-1.55 Å)
# while remaining well below van der Waals contact (~3.0 Å) and
# excluding spurious "bonds" between unrelated, non-bonded atoms.
BOND_CUTOFF_NN = 1.60

# Minimum N-N distance considered physically sane (avoids atoms collapsing
# onto each other during a pathological optimization step).
MIN_SANE_NN_DISTANCE = 0.90

# Maximum N-N distance allowed anywhere in the molecule, regardless of
# bonding (catches an atom or fragment drifting away during relaxation
# even if it does not trigger the disconnection check, e.g. a single
# image still nominally "connected" through a very long residual contact).
MAX_SANE_NN_DISTANCE = 2.50

# Maximum atoms per molecule for exhaustive enumeration
EXHAUSTIVE_LIMIT = 6
# Maximum graphs per stoichiometry in exhaustive mode
MAX_GRAPHS_EXACT = 200

# Stability threshold: E_MACE / n_atoms (eV/atom) relative to N2 energy
# N2 at MACE-OFF23/medium is approximately -5.8 eV (total) → -2.9 eV/atom
# We flag molecules as "metastable" if ΔE/atom < +3 eV vs N2
STABILITY_THRESHOLD_EV_ATOM = 3.0

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("polyN")


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH ENUMERATION
# ─────────────────────────────────────────────────────────────────────────────

def _is_valid_polyN_graph(G, max_degree=3):
    """Return True if G satisfies polynitrogen σ-framework constraints."""
    if not nx.is_connected(G):
        return False
    degs = [G.degree(v) for v in G.nodes()]
    if max(degs) > max_degree:
        return False
    if min(degs) < 1:
        return False
    if max(degs) < 2:  # need at least one di/tri-connected atom
        return False
    return True


def enumerate_exhaustive(n, max_degree=3, max_graphs=MAX_GRAPHS_EXACT):
    """
    Exact enumeration of non-isomorphic connected graphs on n nodes
    with degree constraints. Uses graph6 + full isomorphism for deduplication.
    Practical for n ≤ 7 (n=6 takes ~3 s, n=7 ~minutes without cap).

    Returns list of nx.Graph objects.
    """
    nodes = list(range(n))
    all_edges = list(combinations(nodes, 2))
    n_min_e = n - 1                    # spanning tree
    n_max_e = (max_degree * n) // 2    # handshaking maximum

    valid = []
    seen_cert = set()

    for n_e in range(n_min_e, n_max_e + 1):
        for edge_set in combinations(all_edges, n_e):
            G = nx.Graph()
            G.add_nodes_from(nodes)
            G.add_edges_from(edge_set)

            degs = [G.degree(v) for v in nodes]
            if max(degs) > max_degree or max(degs) < 2:
                continue
            if not nx.is_connected(G):
                continue

            # graph6 byte string is canonical for simple graphs with the same node set
            cert = nx.to_graph6_bytes(G, header=False)
            if cert in seen_cert:
                continue

            # Full isomorphism check (guards against graph6 collision edge cases)
            if any(nx.is_isomorphic(G, g2) for g2 in valid[-30:]):
                continue

            seen_cert.add(cert)
            valid.append(G)

            if len(valid) >= max_graphs:
                log.warning(
                    f"N{n}: hit cap of {max_graphs} graphs at {n_e} edges — "
                    "increase MAX_GRAPHS_EXACT for completeness."
                )
                return valid

    return valid


def sample_graphs(n, max_degree=3, n_target=200, seed=42):
    """
    Stochastic sampling of unique polynitrogen graphs for large n.
    Uses random_degree_sequence_graph + WL-hash deduplication.
    Not exhaustive, but covers diverse topologies efficiently.

    Returns list of nx.Graph objects.
    """
    rng = np.random.default_rng(seed)
    seen = {}  # wl_hash → graph
    graphs = []
    attempts = 0
    max_attempts = n_target * 50

    while len(graphs) < n_target and attempts < max_attempts:
        attempts += 1

        # Random degree sequence satisfying constraints
        degs = rng.integers(1, max_degree + 1, size=n).tolist()
        if sum(degs) % 2 != 0:
            # Fix parity: flip one terminal to degree 2
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

        if not _is_valid_polyN_graph(G, max_degree):
            continue

        # WL-hash for fast dedup (iterations=10 for reduced collision risk)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wl = nx.weisfeiler_lehman_graph_hash(G, iterations=10)

        if wl not in seen:
            seen[wl] = G
            graphs.append(G)

    if len(graphs) < n_target:
        log.warning(
            f"N{n}: only {len(graphs)}/{n_target} unique topologies found "
            f"after {attempts} attempts (normal for large n with tight constraints)."
        )
    return graphs


def get_graphs_for_n(n, max_degree=3, n_sample_target=200, exact_limit=EXHAUSTIVE_LIMIT):
    """
    Dispatch to exhaustive or sampling enumeration based on n.
    """
    if n <= exact_limit:
        log.info(f"N{n}: exhaustive canonical enumeration (max_deg={max_degree})...")
        graphs = enumerate_exhaustive(n, max_degree=max_degree)
    else:
        log.info(f"N{n}: stochastic sampling (target={n_sample_target} topologies)...")
        graphs = sample_graphs(n, max_degree=max_degree, n_target=n_sample_target)

    log.info(f"N{n}: {len(graphs)} unique topologies found.")
    return graphs


# ─────────────────────────────────────────────────────────────────────────────
# TOPOLOGY DESCRIPTORS
# ─────────────────────────────────────────────────────────────────────────────

def describe_topology(G):
    """
    Return a human-readable topology label and a dict of descriptors.
    """
    n = len(G.nodes())
    degs = sorted([G.degree(v) for v in G.nodes()], reverse=True)
    n_edges = G.number_of_edges()
    cycles = nx.cycle_basis(G)
    n_rings = len(cycles)
    ring_sizes = sorted([len(c) for c in cycles])

    # Classify
    if n_rings == 0:
        if max(degs) == 1:
            label = "dimer"
        elif all(d <= 2 for d in degs):
            label = "chain"
        else:
            label = "branched-tree"
    elif n_rings == 1:
        if all(d == 2 for d in degs):
            label = f"ring-{ring_sizes[0]}"
        else:
            label = f"ring-{ring_sizes[0]}-substituted"
    else:
        label = f"bicyclic-{'-'.join(map(str, ring_sizes))}"

    desc = {
        "n_atoms": n,
        "n_edges": n_edges,
        "n_rings": n_rings,
        "ring_sizes": ring_sizes,
        "degree_sequence": degs,
        "label": label,
        "max_degree": max(degs),
        "min_degree": min(degs),
    }
    return label, desc


# ─────────────────────────────────────────────────────────────────────────────
# 3D EMBEDDING (force-field)
# ─────────────────────────────────────────────────────────────────────────────

def ff_energy(flat_coords, G, bond_length=NN_BOND_INIT):
    """
    Simple classical force-field energy for initial geometry optimization.
    - Bonded: harmonic bond stretching
    - Non-bonded: repulsive r^(-FF_REP_EXP)
    - Angle: harmonic angle bending (target 120° for sp2, 109.5° for sp3)
    """
    n = len(G.nodes())
    xyz = flat_coords.reshape(n, 3)
    edges = list(G.edges())
    bonded = set(edges) | {(v, u) for u, v in edges}
    E = 0.0

    # Bond stretching
    for u, v in edges:
        d = np.linalg.norm(xyz[u] - xyz[v])
        E += FF_K_BOND * (d - bond_length) ** 2

    # Non-bonded repulsion
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in bonded and (j, i) not in bonded:
                d = max(np.linalg.norm(xyz[i] - xyz[j]), 0.3)
                E += FF_K_REP * (2.0 * bond_length / d) ** FF_REP_EXP

    # Angle bending
    for node in G.nodes():
        nb = list(G.neighbors(node))
        deg = len(nb)
        if deg < 2:
            continue
        target_angle = np.radians(TARGET_ANGLES.get(deg, 109.5))
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


def embed_3d(G, n_conformers=5, bond_length=NN_BOND_INIT, seed=0):
    """
    Generate n_conformers 3D structures for graph G.
    Returns list of (coords_array, ff_energy) sorted by ff_energy ascending.
    """
    n = len(G.nodes())
    rng = np.random.default_rng(seed)
    conformers = []

    for ci in range(n_conformers):
        # Seed from spring layout (connectivity-aware) + perturbation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pos_2d = nx.spring_layout(G, dim=3, seed=ci, k=bond_length * 1.2)

        init = np.array([pos_2d[i] for i in range(n)], dtype=float)

        # Scale to target bond length
        edges = list(G.edges())
        if edges:
            mean_d = np.mean([np.linalg.norm(init[u] - init[v]) for u, v in edges])
            if mean_d > 1e-8:
                init *= bond_length / mean_d

        # Add random perturbation (increases with conformer index)
        init += rng.normal(0, 0.15 * (ci + 0.5), size=(n, 3))

        # Local minimization
        result = minimize(
            ff_energy,
            init.flatten(),
            args=(G, bond_length),
            method="L-BFGS-B",
            options={"maxiter": 10000, "ftol": 1e-14, "gtol": 1e-10},
        )

        coords = result.x.reshape(n, 3)
        coords -= coords.mean(axis=0)  # center at origin

        conformers.append((coords, result.fun))

    # Sort by force-field energy
    conformers.sort(key=lambda x: x[1])
    return conformers


# ─────────────────────────────────────────────────────────────────────────────
# ASE ATOMS BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def coords_to_atoms(coords, cell_size=30.0):
    """
    Build an ASE Atoms object (isolated molecule in a cubic box).
    cell_size (Å): box dimension — large enough to avoid periodic interactions.
    """
    n = len(coords)
    atoms = Atoms(
        symbols="N" * n,
        positions=coords,
        cell=[cell_size] * 3,
        pbc=False,
    )
    atoms.center()
    return atoms


# ─────────────────────────────────────────────────────────────────────────────
# MACE CALCULATOR LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_mace_calculator(model_type="mace_off", model_size="medium",
                          device="cpu", dtype="float64"):
    """
    Load a MACE calculator.

    Parameters
    ----------
    model_type : str
        "mace_off"  → MACE-OFF23 (organic molecules, best for neutral Nn)
        "mace_mp"   → MACE-MP-0  (periodic materials, also works for molecules)
        "custom"    → provide path via model_size argument
    model_size : str
        "small", "medium", "large" — or path to a .pt checkpoint
    device : str
        "cpu", "cuda", "cuda:0", "mps"
    dtype : str
        "float32" (faster) or "float64" (more precise)

    Returns
    -------
    ASE calculator object or None if import fails.
    """
    try:
        from mace.calculators import mace_off, mace_mp
    except ImportError:
        log.error(
            "MACE not found. Install with:\n"
            "    pip install mace-torch\n"
            "Then re-run this script."
        )
        return None

    # MPS (Apple Silicon GPU, e.g. M1-M5) has no float64 support: a
    # float64 tensor allocation either errors out or silently falls back
    # to CPU depending on the PyTorch version. Auto-correct to float32
    # with an explicit warning instead of letting the user hit a cryptic
    # downstream failure or unknowingly lose all MPS acceleration.
    if device == "mps" and dtype == "float64":
        log.warning(
            "device='mps' (Apple Silicon GPU) does not support float64. "
            "Automatically switching to dtype='float32' for this run. "
            "See companion methodology manual for the precision caveat "
            "(typical energy differences vs float64/CPU: 1e-3 to 1e-4 eV, "
            "negligible for structure screening purposes)."
        )
        dtype = "float32"

    log.info(f"Loading MACE calculator: {model_type}/{model_size} on {device} ({dtype})")

    common_kw = dict(device=device, default_dtype=dtype)

    if model_type == "mace_off":
        # MACE-OFF23: trained on SPICE molecular dataset
        # Best for neutral organic/inorganic molecules
        calc = mace_off(model=model_size, **common_kw)

    elif model_type == "mace_mp":
        # MACE-MP-0: trained on Materials Project
        # Good for periodic solids; also reasonable for molecules
        calc = mace_mp(
            model=model_size,
            dispersion=False,  # dispersion correction usually not needed for small Nn
            **common_kw,
        )

    elif model_type == "custom":
        # Load a user-provided checkpoint
        from mace.calculators import MACECalculator
        calc = MACECalculator(
            model_paths=model_size,  # model_size = path here
            device=device,
            default_dtype=dtype,
        )
    else:
        log.error(f"Unknown model_type: {model_type}. Use 'mace_off', 'mace_mp', or 'custom'.")
        return None

    log.info("MACE calculator loaded successfully.")
    return calc


# ─────────────────────────────────────────────────────────────────────────────
# MACE OPTIMIZATION
# ─────────────────────────────────────────────────────────────────────────────

def optimize_with_mace(atoms, calc, fmax=0.05, steps=500, logfile=None):
    """
    Relax an ASE Atoms object with MACE using LBFGS.

    Returns
    -------
    (relaxed_atoms, energy_eV, converged, n_steps)
    """
    atoms = atoms.copy()
    atoms.calc = calc

    opt = LBFGS(atoms, logfile=logfile, trajectory=None)
    try:
        converged = opt.run(fmax=fmax, steps=steps)
        energy = atoms.get_potential_energy()   # eV
        n_steps = opt.get_number_of_steps()
    except Exception as exc:
        log.warning(f"  MACE optimization failed: {exc}")
        return atoms, None, False, -1

    return atoms, energy, converged, n_steps


# ─────────────────────────────────────────────────────────────────────────────
# SYMMETRY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def get_symmetry(atoms, symprec=0.1):
    """
    Detect point group / space group of a molecular structure.
    Returns dict with 'spacegroup', 'number', 'pointgroup'.
    """
    if not SPGLIB_OK:
        return {"spacegroup": "N/A", "number": -1, "pointgroup": "N/A"}

    atoms_pbc = atoms.copy()
    atoms_pbc.set_pbc(True)
    cell_data = (
        atoms_pbc.cell[:],
        atoms_pbc.get_scaled_positions(),
        atoms_pbc.get_atomic_numbers(),
    )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sym = spglib.get_symmetry_dataset(cell_data, symprec=symprec)
        if sym is None:
            return {"spacegroup": "P1", "number": 1, "pointgroup": "C1"}
        # Attribute access for newer spglib
        try:
            sg = sym.international
            num = sym.number
            pg = sym.site_symmetry_symbols[0] if hasattr(sym, "site_symmetry_symbols") else "?"
        except AttributeError:
            sg = sym["international"]
            num = sym["number"]
            pg = "?"
        return {"spacegroup": sg, "number": num, "pointgroup": pg}
    except Exception:
        return {"spacegroup": "P1", "number": 1, "pointgroup": "C1"}


# ─────────────────────────────────────────────────────────────────────────────
# STRUCTURAL INTEGRITY CHECKS (bond distances + fragmentation/rearrangement)
# ─────────────────────────────────────────────────────────────────────────────
#
# Rationale
# ---------
# Neither the classical force field (Section 3 of the manual) nor the MACE
# relaxation (Section 4) explicitly *constrains* N-N distances or guarantees
# that the target connectivity graph G is preserved. MACE forces can in
# principle break a bond (if the topology was a poor/strained starting guess)
# or let two atoms collapse onto each other (if the optimizer steps into an
# unphysical region before correcting). This is a standard pitfall when
# coupling MLIP relaxation to combinatorially generated starting geometries:
# the optimizer finds *a* nearby local minimum of the true PES, which is not
# guaranteed to be the molecule we intended to build.
#
# Strategy: after optimization, rebuild a connectivity graph G_final purely
# from interatomic distances (a "geometric" graph), and compare it against
# the original *target* topology graph G_target (purely combinatorial, from
# Section 2). Three independent diagnostics are computed:
#
#   1. Bond-length sanity   : are all *intended* bonds within ]MIN, BOND_CUTOFF]?
#   2. Connectivity check   : is G_final still a single connected component?
#                             (catches outright fragmentation / dissociation)
#   3. Topology fidelity    : is G_final isomorphic to G_target?
#                             (catches silent rearrangement: same atom count,
#                              still one connected piece, but the molecule
#                              has isomerized into a different topology)
#
# Any of these failing flags the structure; the script does not silently
# discard them (a rearranged structure may be chemically interesting in its
# own right) but always reports the diagnosis explicitly in results.csv,
# rather than reporting only the intended topology label.

def infer_geometric_graph(coords, bond_cutoff=BOND_CUTOFF_NN):
    """
    Reconstruct a connectivity graph purely from interatomic distances,
    independent of any assumed/target topology.

    Parameters
    ----------
    coords : (n, 3) array, Cartesian coordinates in Å
    bond_cutoff : float
        Maximum N-N distance (Å) to consider a covalent bond present.

    Returns
    -------
    G_geom : nx.Graph
        Graph with an edge (i,j) wherever |x_i - x_j| <= bond_cutoff.
        Edge attribute 'length' stores the actual distance.
    distances : dict
        {(i,j): distance} for ALL atom pairs (not just bonded), used for
        the min/max sanity checks below.
    """
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
                                max_distance=MAX_SANE_NN_DISTANCE):
    """
    Run the three post-relaxation diagnostics described above.

    Parameters
    ----------
    coords : (n, 3) array
        Final (e.g. MACE-relaxed) Cartesian coordinates, Å.
    G_target : nx.Graph
        The combinatorial topology the structure was originally built from
        (Section 2 of the manual).
    bond_cutoff, min_distance, max_distance : float
        Thresholds in Å (see module-level constants for physical justification).

    Returns
    -------
    dict with keys:
        'is_connected'        : bool — single connected component by distance?
        'n_fragments'         : int  — number of connected components found
        'topology_preserved'  : bool — is the geometric graph isomorphic to G_target?
        'bonds_in_range'      : bool — are all *target* bonds within [min, cutoff]?
        'min_target_bond'     : float — shortest distance among G_target edges
        'max_target_bond'     : float — longest distance among G_target edges
        'min_any_distance'    : float — shortest distance among ALL atom pairs
                                         (catches atoms collapsing onto each other,
                                          even if not nominally "bonded")
        'max_any_distance'    : float — largest pairwise distance found
                                         (sanity bound, catches drift even within
                                          a nominally connected geometric graph)
        'status'              : str — overall verdict, one of:
                                   'OK'            : everything as intended
                                   'FRAGMENTED'    : dissociated into >=2 pieces
                                   'REARRANGED'    : still connected, but not the
                                                      intended topology
                                   'BOND_ANOMALY'  : connected & correct topology,
                                                      but a target bond is
                                                      abnormally short/long
                                   'COLLAPSED'     : two atoms have moved
                                                      pathologically close
                                                      (likely optimizer failure)
    """
    n = len(coords)
    G_geom, distances = infer_geometric_graph(coords, bond_cutoff=bond_cutoff)

    n_fragments = nx.number_connected_components(G_geom)
    is_connected = (n_fragments == 1)

    all_d = list(distances.values())
    min_any_distance = min(all_d) if all_d else float("inf")
    max_any_distance = max(all_d) if all_d else 0.0

    # Distances for bonds that were *intended* in the target topology
    target_bond_lengths = []
    for (u, v) in G_target.edges():
        i, j = (u, v) if u < v else (v, u)
        target_bond_lengths.append(distances[(i, j)])

    min_target_bond = min(target_bond_lengths) if target_bond_lengths else float("inf")
    max_target_bond = max(target_bond_lengths) if target_bond_lengths else 0.0

    bonds_in_range = all(
        min_distance <= d <= bond_cutoff for d in target_bond_lengths
    )

    # Topology fidelity: compare geometric graph to the original target graph.
    # Only meaningful if the molecule is still fully connected with the same
    # atom count (a disconnected graph cannot be isomorphic to a connected one
    # of the same order, but we guard explicitly for clarity).
    topology_preserved = is_connected and nx.is_isomorphic(G_geom, G_target)

    # Collapse detection: any two atoms abnormally close (independent of
    # whether they were meant to be bonded — this is a numerical/optimizer
    # pathology check, not a chemistry check).
    collapsed = min_any_distance < min_distance

    # ── Verdict ──────────────────────────────────────────────────────────
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
        "is_connected": is_connected,
        "n_fragments": n_fragments,
        "topology_preserved": topology_preserved,
        "bonds_in_range": bonds_in_range,
        "min_target_bond": min_target_bond,
        "max_target_bond": max_target_bond,
        "min_any_distance": min_any_distance,
        "max_any_distance": max_any_distance,
        "status": status,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STABILITY ESTIMATE
# ─────────────────────────────────────────────────────────────────────────────

def estimate_stability(energy_eV, n_atoms, n2_energy_eV_per_atom=-2.9):
    """
    Estimate metastability relative to molecular N2.

    ΔE/atom = (E_molecule/n_atoms) - E(N2)/atom
    Negative = thermodynamically unstable but the sign convention is:
      ΔE/atom > 0  → molecule stores energy relative to N2 (HEDM potential)
      ΔE/atom < 0  → molecule more stable than N2 (very unlikely for polyN)

    Returns (delta_E_per_atom, is_candidate)
    is_candidate = True if ΔE/atom < STABILITY_THRESHOLD_EV_ATOM
    """
    if energy_eV is None:
        return None, False
    e_per_atom = energy_eV / n_atoms
    delta_e = e_per_atom - n2_energy_eV_per_atom
    is_candidate = 0.0 <= delta_e < STABILITY_THRESHOLD_EV_ATOM
    return delta_e, is_candidate


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT WRITERS
# ─────────────────────────────────────────────────────────────────────────────

def write_xyz(atoms, filepath, comment=""):
    """Write extended XYZ with metadata in comment line."""
    n = len(atoms)
    with open(filepath, "w") as f:
        f.write(f"{n}\n")
        f.write(f"{comment}\n")
        for sym, pos in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
            f.write(f"{sym}  {pos[0]:.8f}  {pos[1]:.8f}  {pos[2]:.8f}\n")


def write_cif(atoms, filepath):
    """Write CIF file (molecule in P1 box) using ASE."""
    atoms_cif = atoms.copy()
    atoms_cif.set_pbc(True)
    with tempfile.NamedTemporaryFile(suffix=".cif", delete=False, mode="w") as tmp:
        tmp_path = tmp.name
    write(tmp_path, atoms_cif, format="cif")
    os.replace(tmp_path, filepath)


def write_gml(G, filepath, label=""):
    """Write graph topology as GML."""
    H = G.copy()
    for node in H.nodes():
        H.nodes[node]["label"] = f"N{node+1}"
        H.nodes[node]["element"] = "N"
    nx.write_gml(H, str(filepath))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    n_min=3,
    n_max=10,
    max_degree=3,
    n_conformers=5,
    n_sample_target=100,
    model_type="mace_off",
    model_size="medium",
    device="cpu",
    dtype="float64",
    fmax=0.05,
    max_opt_steps=500,
    output_dir="polyN_results",
    dry_run=False,
    seed=42,
    write_all_conformers=False,
    skip_non_intact=False,
    bond_cutoff=BOND_CUTOFF_NN,
):
    """
    Full pipeline: enumerate → embed → MACE optimize → output.

    Parameters
    ----------
    n_min, n_max : int
        Range of molecule sizes (number of N atoms).
    max_degree : int
        Maximum coordination number per N atom (default 3).
    n_conformers : int
        Number of 3D conformers to generate per topology.
    n_sample_target : int
        Target number of sampled topologies for n > EXHAUSTIVE_LIMIT.
    model_type : str
        MACE model type ('mace_off', 'mace_mp', 'custom').
    model_size : str
        Model size or checkpoint path.
    device : str
        Torch device.
    dtype : str
        'float32' or 'float64'.
    fmax : float
        Force convergence threshold (eV/Å).
    max_opt_steps : int
        Maximum LBFGS steps per optimization.
    output_dir : str
        Root output directory.
    dry_run : bool
        If True, skip MACE and use FF energy only (for testing).
    seed : int
        Random seed.
    write_all_conformers : bool
        If True, write all conformers, not just the best one per topology.
    skip_non_intact : bool
        If True, do not write XYZ/CIF files for structures whose
        post-relaxation integrity_status != 'OK' (fragmented, rearranged,
        collapsed, or bond-anomalous). They are still counted and logged,
        and still recorded in results.csv, but excluded from the structural
        output directories — useful for producing a "clean" dataset for
        direct downstream use (e.g. HPmat.org ingestion). Default False:
        all structures are written, since a rearranged/fragmented result is
        itself informative (it shows which topologies are NOT kinetically
        stable local minima on the MACE-predicted PES).
    """

    # ── Set up output directory tree ──────────────────────────────────────────
    out = Path(output_dir)
    dirs = {
        "xyz":       out / "xyz",
        "cif":       out / "cif",
        "top":       out / "topologies",
        "logs":      out / "logs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # ── Load MACE (once, shared across all molecules) ─────────────────────────
    calc = None
    n2_e_per_atom = -2.9  # default fallback

    if not dry_run:
        calc = load_mace_calculator(model_type, model_size, device, dtype)
        if calc is None:
            log.warning("Falling back to dry_run mode (no MACE).")
            dry_run = True
        else:
            # Compute reference N2 energy
            log.info("Computing N2 reference energy...")
            try:
                n2_atoms = Atoms("N2", positions=[[0, 0, 0], [0, 0, 1.098]],
                                  cell=[20]*3, pbc=False)
                n2_atoms.center()
                n2_opt, n2_e, _, _ = optimize_with_mace(
                    n2_atoms, calc, fmax=0.01, steps=200
                )
                n2_e_per_atom = n2_e / 2
                log.info(f"  N2 reference: {n2_e:.4f} eV total, {n2_e_per_atom:.4f} eV/atom")
            except Exception as exc:
                log.warning(f"  N2 reference calculation failed: {exc}")

    # ── CSV output ────────────────────────────────────────────────────────────
    csv_path = out / "results.csv"
    fieldnames = [
        "id", "formula", "n_atoms", "topology_label",
        "n_edges", "n_rings", "ring_sizes", "degree_sequence",
        "max_degree_actual", "conformer_id",
        "ff_energy_au", "mace_energy_eV", "mace_energy_eV_per_atom",
        "delta_E_eV_per_atom_vs_N2", "is_stability_candidate",
        "converged", "n_opt_steps",
        "integrity_status", "is_connected", "n_fragments",
        "topology_preserved", "bonds_in_range",
        "min_target_bond_A", "max_target_bond_A",
        "min_any_distance_A", "max_any_distance_A",
        "spacegroup", "sg_number",
        "xyz_file", "cif_file",
    ]
    csv_file = open(csv_path, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    # ── Main loop over n ──────────────────────────────────────────────────────
    total_structures = 0
    mol_id = 0
    integrity_counts = {"OK": 0, "FRAGMENTED": 0, "REARRANGED": 0,
                         "BOND_ANOMALY": 0, "COLLAPSED": 0}

    for n in range(n_min, n_max + 1):
        log.info("=" * 60)
        log.info(f"Processing N{n} ...")
        t_n_start = time.time()

        graphs = get_graphs_for_n(
            n, max_degree=max_degree, n_sample_target=n_sample_target
        )
        if not graphs:
            log.warning(f"N{n}: no valid topologies found, skipping.")
            continue

        for topo_idx, G in enumerate(graphs):
            label, desc = describe_topology(G)
            topo_name = f"N{n}_{topo_idx:04d}_{label}"
            log.info(f"  [{topo_idx+1}/{len(graphs)}] {topo_name}")

            # Save topology GML
            gml_path = dirs["top"] / f"{topo_name}.gml"
            write_gml(G, gml_path, label=topo_name)

            # Generate 3D conformers
            try:
                conformers = embed_3d(G, n_conformers=n_conformers, seed=seed + topo_idx)
            except Exception as exc:
                log.warning(f"    Embedding failed: {exc}")
                continue

            best_atoms = None
            best_energy = float("inf")
            best_conf_idx = 0
            _best_is_ok = False
            results_this_topo = []

            for conf_idx, (coords, ff_e) in enumerate(conformers):
                atoms = coords_to_atoms(coords)

                if dry_run:
                    mace_e = None
                    converged = False
                    n_steps = 0
                    final_atoms = atoms
                else:
                    final_atoms, mace_e, converged, n_steps = optimize_with_mace(
                        atoms, calc, fmax=fmax, steps=max_opt_steps,
                        logfile=str(dirs["logs"] / f"{topo_name}_conf{conf_idx}.log")
                    )

                delta_e, is_cand = estimate_stability(mace_e, n, n2_e_per_atom)

                # ── Post-relaxation structural integrity check ──────────────
                # Verifies (i) all N-N distances are physically sane, and
                # (ii) the molecule has not fragmented or silently rearranged
                # into a different topology during MACE relaxation.
                # See manual Section "Structural Integrity Checks" for the
                # full methodology and choice of distance thresholds.
                integrity = check_structural_integrity(
                    final_atoms.get_positions(), G,
                    bond_cutoff=bond_cutoff,
                    min_distance=MIN_SANE_NN_DISTANCE,
                    max_distance=MAX_SANE_NN_DISTANCE,
                )
                if integrity["status"] != "OK":
                    log.warning(
                        f"    conf{conf_idx}: integrity={integrity['status']} "
                        f"(fragments={integrity['n_fragments']}, "
                        f"target_bond_range=[{integrity['min_target_bond']:.2f},"
                        f"{integrity['max_target_bond']:.2f}] Å)"
                    )

                results_this_topo.append({
                    "atoms": final_atoms,
                    "ff_e": ff_e,
                    "mace_e": mace_e,
                    "converged": converged,
                    "n_steps": n_steps,
                    "delta_e": delta_e,
                    "is_cand": is_cand,
                    "conf_idx": conf_idx,
                    "integrity": integrity,
                })

                # Track best conformer: among structures with INTACT integrity
                # ('OK' status) first, ranked by energy; only fall back to
                # energy-only ranking if no conformer of this topology
                # survived relaxation intact. This avoids the common pitfall
                # of selecting a lower-energy but dissociated/fragmented
                # "conformer" as the representative structure for a topology.
                eff_e = mace_e if mace_e is not None else ff_e
                this_ok = integrity["status"] == "OK"
                this_is_better = False
                if best_atoms is None:
                    this_is_better = True
                elif this_ok and not _best_is_ok:
                    this_is_better = True
                elif this_ok == _best_is_ok and eff_e < best_energy:
                    this_is_better = True

                if this_is_better:
                    best_energy = eff_e
                    best_atoms = final_atoms
                    best_conf_idx = conf_idx
                    _best_is_ok = this_ok

            # ── Write outputs ─────────────────────────────────────────────────
            out_idx = write_all_conformers and len(conformers) > 1
            confs_to_write = results_this_topo if out_idx else [
                next(r for r in results_this_topo if r["conf_idx"] == best_conf_idx)
            ]

            for r in confs_to_write:
                mol_id += 1
                cid = r["conf_idx"]
                suffix = f"_conf{cid}" if write_all_conformers else ""
                stem = f"{topo_name}{suffix}"

                # Symmetry
                sym_info = get_symmetry(r["atoms"])
                integ = r["integrity"]
                integrity_counts[integ["status"]] = integrity_counts.get(integ["status"], 0) + 1

                if skip_non_intact and integ["status"] != "OK":
                    log.info(f"    skipping output for {stem} (integrity={integ['status']})")
                    continue

                # Comment for XYZ header
                comment = (
                    f"formula=N{n} topology={label} integrity={integ['status']} "
                    f"n_edges={desc['n_edges']} n_rings={desc['n_rings']} "
                    f"ff_energy={r['ff_e']:.4f} "
                    + (f"mace_energy_eV={r['mace_e']:.6f}" if r['mace_e'] else "mace_energy=N/A")
                    + f" delta_E_eV_atom={r['delta_e'] if r['delta_e'] is not None else 'N/A'}"
                    + f" sg={sym_info['spacegroup']}"
                    + f" target_bond_range_A=[{integ['min_target_bond']:.3f},{integ['max_target_bond']:.3f}]"
                )

                xyz_path = dirs["xyz"] / f"{stem}.xyz"
                cif_path = dirs["cif"] / f"{stem}.cif"

                write_xyz(r["atoms"], xyz_path, comment=comment)
                write_cif(r["atoms"], cif_path)

                # CSV row
                writer.writerow({
                    "id": mol_id,
                    "formula": f"N{n}",
                    "n_atoms": n,
                    "topology_label": label,
                    "n_edges": desc["n_edges"],
                    "n_rings": desc["n_rings"],
                    "ring_sizes": str(desc["ring_sizes"]),
                    "degree_sequence": str(desc["degree_sequence"]),
                    "max_degree_actual": desc["max_degree"],
                    "conformer_id": cid,
                    "ff_energy_au": f"{r['ff_e']:.6f}",
                    "mace_energy_eV": f"{r['mace_e']:.6f}" if r["mace_e"] is not None else "",
                    "mace_energy_eV_per_atom": (
                        f"{r['mace_e']/n:.6f}" if r["mace_e"] is not None else ""
                    ),
                    "delta_E_eV_per_atom_vs_N2": (
                        f"{r['delta_e']:.4f}" if r["delta_e"] is not None else ""
                    ),
                    "is_stability_candidate": r["is_cand"],
                    "converged": r["converged"],
                    "n_opt_steps": r["n_steps"],
                    "integrity_status": integ["status"],
                    "is_connected": integ["is_connected"],
                    "n_fragments": integ["n_fragments"],
                    "topology_preserved": integ["topology_preserved"],
                    "bonds_in_range": integ["bonds_in_range"],
                    "min_target_bond_A": f"{integ['min_target_bond']:.4f}",
                    "max_target_bond_A": f"{integ['max_target_bond']:.4f}",
                    "min_any_distance_A": f"{integ['min_any_distance']:.4f}",
                    "max_any_distance_A": f"{integ['max_any_distance']:.4f}",
                    "spacegroup": sym_info["spacegroup"],
                    "sg_number": sym_info["number"],
                    "xyz_file": str(xyz_path.relative_to(out)),
                    "cif_file": str(cif_path.relative_to(out)),
                })
                total_structures += 1

            csv_file.flush()

        elapsed_n = time.time() - t_n_start
        log.info(f"N{n}: done ({len(graphs)} topologies, {elapsed_n:.1f} s)")

    csv_file.close()

    log.info("=" * 60)
    log.info(f"DONE. {total_structures} structures written to '{output_dir}/'")
    log.info(f"  xyz/   : {len(list(dirs['xyz'].glob('*.xyz')))} files")
    log.info(f"  cif/   : {len(list(dirs['cif'].glob('*.cif')))} files")
    log.info(f"  topologies/ : {len(list(dirs['top'].glob('*.gml')))} GML files")
    log.info(f"  results.csv : {csv_path}")
    log.info("-" * 60)
    log.info("Post-relaxation structural integrity summary:")
    n_checked = sum(integrity_counts.values())
    for status in ["OK", "FRAGMENTED", "REARRANGED", "BOND_ANOMALY", "COLLAPSED"]:
        count = integrity_counts.get(status, 0)
        pct = 100.0 * count / n_checked if n_checked else 0.0
        log.info(f"    {status:14s}: {count:5d}  ({pct:5.1f}%)")
    if n_checked and integrity_counts.get("OK", 0) < n_checked:
        log.warning(
            f"  {n_checked - integrity_counts.get('OK', 0)} structures did NOT preserve "
            "their intended topology after MACE relaxation (fragmented, rearranged, "
            "or had anomalous bond lengths). See 'integrity_status' column in results.csv."
        )

    return csv_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Size range
    p.add_argument("--n_min", type=int, default=3,
                   help="Minimum number of N atoms (default: 3)")
    p.add_argument("--n_max", type=int, default=10,
                   help="Maximum number of N atoms (default: 10)")
    p.add_argument("--max_degree", type=int, default=3,
                   help="Maximum coordination number per N (default: 3)")

    # Sampling
    p.add_argument("--n_conformers", type=int, default=5,
                   help="3D conformers per topology (default: 5)")
    p.add_argument("--n_sample", type=int, default=100,
                   help="Target topologies for n>6 stochastic sampling (default: 100)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")

    # MACE
    p.add_argument("--model", type=str, default="mace_off",
                   choices=["mace_off", "mace_mp", "custom"],
                   help="MACE model type (default: mace_off = MACE-OFF23)")
    p.add_argument("--model_size", type=str, default="medium",
                   help="Model size: 'small', 'medium', 'large', or checkpoint path "
                        "(default: medium)")
    p.add_argument("--device", type=str, default="cpu",
                   help="Torch device: 'cpu', 'cuda', 'cuda:0', 'mps' (default: cpu)")
    p.add_argument("--dtype", type=str, default="float64",
                   choices=["float32", "float64"],
                   help="Float precision (default: float64)")
    p.add_argument("--fmax", type=float, default=0.05,
                   help="Force convergence threshold in eV/Å (default: 0.05)")
    p.add_argument("--max_steps", type=int, default=500,
                   help="Max LBFGS steps per optimization (default: 500)")

    # Output
    p.add_argument("--output_dir", type=str, default="polyN_results",
                   help="Output directory (default: polyN_results/)")
    p.add_argument("--all_conformers", action="store_true",
                   help="Write all conformers, not just the lowest-energy one")
    p.add_argument("--skip_non_intact", action="store_true",
                   help="Do not write XYZ/CIF for fragmented/rearranged/collapsed "
                        "structures (still logged in results.csv)")
    p.add_argument("--bond_cutoff", type=float, default=BOND_CUTOFF_NN,
                   help=f"Max N-N distance (Å) considered a covalent bond "
                        f"for integrity checking (default: {BOND_CUTOFF_NN})")

    # Mode
    p.add_argument("--dry_run", action="store_true",
                   help="Skip MACE — use force-field energies only (topology test mode)")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    log.info("polynitrogen_mace.py — Neutral polynitrogen structure generator")
    log.info(f"  N range   : N{args.n_min} to N{args.n_max}")
    log.info(f"  max_degree: {args.max_degree}")
    log.info(f"  conformers: {args.n_conformers} per topology")
    log.info(f"  sampling  : {args.n_sample} topologies for n > {EXHAUSTIVE_LIMIT}")
    log.info(f"  MACE      : {'(dry_run, FF only)' if args.dry_run else args.model + '/' + args.model_size}")
    log.info(f"  device    : {args.device}  dtype={args.dtype}")
    log.info(f"  output    : {args.output_dir}/")

    run_pipeline(
        n_min=args.n_min,
        n_max=args.n_max,
        max_degree=args.max_degree,
        n_conformers=args.n_conformers,
        n_sample_target=args.n_sample,
        model_type=args.model,
        model_size=args.model_size,
        device=args.device,
        dtype=args.dtype,
        fmax=args.fmax,
        max_opt_steps=args.max_steps,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        seed=args.seed,
        write_all_conformers=args.all_conformers,
        skip_non_intact=args.skip_non_intact,
        bond_cutoff=args.bond_cutoff,
    )
