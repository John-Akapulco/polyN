#!/usr/bin/env python3
"""
test_large_anionic_rings.py
============================
Targeted test to determine whether monocyclic ring topologies exist as
local minima on the DFTB+ potential-energy surface for N9- and N11-.

SCIENTIFIC MOTIVATION
----------------------
A production Minima Hopping campaign (anion_mh_N9N11/) found no ring-9
or ring-11 structures among the 5 intact minima recovered -- only chains
and one polycyclic. However, that campaign suffered very high DFTB+ crash
rates (4/6 seeds for N9-, 5/6 for N11-), leaving the PES largely
unexplored. Two competing interpretations remain open:

  (A) ring-9 and ring-11 DO exist as local minima but were not found
      because no seed started close enough to the ring basin, and the
      crash rate prevented sufficient exploration.

  (B) Large monocyclic rings (n >= 9) are NOT local minima under
      DFTB+/mio-1-1: the ring geometry relaxes away to a chain or
      fragments, consistent with the known loss of ring strain tolerance
      beyond n=7 in polynitrogen chemistry and the absence of a
      Huckel-aromatic driving force for n > 7 (N5- benefits from 6-pi
      aromaticity, but N9- has 10 pi electrons -- anti-aromatic in
      Huckel's rule for monocycles).

This script tests interpretation (A) directly: it starts DFTB+ from
an EXPLICIT RING GEOMETRY (FF-pre-relaxed monocycle, virtually perfect
Dnh symmetry) and checks whether local relaxation converges to a ring
minimum or escapes to a chain/fragmented state.

If the ring IS a local minimum under DFTB+: interpretation (A), and
the production campaign simply missed it -- a targeted MH run starting
from the ring seed would find it.

If the ring relaxes AWAY from the ring to a chain or fragments:
interpretation (B) -- the ring topology is a saddle point or barrier-
free funnel toward another basin, and no amount of additional MH
sampling starting from non-ring seeds would find it.

GEOMETRIES TESTED
------------------
  1. N9-  monocyclic ring (D9h, d(N-N)=1.376 A from FF pre-relaxation)
  2. N11- monocyclic ring (D11h, d(N-N)=1.376 A from FF pre-relaxation)

For comparison, we also re-run N5- ring-5 (already validated) and N7-
ring-7 (already found in production) as positive controls -- if these
do NOT relax to rings under DFTB+, something is wrong with the setup.

USAGE
------
    python test_large_anionic_rings.py --slako_dir ~/dftb_params/mio-1-1/

AUTHOR
-------
    Generated for IC2MP/E4 Mediacat -- Universite de Poitiers
"""

import argparse
import shutil
import sys
from pathlib import Path

import networkx as nx
import numpy as np

try:
    from ase import Atoms
    from ase.optimize import LBFGS
except ImportError:
    sys.exit("ERROR: ASE not found. Run: pip install ase")

try:
    from ase.calculators.dftb import Dftb
except ImportError:
    sys.exit("ERROR: ASE's DFTB+ calculator module not found.")

# Reuse FF pre-relaxation and integrity check from the main exploration script
sys.path.insert(0, str(Path(__file__).parent))
try:
    from polynitrogen_charged_explore import (
        embed_graph_3d_ff, check_structural_integrity, classify_topology,
        infer_geometric_graph, BOND_CUTOFF_NN,
    )
except ImportError:
    sys.exit("ERROR: polynitrogen_charged_explore.py not found in the same "
             "directory. Run this script from your polyN_study/ folder.")


def check_dftb_available():
    if shutil.which("dftb+") is None:
        sys.exit(
            "ERROR: 'dftb+' not on PATH. "
            "Install: conda install -c conda-forge dftbplus"
        )


def load_dftb_calculator(slako_dir, charge, label):
    slako_path = str(Path(slako_dir).expanduser().resolve()) + "/"
    Path(label).parent.mkdir(parents=True, exist_ok=True)
    return Dftb(
        label=label,
        Hamiltonian_="DFTB",
        Hamiltonian_SCC="Yes",
        Hamiltonian_SCCTolerance=1e-7,
        Hamiltonian_MaxSCCIterations=200,
        Hamiltonian_MaxAngularMomentum_="",
        Hamiltonian_MaxAngularMomentum_N="p",
        Hamiltonian_Charge=charge,
        slako_dir=slako_path,
    )


def ring_geometry(n):
    """Build an explicit monocyclic ring Nn using FF pre-relaxation
    of the cycle graph -- same procedure validated in embed_graph_3d_ff."""
    G = nx.cycle_graph(n)
    coords = embed_graph_3d_ff(G, seed=0)
    return coords


def relax_and_classify(atoms, calc, label, fmax=0.05, steps=300):
    atoms = atoms.copy()
    atoms.calc = calc
    opt = LBFGS(atoms, logfile=None)
    try:
        opt.run(fmax=fmax, steps=steps)
        energy = atoms.get_potential_energy()
    except Exception as exc:
        return None, None, f"CRASH: {exc}"

    integ = check_structural_integrity(atoms.get_positions())
    G_final = integ["final_graph"]
    topo = classify_topology(G_final)
    return energy, topo, integ["status"]


# ─────────────────────────────────────────────────────────────────────────────
# TEST CASES
# ─────────────────────────────────────────────────────────────────────────────

CASES = [
    # (label, n_atoms, charge, expected_topology_if_ring_is_minimum)
    ("N5- ring-5 [positive control]",   5,  -1, "ring-5"),
    ("N7- ring-7 [positive control]",   7,  -1, "ring-7"),
    ("N9- ring-9 [hypothesis test]",    9,  -1, "ring-9"),
    ("N11- ring-11 [hypothesis test]", 11,  -1, "ring-11"),
]


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--slako_dir", type=str, required=True)
    p.add_argument("--fmax", type=float, default=0.05)
    p.add_argument("--max_steps", type=int, default=300)
    args = p.parse_args()

    check_dftb_available()

    print("=" * 70)
    print("Targeted test: large monocyclic anions N9- and N11-")
    print("Starting geometry: explicit FF-pre-relaxed ring (Dnh, d~1.376 A)")
    print("=" * 70)
    print()

    results = []
    for label, n, charge, expected in CASES:
        print(f"--- {label} ---")
        coords = ring_geometry(n)

        # Report starting geometry quality
        bonded = [(i, (i+1)%n) for i in range(n)]
        bond_dists = [np.linalg.norm(coords[u]-coords[v]) for u,v in bonded]
        print(f"  Starting geometry: d(N-N)={np.mean(bond_dists):.4f} ± "
              f"{np.std(bond_dists):.4f} A")

        calc = load_dftb_calculator(
            args.slako_dir, charge=charge,
            label=f"dftb_ring_test/N{n}_ring/calc"
        )
        atoms = Atoms("N" * n, positions=coords)
        # Set a large periodic cell so DFTB+ treats the molecule as isolated.
        # Without an explicit cell, DFTB+ may have undefined behavior for
        # larger geometries (observed: ring-7 crashes without a cell, passes
        # with one -- the N5- ring-5 control works without because it fits
        # within DFTB+'s implicit defaults, but larger rings do not).
        atoms.set_cell([30.0, 30.0, 30.0])
        atoms.center()
        atoms.set_pbc(True)
        energy, topo, status = relax_and_classify(
            atoms, calc,
            label=f"dftb_ring_test/N{n}_ring/calc",
            fmax=args.fmax, steps=args.max_steps,
        )

        if energy is None:
            print(f"  RESULT: DFTB+ CRASHED ({status})")
            verdict = "CRASH"
        else:
            e_per_atom = energy / n
            print(f"  RESULT: E = {energy:.4f} eV ({e_per_atom:.4f} eV/atom)")
            print(f"  Relaxed topology: {topo}  (integrity: {status})")

            if topo == expected:
                verdict = "RING_STABLE"
                print(f"  VERDICT: Ring IS a local minimum -- "
                      f"topology preserved after DFTB+ relaxation ✓")
            elif status == "FRAGMENTED":
                verdict = "FRAGMENTED"
                print(f"  VERDICT: Ring relaxed to FRAGMENTS ({topo}) -- "
                      f"ring is NOT a local minimum under DFTB+")
            else:
                verdict = "REARRANGED"
                print(f"  VERDICT: Ring rearranged to {topo} -- "
                      f"ring is NOT a local minimum, but molecule stayed intact")

        results.append((label, n, verdict, topo if energy else "N/A"))
        print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for label, n, verdict, topo in results:
        icon = {"RING_STABLE": "✓", "FRAGMENTED": "✗",
                "REARRANGED": "~", "CRASH": "!"}.get(verdict, "?")
        print(f"  [{icon}] {label}: {verdict} → {topo}")

    print()
    controls_ok = all(v == "RING_STABLE"
                      for _, n, v, _ in results if n in [5, 7])
    large_rings = [(n, v) for _, n, v, _ in results if n in [9, 11]]

    if not controls_ok:
        print("WARNING: positive controls (N5-, N7-) did NOT give RING_STABLE.")
        print("Check DFTB+ installation before interpreting N9-/N11- results.")
    else:
        print("Positive controls OK (N5- ring-5 and N7- ring-7 both stable).")
        print()
        for n, verdict in large_rings:
            if verdict == "RING_STABLE":
                print(f"N{n}-: ring-{n} IS a local minimum under DFTB+.")
                print(f"  → Interpretation (A): the production MH campaign")
                print(f"    simply missed it due to crash-dominated exploration.")
                print(f"    A targeted MH run seeded from ring-{n} should find it.")
            elif verdict == "FRAGMENTED":
                print(f"N{n}-: ring-{n} is NOT a local minimum (fragments).")
                print(f"  → Interpretation (B): large monocyclic rings are")
                print(f"    thermodynamically unstable at this size under DFTB+.")
                print(f"    Consistent with loss of Huckel aromaticity beyond N7-.")
            elif verdict == "REARRANGED":
                print(f"N{n}-: ring-{n} rearranges to a different intact topology.")
                print(f"  → The ring is a transition state or saddle point,")
                print(f"    not a true local minimum.")
            else:
                print(f"N{n}-: DFTB+ crashed -- inconclusive. Try with a")
                print(f"    slightly different starting geometry (--fmax relaxed).")


if __name__ == "__main__":
    main()
