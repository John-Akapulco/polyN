#!/usr/bin/env python3
"""
test_dftb_known_geometries.py
================================
Validation of the DFTB+/mio-1-1 installation against a small set of
HAND-BUILT, chemically well-defined geometries -- deliberately NOT
produced by the random AIRSS-style generator -- to separate two
competing hypotheses for the high crash rate observed in
polynitrogen_charged_explore.py production runs:

    H1: the random generator occasionally produces geometries with
        pathological interatomic distances (too short / too close to
        degenerate) that DFTB+ cannot handle.
    H2: the DFTB+ installation itself (binary build, or the mio-1-1
        Slater-Koster parameter set) is fragile/misconfigured in a way
        that causes it to crash even on ordinary, sensible geometries.

This script tests H2 directly: every geometry here is constructed from
textbook bond lengths/angles, not random sampling, so if DFTB+ crashes
on any of these, the problem cannot be attributed to the generator
(supporting H2 or a combination of both); if all of these succeed
cleanly, H2 is disfavored and the crash rate is more likely a genuine
property of the wider random structure space being sampled (closer to
H1, or simply an intrinsic rate of numerically difficult geometries
that any sufficiently broad random search will encounter).

GEOMETRIES TESTED
-------------------
  1. N2 (neutral)         : sanity reference, bond length vs experiment
  2. N3- (linear azide)   : textbook D-infinity-h geometry,
                             d(N-N) = 1.18 A (experimental azide ion)
  3. N5- (planar pentazolate) : regular pentagon, d(N-N) = 1.32 A
                             (the same geometry already validated
                             successfully in polynitrogen_charged_dftb.py)
  4. N7- (regular zig-zag chain) : NOT a random geometry -- built
                             point-by-point with a fixed, chemically
                             reasonable bond length (1.35 A) and bond
                             angle (110 deg), the same generator used
                             for the N5+ literature seed in
                             polynitrogen_charged_dftb.py

USAGE
------
    python test_dftb_known_geometries.py --slako_dir ~/dftb_params/mio-1-1/

AUTHOR
-------
    Generated for IC2MP/E4 Mediacat -- Universite de Poitiers
"""

import argparse
import shutil
import sys
from pathlib import Path

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


def check_dftb_available():
    if shutil.which("dftb+") is None:
        sys.exit(
            "ERROR: the 'dftb+' executable was not found on your PATH.\n"
            "Install it first, e.g.: conda install -c conda-forge dftbplus"
        )


def load_dftb_calculator(slako_dir, charge=0, label="dftb_test/calc"):
    slako_path = str(Path(slako_dir).expanduser().resolve()) + "/"
    if not Path(slako_dir).expanduser().exists():
        sys.exit(f"ERROR: --slako_dir '{slako_dir}' does not exist.")
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


def geom_n2():
    """Neutral N2, experimental bond length 1.0977 A."""
    return Atoms("N2", positions=[[0, 0, 0], [0, 0, 1.0977]]), 0


def geom_n3_linear_azide():
    """
    N3- (azide ion), linear D-infinity-h, the well-known textbook
    structure (symmetric, both N-N distances equal). Experimental
    azide N-N bond length ~1.18 A (intermediate between double and
    triple bond character, consistent with its resonance structure).
    """
    d = 1.18
    coords = [[0, 0, -d], [0, 0, 0], [0, 0, d]]
    return Atoms("N3", positions=coords), -1


def geom_n5_planar_pentazolate():
    """
    N5- (cyclo-pentazolate), regular planar pentagon, D5h, the same
    geometry already validated successfully end-to-end in
    polynitrogen_charged_dftb.py (relaxed cleanly to ring-5, OK
    integrity, in this project's earlier validation). N-N = 1.32 A,
    a typical aromatic-ring N-N bond length for this species.
    """
    n = 5
    bond_length = 1.32
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = bond_length / (2 * np.sin(np.pi / n))
    coords = [[r * np.cos(a), r * np.sin(a), 0.0] for a in angles]
    return Atoms("N5", positions=coords), -1


def geom_n7_regular_zigzag():
    """
    N7- as a REGULAR (non-random) planar zig-zag chain: fixed bond
    length 1.35 A, fixed bond angle 110 deg, built point-by-point
    exactly like polynitrogen_charged_dftb.py's validated
    generate_bent_chain() function (the one used for the successful
    N5+ test). This is deliberately NOT produced by the random AIRSS
    generator -- if DFTB+ crashes on this clean, textbook-style
    geometry, the crash cannot be blamed on the random generator.
    """
    n = 7
    bond_length = 1.35
    bend_angle_deg = 110.0
    coords = np.zeros((n, 3))
    coords[0] = [0.0, 0.0, 0.0]
    coords[1] = [bond_length, 0.0, 0.0]
    deviation = np.radians(180.0 - bend_angle_deg)
    direction_sign = 1
    for i in range(2, n):
        prev_vec = coords[i - 1] - coords[i - 2]
        prev_vec /= np.linalg.norm(prev_vec)
        angle = direction_sign * deviation
        c, s = np.cos(angle), np.sin(angle)
        new_vec = np.array([
            prev_vec[0] * c - prev_vec[1] * s,
            prev_vec[0] * s + prev_vec[1] * c,
            0.0,
        ])
        coords[i] = coords[i - 1] + new_vec * bond_length
        direction_sign *= -1
    coords -= coords.mean(axis=0)
    return Atoms("N7", positions=coords), -1


GEOMETRIES = [
    ("N2 (neutral, sanity reference)", geom_n2),
    ("N3- (linear azide, textbook)", geom_n3_linear_azide),
    ("N5- (planar pentazolate, textbook)", geom_n5_planar_pentazolate),
    ("N7- (regular zig-zag chain, non-random)", geom_n7_regular_zigzag),
]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--slako_dir", type=str, required=True)
    p.add_argument("--fmax", type=float, default=0.05)
    p.add_argument("--max_steps", type=int, default=200)
    args = p.parse_args()

    check_dftb_available()

    print("=" * 70)
    print("DFTB+/mio-1-1 validation against hand-built, known geometries")
    print("=" * 70)
    print()

    results = []
    for label, geom_fn in GEOMETRIES:
        print(f"--- {label} ---")
        atoms, charge = geom_fn()
        n = len(atoms)
        safe_label = label.split()[0].replace("-", "anion").replace("+", "cation")
        calc = load_dftb_calculator(
            args.slako_dir, charge=charge,
            label=f"dftb_test/{safe_label}/calc"
        )
        atoms.calc = calc
        opt = LBFGS(atoms, logfile=None)
        try:
            opt.run(fmax=args.fmax, steps=args.max_steps)
            energy = atoms.get_potential_energy()
            e_per_atom = energy / n
            print(f"  SUCCESS: E = {energy:.4f} eV ({e_per_atom:.4f} eV/atom)")
            if n == 2:
                d = atoms.get_distance(0, 1)
                print(f"  d(N-N) = {d:.4f} A (experimental: 1.0977 A)")
            results.append((label, "SUCCESS", energy))
        except Exception as exc:
            print(f"  CRASH: {type(exc).__name__}: {exc}")
            results.append((label, "CRASH", None))
        print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    n_success = sum(1 for _, status, _ in results if status == "SUCCESS")
    n_crash = sum(1 for _, status, _ in results if status == "CRASH")
    for label, status, energy in results:
        marker = "OK  " if status == "SUCCESS" else "FAIL"
        print(f"  [{marker}] {label}")
    print()
    print(f"{n_success}/{len(results)} succeeded, {n_crash}/{len(results)} crashed")
    print()

    if n_crash == 0:
        print("CONCLUSION: DFTB+/mio-1-1 handled every hand-built, chemically")
        print("sensible geometry cleanly, INCLUDING a non-random N7- chain.")
        print("This argues AGAINST a fundamentally broken DFTB+ installation")
        print("(hypothesis H2) and is more consistent with the production")
        print("crashes being a genuine property of the wider, less")
        print("constrained random search space the AIRSS generator explores")
        print("(some fraction of which lands on numerically difficult")
        print("electronic structures even when interatomic distances")
        print("themselves are individually reasonable) -- hypothesis H1's")
        print("distance-violation variant was already ruled out separately")
        print("(500/500 random N7 generations respected MINSEP exactly).")
    else:
        print("CONCLUSION: DFTB+ crashed on at least one hand-built, textbook")
        print("geometry. This is strong evidence for a problem with the")
        print("DFTB+ installation or Slater-Koster parameter set itself")
        print("(hypothesis H2), independent of anything the random generator")
        print("does. Recommend re-checking the dftb+ build/installation and")
        print("re-downloading the mio-1-1 .skf files before trusting any")
        print("further production campaign.")


if __name__ == "__main__":
    main()
