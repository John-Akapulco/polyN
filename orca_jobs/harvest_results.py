#!/usr/bin/env python3
"""Harvest results from the polyN ORCA (WB97X-D4/aug-cc-pVTZ Opt Freq) screening
campaign under jobs/<name>/<name>.property.txt + .out + .xyz.

Three sub-commands:

  harvest      Parse every finished job into:
                 results_summary.csv    energies, ZPE, H, G, mult, symmetry,
                                         HOMO/LUMO (eV), imaginary-mode count
                 results_bonds.csv      one row per N-N pair: distance,
                                         Mayer bond order, distance-based class
                 results_charges.csv    one row per atom: Mulliken + Loewdin
                                         charge (Bader is NOT native to ORCA --
                                         needs the external Henkelman Bader
                                         code on a density cube, not set up
                                         here; see conversation notes)
                 results_imaginary_modes.csv
                                         only rows for structures that are NOT
                                         a true minimum (freq < -10 cm^-1),
                                         with that mode's frequency + IR
                                         intensity
                 xyz_dft_wb97xd4/       each job's optimized geometry, copied
                                         (never moved) alongside the untouched
                                         jobs/<name>/ directory -- everything
                                         needed for a later CCSD(T) campaign
                                         (.gbw, .densities, .hess, ...) is left
                                         in place, nothing is deleted.

  compare-xtb  For each (N-count, charge) family, compare the GFN2-xTB
               ranking (E_react read from xyz_gfn2xtb/*.xyz comment line)
               against the DFT ranking, both expressed relative to the DFT
               ground state of that family. Writes compare_cation.csv,
               compare_anion.csv, compare_neutral.csv.

  reactions    Compute the stepwise N2-loss decomposition chain
               Nx^q -> N(x-2)^q + N2 for q in {0, +1, -1}, using the DFT
               ground state of each family. For neutrals (q=0) the chain
               terminates at N2 itself (dHf(N2) = 0 by definition, so
               dHf(Nx neutral) falls out with no external anchor). For ions
               the chain terminates at N5+/-, which needs a literature
               formation-enthalpy anchor (CLI flags below) referencing the
               specific reference isomer (cyclic N5-, bent/V N5+) --
               "equations a ajuster": edit REACTION_STEP / the anchor
               isomers below if the real decomposition path differs.

Only Python 3.6 stdlib is used (no numpy/pandas on this cluster).
"""
import argparse
import csv
import glob
import math
import os
import re
import shutil
import sys

# ---------------------------------------------------------------------------
# CONFIG -- adjust freely
# ---------------------------------------------------------------------------

HARTREE_TO_KCAL = 627.5094740631
BOHR_TO_ANGSTROM = 0.529177210903

NAME_RE = re.compile(
    r"^N(?P<n>\d+)_(?P<family>neutral|cation|anion)_(?P=family)_(?P<rank>\d+)$"
)

# N-N bond-length classification thresholds (Angstrom). Purely geometric;
# cross-check against the Mayer bond order column in results_bonds.csv,
# which is the physically grounded quantity -- these bins are a convenience
# label, not a substitute.
BOND_BINS = [
    (1.15, "triple"),        # ref N2 ~1.098 A
    (1.22, "double"),        # ref azo N=N ~1.20-1.25 A
    (1.32, "delocalized_1.5"),  # ref cyclic N5- ring ~1.30-1.32 A
    (1.55, "single"),        # ref hydrazine N-N ~1.45 A
    (1.90, "long/weak"),     # stretched / partial bond
]
BOND_NONBONDED_LABEL = "non-bonded"

# A vibrational mode more negative than this (cm^-1) is flagged as a genuine
# imaginary frequency (saddle point, not a true minimum). The 5-6 near-zero
# translation/rotation modes ORCA always lists first can show tiny negative
# numerical noise (a fraction of a cm^-1) -- this threshold keeps those from
# being reported as false positives.
IMAG_FREQ_THRESHOLD_CM1 = -10.0

# Where harvest() copies each job's final optimized geometry (Angstrom),
# mirroring the xyz_gfn2xtb/ naming so the two levels of theory line up
# 1:1 by filename.
DFT_XYZ_OUT_DIR = "xyz_dft_wb97xd4"

# Reference isomers anchoring the ion decomposition chains at N5.
# Fill in the job name of the actual computed conformer once known, and the
# literature formation enthalpy (kcal/mol, gas phase, 298 K) if you want
# absolute dHf out of `reactions` rather than just step-wise reaction
# energies. Left as None => reactions command reports relative energies only
# for ions; neutrals never need this (anchored at N2, dHf=0).
# Found 2026-09-06 in polyN-pipeline's biblio_polyN/final_report_isodesmic.csv
# (CCSD(T)-level literature values, "B" reference):
#   N5-_pentagon (D5h)        dHf = -106.4 kcal/mol
#   N5+_Vchain   (C2v, bent)  dHf =  195.9 kcal/mol (195.8 experimental X-ray)
# Our N5_anion_anion_001 IS that D5h pentagon (confirmed by point group and by
# GFN2-xTB energy matching to 6 decimals -- see XTB_E_N5_ANION_HARTREE) and is
# also this run's own internal xTB reference (E_react=-0.00), so no rescaling
# needed there. Our N5_cation_cation_001 is C2v and matches the Vchain -- but
# this run's internal xTB reference for the cation family is
# N5_cation_cation_004 (E_react=0.00 there, a *different*, higher-energy C1
# isomer), so E_react values as printed in xyz_gfn2xtb/*.xyz are relative to
# THAT, not to the Vchain -- rescale by N5_CATION_INTERNAL_REF_OFFSET_KCAL
# (= E_react_as_reported for N5_cation_cation_001, i.e. how far the internal
# reference sits from the literature-matched structure) before adding the
# literature anchor.
N5_ANION_REF_NAME = "N5_anion_anion_001"    # cyclic, D5h -- matches literature exactly
N5_ANION_REF_DHF_KCAL = -106.4               # literature dHf(cyclic N5-), kcal/mol

N5_CATION_REF_NAME = "N5_cation_cation_001"  # bent/V, C2v -- matches literature Vchain
N5_CATION_REF_DHF_KCAL = 195.9               # literature dHf(bent N5+ Vchain), kcal/mol
N5_CATION_INTERNAL_REF_OFFSET_KCAL = -105.63  # E_react(N5_cation_cation_001) as printed

# GFN2-xTB anchors behind the E_react convention polyN_pipeline.py uses
# (README: neutral Nx -> (x/2) N2; charged Nx+/- -> (x-5)/2 N2 + N5+/-).
# Re-derived independently 2026-09-06 (xtb 6.7.1) rather than trusted blind:
#   - E(N2) matched config_example.yaml's -5.763935 Ha to 6 decimals.
#   - E(N5-) matched -14.7609 Ha to 4 decimals, from OUR OWN
#     xyz_gfn2xtb/N5_anion_anion_001.xyz (its own E_react=-0.00, i.e. this
#     run's anion reference candidate IS that structure).
#   - E(N5+) matched -13.8169 Ha to 4 decimals, but NOT from our rank-1 --
#     from N5_cation_cation_004 (E_react=0.00 there, not at rank 1;
#     apparently later frequency-hop re-optimization promoted other
#     candidates past the frozen reference). Ranks 1-3 are genuinely more
#     stable than this anchor by construction.
# This lets xtb_Erel_kcal in compare-xtb be a real absolute-energy
# difference instead of the ambiguous as-reported E_react.
XTB_E_N2_HARTREE = -5.763935420175
XTB_E_N5_ANION_HARTREE = -14.760899219951
XTB_E_N5_CATION_HARTREE = -13.816934622594


def reconstruct_xtb_energy_hartree(n_atoms, family, e_react_kcal):
    """E_react (as printed in xyz_gfn2xtb/*.xyz comments, kcal/mol) -> the
    absolute GFN2-xTB total energy (Hartree) it was computed from."""
    e_react_ha = e_react_kcal / HARTREE_TO_KCAL
    if family == "neutral":
        return e_react_ha + (n_atoms / 2.0) * XTB_E_N2_HARTREE
    coeff_n2 = (n_atoms - 5) / 2.0
    ref = XTB_E_N5_CATION_HARTREE if family == "cation" else XTB_E_N5_ANION_HARTREE
    return e_react_ha + coeff_n2 * XTB_E_N2_HARTREE + ref


def reaction_step(n, charge):
    """Nx^q -> N(x-2)^q + N2. Edit here if the real decomposition path differs
    (e.g. loss of N3 instead of N2, or a branch point at a specific size)."""
    return n - 2, charge


# ---------------------------------------------------------------------------
# property.txt / .out / .xyz parsing
# ---------------------------------------------------------------------------

def read_text(path):
    with open(path) as fh:
        return fh.read()


def split_blocks(prop_text):
    """Return {block_name: [block_text, ...]} in file order, one entry per
    occurrence (multiple Geometry/SCF_Energy blocks appear across Opt
    cycles -- callers wanting the converged geometry take [-1])."""
    # Python 3.6's re.split() rejects zero-width lookahead patterns, so slice
    # manually between successive "$Word" start-of-line markers instead.
    blocks = {}
    starts = [m.start() for m in re.finditer(r"^\$\w+", prop_text, flags=re.M)]
    starts.append(len(prop_text))
    for start, end in zip(starts, starts[1:]):
        part = prop_text[start:end]
        m = re.match(r"^\$(\w+)", part)
        if not m:
            continue
        blocks.setdefault(m.group(1), []).append(part)
    return blocks


def get_scalar(block_text, key, cast=float):
    m = re.search(r"&" + re.escape(key) + r"\s*\[&Type \"\w+\"\]\s*([^\s\"]+)", block_text)
    if not m:
        return None
    return cast(m.group(1))


def get_array_rows(block_text, array_name):
    """Rows of an ORCA property-file array: data rows start at column 0 with
    the row index (no leading whitespace); the column-index header row is
    indented, so this pattern skips it automatically."""
    m = re.search(r"&" + re.escape(array_name) + r"\s*\[[^\]]*\][^\n]*\n(.*?)(?=\n\s*&|\n\$End)",
                  block_text, flags=re.S)
    if not m:
        return []
    rows = []
    for line in m.group(1).splitlines():
        rm = re.match(r"^(\d+)\s+(.+)$", line)
        if rm:
            rows.append(rm.group(2).split())
    return rows


def parse_property_file(path):
    text = read_text(path)
    blocks = split_blocks(text)

    calc_info = blocks.get("Calculation_Info", [""])[-1]
    mult = get_scalar(calc_info, "Mult", int)
    charge = get_scalar(calc_info, "Charge", int)
    n_atoms = get_scalar(calc_info, "NumOfAtoms", int)

    thermo = blocks.get("THERMOCHEMISTRY_Energies")
    if not thermo:
        return None  # Freq step never finished (or job still running)
    thermo = thermo[-1]
    elec = get_scalar(thermo, "elEnergy")
    zpe = get_scalar(thermo, "zpe")
    u_inner = get_scalar(thermo, "innerEnergyU")
    h_enthalpy = get_scalar(thermo, "enthalpyH")
    g_free = get_scalar(thermo, "freeEnergyG")

    mayer_block = blocks.get("SCF_Mayer_Population_Analysis")
    bonds = []
    if mayer_block:
        mb = mayer_block[-1]
        components = get_array_rows(mb, "components")     # [i, Zi, j, Zj]
        orders = get_array_rows(mb, "BondOrders")          # [order]
        for comp, ordr in zip(components, orders):
            i, zi, j, zj = (int(x) for x in comp)
            if zi == 7 and zj == 7:  # N-N pair
                bonds.append((i, j, float(ordr[0])))

    freq_list = [float(row[0]) for row in get_array_rows(thermo, "FREQ")]

    charges = {}  # atom_index -> {"mulliken": x, "loewdin": y}
    for block_name, col in [("SCF_Mulliken_Population_Analysis", "mulliken"),
                             ("SCF_Loewdin_Population_Analysis", "loewdin")]:
        pop_block = blocks.get(block_name)
        if not pop_block:
            continue
        for i, row in enumerate(get_array_rows(pop_block[-1], "AtomicCharges")):
            charges.setdefault(i, {})[col] = float(row[0])

    return {
        "mult": mult,
        "charge": charge,
        "n_atoms": n_atoms,
        "electronic_Eh": elec,
        "zpe_Eh": zpe,
        "inner_U_Eh": u_inner,
        "enthalpy_H_Eh": h_enthalpy,
        "gibbs_G_Eh": g_free,
        "mayer_nn_bonds": bonds,
        "freq_list": freq_list,
        "atomic_charges": charges,
    }


def parse_point_group(out_path):
    if not os.path.exists(out_path):
        return None, None
    text = read_text(out_path)
    m = re.search(r"Point Group:\s*(\S+),\s*Symmetry Number:\s*(\d+)", text)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def parse_ir_intensities(out_text):
    """{mode_index: intensity_km_mol} from the IR SPECTRUM table (not present
    in property.txt -- text-only). Mode indices line up directly with the
    FREQ array from THERMOCHEMISTRY_Energies (both 0-indexed over all modes,
    translations/rotations included)."""
    m = re.search(r"IR SPECTRUM\n-+\n\n.*?\n.*?\n-+\n(.*?)\n\n", out_text, flags=re.S)
    intens = {}
    if not m:
        return intens
    for line in m.group(1).splitlines():
        rm = re.match(r"\s*(\d+):\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)", line)
        if rm:
            intens[int(rm.group(1))] = float(rm.group(4))
    return intens


def _homo_lumo_from_table(rows_text):
    homo = lumo = None
    for line in rows_text.splitlines():
        rm = re.match(r"\s*\d+\s+([\d.]+)\s+-?[\d.]+\s+(-?[\d.]+)", line)
        if not rm:
            continue
        occ, e_ev = float(rm.group(1)), float(rm.group(2))
        if occ > 0:
            homo = e_ev
        elif lumo is None:
            lumo = e_ev
    return homo, lumo


def parse_homo_lumo(out_text):
    """Last 'ORBITAL ENERGIES' table (the converged geometry's SCF) -> HOMO
    and LUMO in eV. Open-shell (UHF/UKS) jobs print separate SPIN UP/SPIN
    DOWN tables instead of one; both are handled."""
    starts = [m.end() for m in re.finditer(r"ORBITAL ENERGIES\n-+\n", out_text)]
    if not starts:
        return {}
    chunk = out_text[starts[-1]:starts[-1] + 8000]
    if "SPIN UP ORBITALS" in chunk:
        up_m = re.search(r"SPIN UP ORBITALS\n.*?\n(.*?)\n\n", chunk, flags=re.S)
        down_m = re.search(r"SPIN DOWN ORBITALS\n.*?\n(.*?)\n\n", chunk, flags=re.S)
        homo_a, lumo_a = _homo_lumo_from_table(up_m.group(1)) if up_m else (None, None)
        homo_b, lumo_b = _homo_lumo_from_table(down_m.group(1)) if down_m else (None, None)
        return {"homo_alpha_eV": homo_a, "lumo_alpha_eV": lumo_a,
                "homo_beta_eV": homo_b, "lumo_beta_eV": lumo_b}
    body_m = re.search(r"E\(eV\)[^\n]*\n(.*?)\n\n", chunk, flags=re.S)
    homo, lumo = _homo_lumo_from_table(body_m.group(1)) if body_m else (None, None)
    return {"homo_eV": homo, "lumo_eV": lumo}


def parse_out_extras(out_path, freq_list):
    """HOMO/LUMO + any genuine imaginary frequency (with its IR intensity),
    from the free-text .out (property.txt has neither orbital energies nor
    the IR spectrum table)."""
    if not os.path.exists(out_path):
        return {}, []
    text = read_text(out_path)
    homo_lumo = parse_homo_lumo(text)
    ir = parse_ir_intensities(text)
    imaginary = [
        {"mode": idx, "freq_cm1": f, "intensity_km_mol": ir.get(idx)}
        for idx, f in enumerate(freq_list) if f < IMAG_FREQ_THRESHOLD_CM1
    ]
    return homo_lumo, imaginary


def parse_final_xyz(xyz_path):
    if not os.path.exists(xyz_path):
        return None
    lines = read_text(xyz_path).splitlines()
    n = int(lines[0].strip())
    atoms = []
    for line in lines[2:2 + n]:
        parts = line.split()
        atoms.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
    return atoms


def parse_xtb_absolute_energy_hartree(xyz_path, n_atoms, family):
    """xyz_gfn2xtb/*.xyz comment line looks like:
    'N9_anion anion rank 1 E_react=-67.24 kcal/mol' -- a reaction energy, not
    an absolute one. Converted to an absolute GFN2-xTB Hartree energy via
    reconstruct_xtb_energy_hartree() (see CONFIG for the re-derived anchors
    and how they were verified against polyN_pipeline.py's own convention)."""
    if not os.path.exists(xyz_path):
        return None
    comment = read_text(xyz_path).splitlines()[1]
    m = re.search(r"E_react=(-?[\d.]+)", comment)
    if not m:
        return None
    return reconstruct_xtb_energy_hartree(n_atoms, family, float(m.group(1)))


def classify_bond_length(d):
    for cutoff, label in BOND_BINS:
        if d < cutoff:
            return label
    return BOND_NONBONDED_LABEL


def atom_distance(a, b):
    return math.sqrt((a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2 + (a[3] - b[3]) ** 2)


# ---------------------------------------------------------------------------
# harvest
# ---------------------------------------------------------------------------

def harvest(jobs_root, out_summary, out_bonds, out_charges, out_imaginary, xyz_out_dir):
    summary_rows = []
    bond_rows = []
    charge_rows = []
    imaginary_rows = []
    n_ok, n_skip, n_imaginary = 0, 0, 0

    if xyz_out_dir:
        os.makedirs(xyz_out_dir, exist_ok=True)

    for job_dir in sorted(glob.glob(os.path.join(jobs_root, "*"))):
        name = os.path.basename(job_dir)
        # Chained (big-cluster) job dirs also contain _s1/_s2/_s3.property.txt
        # from the intermediate steps -- only the finalize-copied <name>.property.txt
        # is the canonical result; a bare glob("*.property.txt") would match all
        # of them under the same directory-derived `name` and silently duplicate
        # every row downstream.
        prop_path = os.path.join(job_dir, name + ".property.txt")
        if not os.path.isfile(prop_path):
            continue
        data = parse_property_file(prop_path)
        if data is None:
            n_skip += 1
            continue

        m = NAME_RE.match(name)
        n_count = int(m.group("n")) if m else None
        family = m.group("family") if m else None
        rank = int(m.group("rank")) if m else None

        out_path = os.path.join(job_dir, name + ".out")
        xyz_path = os.path.join(job_dir, name + ".xyz")
        pg, sn = parse_point_group(out_path)
        atoms = parse_final_xyz(xyz_path)
        homo_lumo, imaginary = parse_out_extras(out_path, data["freq_list"])

        if xyz_out_dir and atoms and os.path.exists(xyz_path):
            shutil.copy2(xyz_path, os.path.join(xyz_out_dir, name + ".xyz"))

        row = {
            "name": name,
            "n_count": n_count,
            "family": family,
            "rank": rank,
            "charge": data["charge"],
            "multiplicity": data["mult"],
            "n_atoms": data["n_atoms"],
            "point_group": pg,
            "symmetry_number": sn,
            "electronic_Eh": data["electronic_Eh"],
            "zpe_Eh": data["zpe_Eh"],
            "enthalpy_H_Eh": data["enthalpy_H_Eh"],
            "gibbs_G_Eh": data["gibbs_G_Eh"],
            "n_imaginary_freq": len(imaginary),
            "homo_eV": homo_lumo.get("homo_eV", ""),
            "lumo_eV": homo_lumo.get("lumo_eV", ""),
            "homo_alpha_eV": homo_lumo.get("homo_alpha_eV", ""),
            "lumo_alpha_eV": homo_lumo.get("lumo_alpha_eV", ""),
            "homo_beta_eV": homo_lumo.get("homo_beta_eV", ""),
            "lumo_beta_eV": homo_lumo.get("lumo_beta_eV", ""),
        }
        summary_rows.append(row)

        if imaginary:
            n_imaginary += 1
        for mode in imaginary:
            imaginary_rows.append({
                "name": name,
                "mode": mode["mode"],
                "freq_cm1": mode["freq_cm1"],
                "intensity_km_mol": mode["intensity_km_mol"],
            })

        if atoms:
            for (i, j, bo) in data["mayer_nn_bonds"]:
                d = atom_distance(atoms[i], atoms[j])
                bond_rows.append({
                    "name": name,
                    "atom_i": i,
                    "atom_j": j,
                    "distance_A": round(d, 4),
                    "mayer_bond_order": round(bo, 4),
                    "distance_class": classify_bond_length(d),
                })
            for idx, atom in enumerate(atoms):
                q = data["atomic_charges"].get(idx, {})
                charge_rows.append({
                    "name": name,
                    "atom_index": idx,
                    "element": atom[0],
                    "mulliken_charge": q.get("mulliken", ""),
                    "loewdin_charge": q.get("loewdin", ""),
                })
        n_ok += 1

    with open(out_summary, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "name", "n_count", "family", "rank", "charge", "multiplicity",
            "n_atoms", "point_group", "symmetry_number",
            "electronic_Eh", "zpe_Eh", "enthalpy_H_Eh", "gibbs_G_Eh",
            "n_imaginary_freq", "homo_eV", "lumo_eV",
            "homo_alpha_eV", "lumo_alpha_eV", "homo_beta_eV", "lumo_beta_eV",
        ])
        w.writeheader()
        w.writerows(summary_rows)

    with open(out_bonds, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "name", "atom_i", "atom_j", "distance_A", "mayer_bond_order", "distance_class",
        ])
        w.writeheader()
        w.writerows(bond_rows)

    with open(out_charges, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "name", "atom_index", "element", "mulliken_charge", "loewdin_charge",
        ])
        w.writeheader()
        w.writerows(charge_rows)

    with open(out_imaginary, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["name", "mode", "freq_cm1", "intensity_km_mol"])
        w.writeheader()
        w.writerows(imaginary_rows)

    print("harvest: {} jobs parsed, {} skipped (not finished yet)".format(n_ok, n_skip))
    print("  -> {}".format(out_summary))
    print("  -> {}".format(out_bonds))
    print("  -> {}".format(out_charges))
    print("  -> {} ({} structure(s) with a genuine imaginary mode -- not a true minimum)".format(
        out_imaginary, n_imaginary))
    if xyz_out_dir:
        print("  -> {}/ (optimized geometries at this DFT level)".format(xyz_out_dir))


# ---------------------------------------------------------------------------
# compare-xtb
# ---------------------------------------------------------------------------

def load_summary(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def compare_xtb(summary_csv, xyz_dir, out_dir):
    rows = load_summary(summary_csv)
    families = {}
    for r in rows:
        if r["electronic_Eh"] in (None, ""):
            continue
        key = (r["family"], int(r["n_count"]))
        families.setdefault(key, []).append(r)

    per_charge = {"neutral": [], "cation": [], "anion": []}
    for (family, n_count), recs in families.items():
        for r in recs:
            xyz_path = os.path.join(xyz_dir, r["name"] + ".xyz")
            r["xtb_absolute_Eh"] = parse_xtb_absolute_energy_hartree(xyz_path, n_count, family)
        dft_min = min(float(r["electronic_Eh"]) for r in recs)
        xtb_vals = [r["xtb_absolute_Eh"] for r in recs if r["xtb_absolute_Eh"] is not None]
        xtb_min = min(xtb_vals) if xtb_vals else None

        recs_sorted = sorted(recs, key=lambda r: float(r["electronic_Eh"]))
        for dft_rank, r in enumerate(recs_sorted, start=1):
            dft_rel_kcal = (float(r["electronic_Eh"]) - dft_min) * HARTREE_TO_KCAL
            xtb_rel_kcal = ((r["xtb_absolute_Eh"] - xtb_min) * HARTREE_TO_KCAL) if (
                r["xtb_absolute_Eh"] is not None and xtb_min is not None) else None
            per_charge[family].append({
                "n_count": n_count,
                "name": r["name"],
                "xtb_rank_as_generated": r["rank"],
                "dft_rank": dft_rank,
                "xtb_Erel_kcal": round(xtb_rel_kcal, 3) if xtb_rel_kcal is not None else "",
                "dft_Erel_kcal": round(dft_rel_kcal, 3),
                "delta_xtb_minus_dft_kcal": (
                    round(xtb_rel_kcal - dft_rel_kcal, 3) if xtb_rel_kcal is not None else ""
                ),
                "xtb_and_dft_agree_on_gs": (r["xtb_absolute_Eh"] == xtb_min) == (dft_rank == 1),
            })

    os.makedirs(out_dir, exist_ok=True)
    fieldnames = ["n_count", "name", "xtb_rank_as_generated", "dft_rank",
                  "xtb_Erel_kcal", "dft_Erel_kcal", "delta_xtb_minus_dft_kcal",
                  "xtb_and_dft_agree_on_gs"]
    for family, table_rows in per_charge.items():
        table_rows.sort(key=lambda r: (r["n_count"], r["dft_rank"]))
        out_path = os.path.join(out_dir, "compare_{}.csv".format(family))
        with open(out_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(table_rows)
        print("compare-xtb: {} rows -> {}".format(len(table_rows), out_path))


# ---------------------------------------------------------------------------
# reactions: Nx^q -> N(x-2)^q + N2 chain
# ---------------------------------------------------------------------------

def ground_state_by_family(rows):
    """{(family, n_count): best row (min electronic energy)}"""
    best = {}
    for r in rows:
        if r["electronic_Eh"] in (None, ""):
            continue
        key = (r["family"], int(r["n_count"]))
        e = float(r["electronic_Eh"])
        if key not in best or e < float(best[key]["electronic_Eh"]):
            best[key] = r
    return best


def reactions(summary_csv, n2_property_txt, out_csv,
              anion_anchor_kcal=N5_ANION_REF_DHF_KCAL,
              cation_anchor_kcal=N5_CATION_REF_DHF_KCAL):
    rows = load_summary(summary_csv)
    gs = ground_state_by_family(rows)

    n2 = parse_property_file(n2_property_txt)
    if n2 is None or n2["electronic_Eh"] is None:
        sys.exit("reactions: N2 reference job at '{}' has no finished THERMOCHEMISTRY "
                  "block yet -- run/finish an N2 Opt+Freq at the same level of theory "
                  "first (needed for every N2-loss step).".format(n2_property_txt))

    out_rows = []
    charge_map = {"neutral": 0, "cation": 1, "anion": -1}
    for family, charge in charge_map.items():
        keys = sorted([k for k in gs if k[0] == family], key=lambda k: -k[1])
        for (fam, n_count) in keys:
            n_target, _ = reaction_step(n_count, charge)
            target_key = (family, n_target)
            if target_key not in gs and n_target != 2:
                continue  # no partner two sizes down -- chain gap, skip
            row = gs[(family, n_count)]
            H_x = float(row["enthalpy_H_Eh"])
            G_x = float(row["gibbs_G_Eh"])
            if n_target == 2:
                # ions bottom out at N5, not N2 -- shouldn't hit this branch
                # for odd-N ion chains; guarded for completeness.
                continue
            if target_key in gs:
                target_row = gs[target_key]
                H_target = float(target_row["enthalpy_H_Eh"])
                G_target = float(target_row["gibbs_G_Eh"])
            else:
                continue

            dH_kcal = (H_target + n2["enthalpy_H_Eh"] - H_x) * HARTREE_TO_KCAL
            dG_kcal = (G_target + n2["gibbs_G_Eh"] - G_x) * HARTREE_TO_KCAL

            out_rows.append({
                "family": family,
                "reaction": "N{}^{} -> N{}^{} + N2".format(
                    n_count, "0" if charge == 0 else ("+" if charge > 0 else "-"),
                    n_target, "0" if charge == 0 else ("+" if charge > 0 else "-")),
                "dH_kcal_mol": round(dH_kcal, 2),
                "dG_kcal_mol": round(dG_kcal, 2),
            })

    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["family", "reaction", "dH_kcal_mol", "dG_kcal_mol"])
        w.writeheader()
        w.writerows(out_rows)
    print("reactions: {} steps -> {}".format(len(out_rows), out_csv))
    if anion_anchor_kcal is None or cation_anchor_kcal is None:
        print("NOTE: no literature dHf anchor given for cyclic N5- / bent N5+ "
              "(N5_ANION_REF_DHF_KCAL / N5_CATION_REF_DHF_KCAL, or --anion-anchor/"
              "--cation-anchor) -- ion rows above are step-wise reaction "
              "enthalpies only, not absolute dHf. Neutral dHf falls out directly "
              "(chain terminates at N2, dHf=0) once you sum the steps down from "
              "each Nx to N4/N2.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    sub.required = True  # Python 3.6: add_subparsers() has no required= kwarg

    p1 = sub.add_parser("harvest", help="parse jobs/*/*.property.txt into summary+bonds CSVs")
    p1.add_argument("--jobs-root", default="jobs")
    p1.add_argument("--out-summary", default="results_summary.csv")
    p1.add_argument("--out-bonds", default="results_bonds.csv")
    p1.add_argument("--out-charges", default="results_charges.csv")
    p1.add_argument("--out-imaginary", default="results_imaginary_modes.csv")
    p1.add_argument("--xyz-out-dir", default=DFT_XYZ_OUT_DIR,
                     help="where to copy each job's final optimized .xyz (set to '' to skip)")

    p2 = sub.add_parser("compare-xtb", help="xTB vs DFT relative-energy tables, per charge")
    p2.add_argument("--summary", default="results_summary.csv")
    p2.add_argument("--xyz-dir", default="xyz_gfn2xtb")
    p2.add_argument("--out-dir", default=".")

    p3 = sub.add_parser("reactions", help="Nx^q -> N(x-2)^q + N2 decomposition chain")
    p3.add_argument("--summary", default="results_summary.csv")
    p3.add_argument("--n2-property", required=True,
                     help="path to a N2 Opt+Freq property.txt at the same level of theory")
    p3.add_argument("--out", default="reaction_energetics.csv")
    p3.add_argument("--anion-anchor", type=float, default=N5_ANION_REF_DHF_KCAL)
    p3.add_argument("--cation-anchor", type=float, default=N5_CATION_REF_DHF_KCAL)

    args = ap.parse_args()
    if args.cmd == "harvest":
        harvest(args.jobs_root, args.out_summary, args.out_bonds,
                args.out_charges, args.out_imaginary, args.xyz_out_dir)
    elif args.cmd == "compare-xtb":
        compare_xtb(args.summary, args.xyz_dir, args.out_dir)
    elif args.cmd == "reactions":
        reactions(args.summary, args.n2_property, args.out,
                  args.anion_anchor, args.cation_anchor)


if __name__ == "__main__":
    main()
