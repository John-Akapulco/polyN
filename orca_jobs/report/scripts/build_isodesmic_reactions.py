#!/usr/bin/env python3
"""Reactions quasi-isodesmiques (demandees par l'utilisateur, 2026-09-07) :
  - neutre  : Nx -> (x/2) N2
  - anion   : Nx- + H2 -> NH2- + ((x-1)/2) N2
  - cation  : Nx+ + 2 H2 -> NH4+ + ((x-1)/2) N2
  - anion   : Nx- -> N5- + ((x-5)/2) N2          (reference N5- = D5h, N5_anion_anion_001)
  - cation  : Nx+ -> N5+ + ((x-5)/2) N2          (reference N5+ = C2v, N5_cation_cation_001,
                                                    structure litterature -- PAS N5_cation_cation_004,
                                                    reference interne differente du pipeline)

Pour chaque groupe (n, charge), la structure utilisee est celle DFT-la-plus-
stable (multifidelity_xtb_dft.csv, epa_dft_ev minimal) -- MEME structure aux
3 niveaux (son energie xTB, DFT, et CCSD(T) si disponible), pour isoler
l'effet de la methode plutot que celui d'un changement d'isomere entre
niveaux. Le niveau CCSD(T) n'a de point que pour les groupes dont cette
structure precise a ete recalculee (couverture partielle, ~81 structures
sur 193 a ce jour) -- absence de point CCSD(T) = donnee non disponible,
jamais interpolee.
"""
import csv
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/gilles/polyN/orca_jobs"
FIG_DIR = f"{ROOT}/report/figures"
HARTREE_TO_KCAL = 627.5094740631
EV_TO_HARTREE = 1 / 27.211386245988

# --- references hors criblage (calculees/rassemblees le 2026-09-07) ---
REFS = {
    "N2":  {"xtb": -5.763935,        "dft": -109.596404831237, "ccsdt": -109.412670614219},
    "H2":  {"xtb": -0.98268617,      "dft": None,               "ccsdt": None},  # complete plus bas si H2 termine
    "NH2-": {"xtb": -4.00317592,     "dft": -55.945130926706,  "ccsdt": -55.840097763678},
    "NH4+": {"xtb": -4.49487674,     "dft": -56.944012617197,  "ccsdt": -56.835980050637},
    "N5-": {"xtb": -14.760899,       "dft": -273.918404229172, "ccsdt": -273.441045726432},
    "N5+": {"xtb": -13.985267,       "dft": -273.444542650912, "ccsdt": -272.976859063786},
}

# --- injecte H2 DFT/CCSD(T) si le job est termine ---
h2_resym = f"{ROOT}/ccsdt_jobs/H2/H2_resym.out"
if os.path.isfile(h2_resym):
    txt = open(h2_resym, errors="ignore").read()
    if "THE OPTIMIZATION HAS CONVERGED" in txt:
        m = re.search(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", txt)
        if m:
            REFS["H2"]["dft"] = float(m.group(1))
h2_ccsdt = f"{ROOT}/ccsdt_jobs/H2/H2_ccsdt.out"
if os.path.isfile(h2_ccsdt):
    txt = open(h2_ccsdt, errors="ignore").read()
    m = re.search(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", txt)
    if m:
        REFS["H2"]["ccsdt"] = float(m.group(1))

missing_refs = [k for k, v in REFS.items() if v["dft"] is None]
if missing_refs:
    print(f"ATTENTION : references incompletes (DFT manquant) : {missing_refs} -- "
          f"les reactions H2-dependantes n'auront pas de courbe DFT/CCSD(T) tant que non resolu.")

# --- 1) structure DFT-la-plus-stable par groupe (n, charge), avec ses 3 energies ---
groups = {}  # (n, family) -> dict(name, xtb_Ha, dft_Ha)
with open(f"{ROOT}/multifidelity_xtb_dft.csv") as f:
    for row in csv.DictReader(f):
        n = int(row["n_atoms"])
        fam = row["family"]
        epa_dft = float(row["epa_dft_ev"])
        key = (n, fam)
        if key not in groups or epa_dft < groups[key]["epa_dft_ev"]:
            groups[key] = {
                "name": row["name"], "epa_dft_ev": epa_dft,
                "epa_xtb_ev": float(row["epa_xtb_ev"]),
                "xtb_Ha": float(row["epa_xtb_ev"]) * n * EV_TO_HARTREE,
                "dft_Ha": epa_dft * n * EV_TO_HARTREE,
            }

# --- 2) energie CCSD(T) de cette structure precise, si disponible ---
ccsdt_E = {}
for d in glob.glob(f"{ROOT}/ccsdt_jobs/*/"):
    name = os.path.basename(d.rstrip("/"))
    out = f"{d}{name}_ccsdt.out"
    if not os.path.isfile(out):
        continue
    txt = open(out, errors="ignore").read()
    m = re.search(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", txt)
    if m:
        ccsdt_E[name] = float(m.group(1))

for (n, fam), g in groups.items():
    g["ccsdt_Ha"] = ccsdt_E.get(g["name"])

CHARGE_OF = {"neutral": 0, "anion": -1, "cation": 1}


def reaction_neutral(n, level):
    g = groups.get((n, "neutral"))
    if g is None or g.get(f"{level}_Ha") is None:
        return None
    E_n2 = REFS["N2"][level]
    if E_n2 is None:
        return None
    dH_Ha = (n / 2) * E_n2 - g[f"{level}_Ha"]
    return dH_Ha * HARTREE_TO_KCAL


def reaction_anion_h2(n, level):
    g = groups.get((n, "anion"))
    if g is None or g.get(f"{level}_Ha") is None:
        return None
    k = (n - 1) / 2
    E_n2, E_h2, E_nh2 = REFS["N2"][level], REFS["H2"][level], REFS["NH2-"][level]
    if None in (E_n2, E_h2, E_nh2):
        return None
    dH_Ha = E_nh2 + k * E_n2 - g[f"{level}_Ha"] - E_h2
    return dH_Ha * HARTREE_TO_KCAL


def reaction_cation_h2(n, level):
    g = groups.get((n, "cation"))
    if g is None or g.get(f"{level}_Ha") is None:
        return None
    k = (n - 1) / 2
    E_n2, E_h2, E_nh4 = REFS["N2"][level], REFS["H2"][level], REFS["NH4+"][level]
    if None in (E_n2, E_h2, E_nh4):
        return None
    dH_Ha = E_nh4 + k * E_n2 - g[f"{level}_Ha"] - 2 * E_h2
    return dH_Ha * HARTREE_TO_KCAL


def reaction_anion_n5(n, level):
    g = groups.get((n, "anion"))
    if g is None or g.get(f"{level}_Ha") is None:
        return None
    k = (n - 5) / 2
    E_n2, E_n5m = REFS["N2"][level], REFS["N5-"][level]
    if None in (E_n2, E_n5m):
        return None
    dH_Ha = E_n5m + k * E_n2 - g[f"{level}_Ha"]
    return dH_Ha * HARTREE_TO_KCAL


def reaction_cation_n5(n, level):
    g = groups.get((n, "cation"))
    if g is None or g.get(f"{level}_Ha") is None:
        return None
    k = (n - 5) / 2
    E_n2, E_n5p = REFS["N2"][level], REFS["N5+"][level]
    if None in (E_n2, E_n5p):
        return None
    dH_Ha = E_n5p + k * E_n2 - g[f"{level}_Ha"]
    return dH_Ha * HARTREE_TO_KCAL


REACTIONS = [
    ("neutral_N2", "Neutre : $N_x \\to (x/2)\\,N_2$", reaction_neutral, "neutral"),
    ("anion_H2", "Anion : $N_x^- + H_2 \\to NH_2^- + \\frac{x-1}{2}\\,N_2$", reaction_anion_h2, "anion"),
    ("cation_H2", "Cation : $N_x^+ + 2H_2 \\to NH_4^+ + \\frac{x-1}{2}\\,N_2$", reaction_cation_h2, "cation"),
    ("anion_N5", "Anion : $N_x^- \\to N_5^- + \\frac{x-5}{2}\\,N_2$", reaction_anion_n5, "anion"),
    ("cation_N5", "Cation : $N_x^+ \\to N_5^+ + \\frac{x-5}{2}\\,N_2$", reaction_cation_n5, "cation"),
]

LEVELS = [("xtb", "GFN2-xTB", "o", "#4C72B0"), ("dft", "WB97X-D4/aug-cc-pVTZ", "s", "#DD8452"),
          ("ccsdt", "DLPNO-CCSD(T)-F12", "^", "#55A868")]

all_n = sorted(set(n for (n, fam) in groups))

print(f"{'reaction':<12} {'niveau':<8} {'n points':>8}")
for slug, label, fn, fam in REACTIONS:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for level, level_label, marker, color in LEVELS:
        xs, ys = [], []
        for n in all_n:
            if CHARGE_OF[fam] == 0 and n % 2 != 0:
                continue
            if CHARGE_OF[fam] != 0 and n % 2 == 0:
                continue
            v = fn(n, level)
            if v is not None:
                xs.append(n)
                ys.append(v)
        print(f"{slug:<12} {level:<8} {len(xs):>8}")
        if xs:
            ax.plot(xs, ys, marker=marker, color=color, label=level_label,
                     linewidth=1.6, markersize=6, alpha=0.9)
    ax.set_xlabel("Nombre d'atomes d'azote $x$")
    ax.set_ylabel(r"$\Delta H_{r\acute{e}action}$ (kcal/mol)")
    ax.set_title(label, fontsize=11)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/isodesmic_{slug}.pdf")
    plt.close(fig)

print(f"\nFigures ecrites dans {FIG_DIR}/")
