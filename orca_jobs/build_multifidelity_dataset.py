"""Exporte les jeux de donnees d'entrainement pour la correction
multi-fidelite du surrogate pool 2 : Delta(DFT-xTB) sur les structures
confirmees de la campagne 1, et Delta(CCSD(T)-DFT) sur les structures
terminees de la campagne 2. Descripteurs = exactement ceux de
core/graph_descriptors.py (memes 11 features, meme graphe N-N a 1.90 A),
pour rester directement utilisables par AdaptiveSurrogate."""
import csv, glob, itertools, os, re, sys

import numpy as np
import networkx as nx

sys.path.insert(0, "/home/gilles/polyN")
from core.graph_descriptors import compute_descriptors, DEFAULT_FEATURE_ORDER

ROOT = "/home/gilles/polyN/orca_jobs"
HARTREE_TO_EV = 27.211386245988
BOND_CUTOFF = 1.90  # coherent avec harvest_results.py (seuil non-liant)

NAME_RE = re.compile(r"^N(?P<n>\d+)_(?P<family>neutral|cation|anion)_(?P=family)_(?P<rank>\d+)$")
XTB_E_N2_HARTREE = -5.763935420175
N5_ANION_REF_HARTREE = -14.760899219951
N5_CATION_REF_HARTREE = -13.816934622594  # reference interne (isomere C1), pas le C2v litterature


def reconstruct_xtb_energy_hartree(n_atoms, family, e_react_kcal):
    e_react_ha = e_react_kcal / 627.5094740631
    if family == "neutral":
        return e_react_ha + (n_atoms / 2.0) * XTB_E_N2_HARTREE
    if family == "anion":
        coeff_n2 = (n_atoms - 5) / 2.0
        return e_react_ha + coeff_n2 * XTB_E_N2_HARTREE + N5_ANION_REF_HARTREE
    coeff_n2 = (n_atoms - 5) / 2.0
    return e_react_ha + coeff_n2 * XTB_E_N2_HARTREE + N5_CATION_REF_HARTREE


def read_xyz(path):
    lines = open(path).read().splitlines()
    n = int(lines[0].strip())
    coords = [tuple(map(float, lines[i].split()[1:4])) for i in range(2, 2 + n)]
    return np.array(coords)


def graph_from_xyz(path):
    coords = read_xyz(path)
    n = len(coords)
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for i, j in itertools.combinations(range(n), 2):
        if np.linalg.norm(coords[i] - coords[j]) < BOND_CUTOFF:
            g.add_edge(i, j)
    return g


# --- 1) Delta(DFT - xTB), campagne 1 ---
frag = {r["name"] for r in csv.DictReader(open(f"{ROOT}/results_fragmentation.csv")) if int(r["n_fragments"]) > 1}
summ = {r["name"]: r for r in csv.DictReader(open(f"{ROOT}/results_summary.csv"))}

rows_dft = []
for xtb_path in sorted(glob.glob(f"{ROOT}/xyz_gfn2xtb/*.xyz")):
    name = os.path.splitext(os.path.basename(xtb_path))[0]
    m = NAME_RE.match(name)
    if not m or name in frag or name not in summ:
        continue
    r = summ[name]
    if not r.get("electronic_Eh"):
        continue
    n_atoms = int(m.group("n"))
    family = m.group("family")
    comment = open(xtb_path).read().splitlines()[1]
    em = re.search(r"E_react=(-?[\d.]+)", comment)
    if not em:
        continue
    e_xtb_ha = reconstruct_xtb_energy_hartree(n_atoms, family, float(em.group(1)))
    e_dft_ha = float(r["electronic_Eh"])
    try:
        desc = compute_descriptors(graph_from_xyz(xtb_path)).as_dict()
    except Exception as e:
        print(f"descripteur echoue pour {name}: {e}", file=sys.stderr)
        continue
    epa_xtb = e_xtb_ha * HARTREE_TO_EV / n_atoms
    epa_dft = e_dft_ha * HARTREE_TO_EV / n_atoms
    row = {"name": name, "n_atoms": n_atoms, "family": family,
           "epa_xtb_ev": epa_xtb, "epa_dft_ev": epa_dft,
           "delta_dft_minus_xtb_ev": epa_dft - epa_xtb}
    row.update({f"desc_{k}": desc[k] for k in DEFAULT_FEATURE_ORDER})
    rows_dft.append(row)

with open("/home/gilles/polyN/orca_jobs/multifidelity_xtb_dft.csv", "w", newline="") as fh:
    fieldnames = ["name", "n_atoms", "family", "epa_xtb_ev", "epa_dft_ev", "delta_dft_minus_xtb_ev"] + \
                 [f"desc_{k}" for k in DEFAULT_FEATURE_ORDER]
    w = csv.DictWriter(fh, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows_dft)
print(f"multifidelity_xtb_dft.csv : {len(rows_dft)} structures "
      f"(neutral={sum(1 for r in rows_dft if r['family']=='neutral')}, "
      f"cation={sum(1 for r in rows_dft if r['family']=='cation')}, "
      f"anion={sum(1 for r in rows_dft if r['family']=='anion')})")

# --- 2) Delta(CCSD(T) - DFT), campagne 2 ---
rows_ccsdt = []
for d in glob.glob(f"{ROOT}/ccsdt_jobs/*/"):
    name = os.path.basename(d.rstrip("/"))
    m = NAME_RE.match(name)
    if not m:
        continue  # N2, NH4_cation, NH2_anion : pas de graphe N-N exploitable ici
    ccsdt_out = f"{d}{name}_ccsdt.out"
    resym_xyz = f"{d}{name}_resym.xyz"
    if not (os.path.isfile(ccsdt_out) and os.path.isfile(resym_xyz)):
        continue
    txt = open(ccsdt_out, errors="ignore").read()
    em = re.search(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", txt)
    if not em:
        continue
    e_ccsdt_ha = float(em.group(1))
    # energie DFT sur cette meme geometrie resymetrisee (coherent avec le
    # single-point CCSD(T), pas l'energie DFT d'origine avant resym)
    resym_out = f"{d}{name}_resym.out"
    dft_txt = open(resym_out, errors="ignore").read()
    dm = re.findall(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", dft_txt)
    if not dm:
        continue
    e_dft_resym_ha = float(dm[-1])
    n_atoms = int(m.group("n"))
    try:
        desc = compute_descriptors(graph_from_xyz(resym_xyz)).as_dict()
    except Exception as e:
        print(f"descripteur echoue pour {name} (ccsdt): {e}", file=sys.stderr)
        continue
    epa_dft = e_dft_resym_ha * HARTREE_TO_EV / n_atoms
    epa_ccsdt = e_ccsdt_ha * HARTREE_TO_EV / n_atoms
    row = {"name": name, "n_atoms": n_atoms, "family": m.group("family"),
           "epa_dft_ev": epa_dft, "epa_ccsdt_ev": epa_ccsdt,
           "delta_ccsdt_minus_dft_ev": epa_ccsdt - epa_dft}
    row.update({f"desc_{k}": desc[k] for k in DEFAULT_FEATURE_ORDER})
    rows_ccsdt.append(row)

with open("/home/gilles/polyN/orca_jobs/multifidelity_dft_ccsdt.csv", "w", newline="") as fh:
    fieldnames = ["name", "n_atoms", "family", "epa_dft_ev", "epa_ccsdt_ev", "delta_ccsdt_minus_dft_ev"] + \
                 [f"desc_{k}" for k in DEFAULT_FEATURE_ORDER]
    w = csv.DictWriter(fh, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows_ccsdt)
print(f"multifidelity_dft_ccsdt.csv : {len(rows_ccsdt)} structures")
