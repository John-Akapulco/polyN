import glob, os, csv, itertools
import numpy as np
import networkx as nx

CUTOFF = 1.70

def read_xyz(path):
    lines = open(path).read().splitlines()
    n = int(lines[0].strip())
    coords = []
    for line in lines[2:2+n]:
        p = line.split()
        coords.append((float(p[1]), float(p[2]), float(p[3])))
    return np.array(coords)

def fragments(coords):
    n = len(coords)
    g = nx.Graph(); g.add_nodes_from(range(n))
    for i, j in itertools.combinations(range(n), 2):
        if np.linalg.norm(coords[i]-coords[j]) < CUTOFF:
            g.add_edge(i, j)
    comps = list(nx.connected_components(g))
    return sorted(len(c) for c in comps)

DFT_DIR = "/home/gilles/polyN/orca_jobs/xyz_dft_wb97xd4"
XTB_DIR = "/home/gilles/polyN/orca_jobs/xyz_gfn2xtb"

# Check EVERY structure's xTB starting geometry independently of whatever
# level results_fragmentation.csv reports as "official" -- the point is to
# catch cases where the DFT-fragmented product was already fragmented
# before DFT ever ran (a pipeline/screening bug), vs. genuinely fragmenting
# during DFT relaxation.
rows = []
for xtb_path in sorted(glob.glob(f"{XTB_DIR}/*.xyz")):
    name = os.path.splitext(os.path.basename(xtb_path))[0]
    comps_xtb = fragments(read_xyz(xtb_path))
    dft_path = f"{DFT_DIR}/{name}.xyz"
    comps_dft = fragments(read_xyz(dft_path)) if os.path.exists(dft_path) else None
    rows.append({
        "name": name,
        "n_fragments_xtb_initial": len(comps_xtb),
        "fragment_sizes_xtb_initial": "+".join(map(str, comps_xtb)),
        "n_fragments_dft": len(comps_dft) if comps_dft is not None else "",
        "fragment_sizes_dft": "+".join(map(str, comps_dft)) if comps_dft is not None else "",
    })

with open("/home/gilles/polyN/orca_jobs/results_fragmentation_xtb_check.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["name","n_fragments_xtb_initial","fragment_sizes_xtb_initial",
                                        "n_fragments_dft","fragment_sizes_dft"])
    w.writeheader(); w.writerows(rows)

already_frag_at_xtb = [r for r in rows if r["n_fragments_xtb_initial"] > 1]
became_frag_at_dft = [r for r in rows if r["n_fragments_xtb_initial"] == 1
                       and r["n_fragments_dft"] not in ("", 1)]

print(f"{len(already_frag_at_xtb)} structures already fragmented at the xTB starting geometry:")
for r in already_frag_at_xtb:
    print("  ", r["name"], "xtb:", r["fragment_sizes_xtb_initial"],
          "-> dft:", r["fragment_sizes_dft"] or "(no DFT geometry yet)")

print(f"\n{len(became_frag_at_dft)} structures intact at xTB but fragmented after DFT relaxation:")
for r in became_frag_at_dft:
    print("  ", r["name"], "xtb: intact ->  dft:", r["fragment_sizes_dft"])
