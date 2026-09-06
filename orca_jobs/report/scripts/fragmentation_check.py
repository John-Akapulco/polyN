import glob, os, re, csv, itertools
import numpy as np
import networkx as nx

CUTOFF = 1.70  # distances above this don't count as a real N-N bond for
               # connectivity purposes -- the point is to catch structures
               # that are really two (or more) separate molecular species
               # (e.g. two ions) held together only by a weak contact.

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

rows = []
for xtb_path in sorted(glob.glob(f"{XTB_DIR}/*.xyz")):
    name = os.path.splitext(os.path.basename(xtb_path))[0]
    dft_path = f"{DFT_DIR}/{name}.xyz"
    if os.path.exists(dft_path):
        path, level = dft_path, "DFT"
    else:
        path, level = xtb_path, "xtb"
    comps = fragments(read_xyz(path))
    rows.append({"name": name, "level": level, "n_fragments": len(comps),
                 "fragment_sizes": "+".join(map(str, comps))})

with open("/home/gilles/polyN/orca_jobs/results_fragmentation.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["name","level","n_fragments","fragment_sizes"])
    w.writeheader(); w.writerows(rows)

frag = [r for r in rows if r["n_fragments"] > 1]
print(f"{len(frag)} / {len(rows)} structures split into >1 fragment at the 1.70 A cutoff")
for r in frag:
    print(" ", r["name"], r["level"], r["fragment_sizes"])
