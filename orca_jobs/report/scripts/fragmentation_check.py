import glob, os, itertools, csv
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
    return sorted(len(c) for c in nx.connected_components(g))

DFT_DIR = "/home/gilles/polyN/orca_jobs/xyz_dft_wb97xd4"
XTB_DIR = "/home/gilles/polyN/orca_jobs/xyz_gfn2xtb"

# Every structure is checked at BOTH levels independently (not "DFT if
# available else xTB") -- this is what lets us tell apart a candidate that
# was already fragmented in the xTB starting geometry (a screening-pipeline
# artifact upstream of this campaign) from one that fragmented only during
# DFT relaxation (a genuine outcome of this campaign's optimization).
rows = []
for xtb_path in sorted(glob.glob(f"{XTB_DIR}/*.xyz")):
    name = os.path.splitext(os.path.basename(xtb_path))[0]
    comps_xtb = fragments(read_xyz(xtb_path))
    dft_path = f"{DFT_DIR}/{name}.xyz"
    comps_dft = fragments(read_xyz(dft_path)) if os.path.exists(dft_path) else None

    if comps_dft is not None:
        level, comps = "DFT", comps_dft
    else:
        level, comps = "xtb", comps_xtb

    rows.append({
        "name": name,
        "level": level,
        "n_fragments": len(comps),
        "fragment_sizes": "+".join(map(str, comps)),
        "n_fragments_xtb_initial": len(comps_xtb),
        "fragment_sizes_xtb_initial": "+".join(map(str, comps_xtb)),
        "n_fragments_dft": len(comps_dft) if comps_dft is not None else "",
        "fragment_sizes_dft": "+".join(map(str, comps_dft)) if comps_dft is not None else "",
        "already_fragmented_at_xtb": len(comps_xtb) > 1,
    })

fieldnames = ["name", "level", "n_fragments", "fragment_sizes",
              "n_fragments_xtb_initial", "fragment_sizes_xtb_initial",
              "n_fragments_dft", "fragment_sizes_dft", "already_fragmented_at_xtb"]
with open("/home/gilles/polyN/orca_jobs/results_fragmentation.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fieldnames)
    w.writeheader(); w.writerows(rows)

frag = [r for r in rows if r["n_fragments"] > 1]
already_at_xtb = [r for r in rows if r["already_fragmented_at_xtb"]]
became_at_dft = [r for r in frag if not r["already_fragmented_at_xtb"]]
print(f"{len(frag)} / {len(rows)} structures reported fragmented (>1 component, "
      f"DFT geometry if available else xTB, cutoff {CUTOFF} A)")
print(f"  of which {len(already_at_xtb)} were ALREADY fragmented in the xTB "
      f"starting geometry (upstream screening artifact, not a DFT outcome)")
print(f"  and {len(became_at_dft)} fragmented only during DFT relaxation "
      f"(intact at xTB)")
for r in frag:
    tag = "already-at-xTB" if r["already_fragmented_at_xtb"] else "DFT-only"
    print("  ", r["name"], r["level"], r["fragment_sizes"], f"[{tag}]")
