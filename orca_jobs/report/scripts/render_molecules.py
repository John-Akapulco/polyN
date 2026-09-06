import csv, os, math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

OUT_DIR = "/home/gilles/polyN/orca_jobs/report/figs"
DFT_DIR = "/home/gilles/polyN/orca_jobs/xyz_dft_wb97xd4"
XTB_DIR = "/home/gilles/polyN/orca_jobs/xyz_gfn2xtb"
BOND_CUTOFF = 1.90


def read_xyz(path):
    lines = open(path).read().splitlines()
    n = int(lines[0].strip())
    coords = []
    for line in lines[2:2 + n]:
        p = line.split()
        coords.append((float(p[1]), float(p[2]), float(p[3])))
    return np.array(coords)


def rot_matrix(rx, ry):
    rx, ry = math.radians(rx), math.radians(ry)
    Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    return Ry @ Rx


def render(xyz_path, out_png):
    coords = read_xyz(xyz_path)
    coords -= coords.mean(axis=0)
    R = rot_matrix(15, 25)
    rc = coords @ R.T
    n = len(rc)
    bonds = [(i, j) for i in range(n) for j in range(i + 1, n)
             if np.linalg.norm(coords[i] - coords[j]) < BOND_CUTOFF]

    fig, ax = plt.subplots(figsize=(2.3, 2.3))
    zmin, zmax = rc[:, 2].min(), rc[:, 2].max()
    zspan = max(zmax - zmin, 1e-6)
    items = [("atom", i, rc[i, 2]) for i in range(n)]
    items += [("bond", (i, j), (rc[i, 2] + rc[j, 2]) / 2) for i, j in bonds]
    items.sort(key=lambda t: t[2])

    for kind, idx, z in items:
        shade = 0.35 + 0.55 * (z - zmin) / zspan
        color = (0.12 * shade + 0.05, 0.35 * shade + 0.05, 0.75 * shade + 0.15)
        if kind == "atom":
            x, y, _ = rc[idx]
            rad = 0.34 + 0.10 * (z - zmin) / zspan
            ax.add_patch(Circle((x, y), rad, facecolor=color, edgecolor="black", linewidth=0.7, zorder=idx))
        else:
            i, j = idx
            ax.plot([rc[i, 0], rc[j, 0]], [rc[i, 1], rc[j, 1]], color=color, linewidth=2.2, solid_capstyle="round")

    lim = np.abs(rc[:, :2]).max() + 0.6
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal"); ax.set_axis_off()
    fig.tight_layout(pad=0.05)
    fig.savefig(out_png, dpi=170, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# --- S1-S3: confirmed minima, excluding fragmented ---
frag_rows = {r["name"]: r for r in csv.DictReader(
    open("/home/gilles/polyN/orca_jobs/results_fragmentation.csv"))}
FRAGMENTED = {n for n, r in frag_rows.items() if int(r["n_fragments"]) > 1}

rows = list(csv.DictReader(open("/home/gilles/polyN/orca_jobs/results_summary.csv")))
confirmed = [r for r in rows if r["n_imaginary_freq"] == "0" and r["name"] not in FRAGMENTED]

names = []
for r in confirmed:
    path = os.path.join(DFT_DIR, r["name"] + ".xyz")
    if not os.path.exists(path):
        continue
    render(path, os.path.join(OUT_DIR, r["name"] + ".png"))
    names.append((r["name"], r["point_group"]))

with open(os.path.join(OUT_DIR, "manifest.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["name", "point_group"]); w.writerows(names)
print("S1-S3 rendered (non-fragmented confirmed minima):", len(names))

# --- Figure S4: fragmented structures, before (xTB seed) / after (DFT) ---
FRAG_DIR = os.path.join(OUT_DIR, "fragmented")
os.makedirs(FRAG_DIR, exist_ok=True)
frag_names = []
for name, r in frag_rows.items():
    if int(r["n_fragments"]) <= 1 or r["level"] != "DFT":
        continue  # only ones where DFT actually finished have a real "after" to show
    xtb_path = os.path.join(XTB_DIR, name + ".xyz")
    dft_path = os.path.join(DFT_DIR, name + ".xyz")
    if not (os.path.exists(xtb_path) and os.path.exists(dft_path)):
        continue
    render(xtb_path, os.path.join(FRAG_DIR, name + "_initial.png"))
    render(dft_path, os.path.join(FRAG_DIR, name + "_final.png"))
    frag_names.append(name)

with open(os.path.join(FRAG_DIR, "manifest.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["name", "fragment_sizes"])
    w.writerows((n, frag_rows[n]["fragment_sizes"]) for n in frag_names)
print("Figure S4 rendered (fragmented, before/after):", len(frag_names))

# --- Figure S2 (non convergees) : avant (xTB de depart) / apres (derniere
# geometrie tentee, cycle limite atteint sans convergence) ---
NC_XYZ_DIR = "/home/gilles/polyN/orca_jobs/xyz_dft_nonconverged"
NC_DIR = os.path.join(OUT_DIR, "nonconverged")
os.makedirs(NC_DIR, exist_ok=True)
nc_rows = list(csv.DictReader(open("/home/gilles/polyN/orca_jobs/results_nonconverged.csv")))
nc_names = []
for r in nc_rows:
    name = r["name"]
    xtb_path = os.path.join(XTB_DIR, name + ".xyz")
    last_path = os.path.join(NC_XYZ_DIR, name + ".xyz")
    if not (os.path.exists(xtb_path) and os.path.exists(last_path)):
        continue
    render(xtb_path, os.path.join(NC_DIR, name + "_initial.png"))
    render(last_path, os.path.join(NC_DIR, name + "_last.png"))
    nc_names.append(name)

with open(os.path.join(NC_DIR, "manifest.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["name", "n_cycles"])
    w.writerows((n, next(r["n_cycles"] for r in nc_rows if r["name"] == n)) for n in nc_names)
print("Figure S2 rendered (non convergees, before/last cycle):", len(nc_names))
