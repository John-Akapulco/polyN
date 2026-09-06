import csv, os, math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

OUT_DIR = "/home/gilles/polyN/orca_jobs/report/figs"
XYZ_DIR = "/home/gilles/polyN/orca_jobs/xyz_dft_wb97xd4"
BOND_CUTOFF = 1.90

def read_xyz(path):
    lines = open(path).read().splitlines()
    n = int(lines[0].strip())
    coords = []
    for line in lines[2:2+n]:
        p = line.split()
        coords.append((float(p[1]), float(p[2]), float(p[3])))
    return np.array(coords)

def rot_matrix(rx, ry):
    rx, ry = math.radians(rx), math.radians(ry)
    Rx = np.array([[1,0,0],[0,math.cos(rx),-math.sin(rx)],[0,math.sin(rx),math.cos(rx)]])
    Ry = np.array([[math.cos(ry),0,math.sin(ry)],[0,1,0],[-math.sin(ry),0,math.cos(ry)]])
    return Ry @ Rx

rows = list(csv.DictReader(open("/home/gilles/polyN/orca_jobs/results_summary.csv")))
confirmed = [r for r in rows if r["n_imaginary_freq"] == "0"]

names = []
for r in confirmed:
    path = os.path.join(XYZ_DIR, r["name"] + ".xyz")
    if not os.path.exists(path):
        continue
    coords = read_xyz(path)
    coords -= coords.mean(axis=0)
    R = rot_matrix(15, 25)
    rc = coords @ R.T  # rotated coords, columns x,y,z (z = depth, camera looks down -z)

    order = np.argsort(rc[:, 2])  # back-to-front painter's algorithm
    n = len(rc)
    bonds = []
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(coords[i] - coords[j]) < BOND_CUTOFF:
                bonds.append((i, j))

    fig, ax = plt.subplots(figsize=(2.3, 2.3))
    zmin, zmax = rc[:, 2].min(), rc[:, 2].max()
    zspan = max(zmax - zmin, 1e-6)

    # depth-sorted draw list: bonds drawn at the average depth of their two atoms
    items = [("atom", i, rc[i, 2]) for i in range(n)]
    items += [("bond", (i, j), (rc[i, 2] + rc[j, 2]) / 2) for i, j in bonds]
    items.sort(key=lambda t: t[2])

    for kind, idx, z in items:
        shade = 0.35 + 0.55 * (z - zmin) / zspan  # farther = darker
        color = (0.12*shade+0.05, 0.35*shade+0.05, 0.75*shade+0.15)
        if kind == "atom":
            x, y, _ = rc[idx]
            rad = 0.34 + 0.10 * (z - zmin) / zspan
            ax.add_patch(Circle((x, y), rad, facecolor=color, edgecolor="black", linewidth=0.7, zorder=idx))
        else:
            i, j = idx
            ax.plot([rc[i,0], rc[j,0]], [rc[i,1], rc[j,1]], color=color, linewidth=2.2, solid_capstyle="round")

    lim = np.abs(rc[:, :2]).max() + 0.6
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal"); ax.set_axis_off()
    fig.tight_layout(pad=0.05)
    fig.savefig(os.path.join(OUT_DIR, r["name"] + ".png"), dpi=170, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    names.append((r["name"], r["point_group"]))

with open(os.path.join(OUT_DIR, "manifest.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["name","point_group"]); w.writerows(names)
print("rendered:", len(names))
