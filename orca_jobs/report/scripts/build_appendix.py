import csv, os

FIGS_DIR = "figs"  # relative to report/ dir where main.tex lives
manifest = list(csv.DictReader(open("/home/gilles/polyN/orca_jobs/report/figs/manifest.csv")))
manifest.sort(key=lambda r: r["name"])

def esc(s):
    return str(s).replace("_", r"\_")

lines = []
for i in range(0, len(manifest), 2):
    pair = manifest[i:i+2]
    lines.append(r"\begin{figure}[H]")
    lines.append(r"\centering")
    for r in pair:
        lines.append(r"\begin{minipage}{0.46\textwidth}")
        lines.append(r"\centering")
        lines.append(f"\\includegraphics[width=0.85\\linewidth]{{{FIGS_DIR}/{r['name']}.png}}\\\\[2pt]")
        lines.append(f"{{\\small \\texttt{{{esc(r['name'])}}} -- {r['point_group']}}}")
        lines.append(r"\end{minipage}\hfill")
    lines.append(r"\end{figure}")

with open("/home/gilles/polyN/orca_jobs/report/annexe_structures.tex", "w") as fh:
    fh.write("\n".join(lines))
print("annexe_structures.tex:", len(manifest), "structures,", (len(manifest)+1)//2, "figures")
