import csv, os, pickle
from collections import defaultdict

FIGS_DIR = "figs"
REPORT_DIR = "/home/gilles/polyN/orca_jobs/report"

state = pickle.load(open("/tmp/provenance_state.pkl", "rb"))
rel_dH, dft_based = state["rel_dH"], state["dft_based"]
fragmented, frag_rows = state["fragmented"], state["frag_rows"]
code_to_num, name_to_ref, topo = state["code_to_num"], state["name_to_ref"], state["topo"]

def esc(s):
    return str(s).replace("_", r"\_")

def reference_str(name):
    t = topo.get(name)
    if t and t["matched_origin"] == "biblio_article" and t["all_biblio_names"]:
        codes = []
        for nm in t["all_biblio_names"].split(";"):
            code = name_to_ref.get(nm)
            if code and code_to_num[code] not in codes:
                codes.append(code_to_num[code])
        if codes:
            nums = ",".join(f"[{c}]" for c in sorted(codes))
            return f"{nums}, our work"
    return "our work"

manifest = list(csv.DictReader(open(f"{REPORT_DIR}/figs/manifest.csv")))
summary = {r["name"]: r for r in csv.DictReader(open("/home/gilles/polyN/orca_jobs/results_summary.csv"))}

by_family = defaultdict(list)
for r in manifest:
    s = summary.get(r["name"])
    n = int(s["n_count"])
    fam = s["family"]
    by_family[fam].append((n, r["name"], r["point_group"]))

order = [("neutral", "S1", "neutres"), ("cation", "S2", "cationiques"), ("anion", "S3", "anioniques")]
out_lines = []
for fam, fig_label, label_fr in order:
    items = sorted(by_family.get(fam, []), key=lambda t: (t[0], rel_dH.get(t[1], 0)))
    out_lines.append(f"\\subsection*{{Figure {fig_label} -- compos\\'es {label_fr}}}")
    for i in range(0, len(items), 2):
        pair = items[i:i+2]
        out_lines.append(r"\begin{figure}[H]")
        out_lines.append(r"\centering")
        for n, name, pg in pair:
            dh = rel_dH.get(name, 0.0)
            lvl = "DFT" if dft_based.get(name) else "xtb"
            flag = (" -- \\textbf{fragment\\'ee (" + frag_rows[name]["fragment_sizes"].replace("+", "$+$") + ")}"
                    if name in fragmented else "")
            ref = reference_str(name)
            out_lines.append(r"\begin{minipage}{0.46\textwidth}")
            out_lines.append(r"\centering")
            out_lines.append(f"\\includegraphics[width=0.8\\linewidth]{{{FIGS_DIR}/{name}.png}}\\\\[2pt]")
            out_lines.append(f"{{\\small \\texttt{{{esc(name)}}} -- {pg} -- $\\Delta H={dh:.2f}$ ({lvl}) -- {ref}{flag}}}")
            out_lines.append(r"\end{minipage}\hfill")
        out_lines.append(r"\end{figure}")

with open(f"{REPORT_DIR}/annexe_structures.tex", "w") as fh:
    fh.write("\n".join(out_lines))
print("annexe_structures.tex rebuilt,", sum(len(v) for v in by_family.values()), "structures")
for fam in by_family:
    print(" ", fam, len(by_family[fam]))
