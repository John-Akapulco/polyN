import csv, json
from collections import defaultdict

REPORT_DIR = "/home/gilles/polyN/orca_jobs/report"

def esc(s):
    return str(s).replace("_", r"\_")

name_to_ref = json.load(open("/tmp/name_to_ref.json"))
topo = {r["name"]: r for r in csv.DictReader(open("/tmp/topology_match_report.csv"))}
dhf = {r["name"]: r for r in csv.DictReader(open("/home/gilles/polyN/orca_jobs/results_dHf_xtb.csv"))}
summary = {r["name"]: r for r in csv.DictReader(open("/home/gilles/polyN/orca_jobs/results_summary.csv"))}

# --- reference code -> sequential number, only for codes actually used ---
used_codes = []
for r in topo.values():
    if r["matched_origin"] == "biblio_article":
        code = name_to_ref.get(r["matched_name"])
        if code and code not in used_codes:
            used_codes.append(code)
code_to_num = {c: i+1 for i, c in enumerate(used_codes)}

FULL_REFS = {
    "GS": r"Glukhovtsev, Jiao, von Rag\'e Schleyer, \emph{Inorg. Chem.} \textbf{1996}, 35, 7124--7133.",
    "B": r"Fau, Mobita, Wilson, Perera, Bartlett, \emph{Quantum Theory Project report}, University of Florida.",
}
# A1-A20 filled from table_references.tex if ever used (none are, currently)
import re
for m in re.finditer(r"\\item\[\[([A-Z0-9]+)\]\]\s*(.+)", open("/tmp/table_references.tex").read()):
    FULL_REFS.setdefault(m.group(1), m.group(2).strip())

def reference_str(name):
    t = topo.get(name)
    if t and t["matched_origin"] == "biblio_article":
        code = name_to_ref.get(t["matched_name"])
        if code:
            return f"[{code_to_num[code]}]"
    return "our work"

# --- relative dH (kcal/mol) within each (n, family) group, best = 0 ---
groups = defaultdict(list)
for name, r in dhf.items():
    groups[(int(r["n"]), r["family"])].append(name)

rel_dH = {}
dft_based = {}
for key, names in groups.items():
    # prefer DFT electronic energy where available
    dft_vals = {n: float(summary[n]["electronic_Eh"]) for n in names
                if n in summary and summary[n].get("electronic_Eh")}
    if dft_vals:
        best_n = min(dft_vals, key=dft_vals.get)
        best_e = dft_vals[best_n]
        for n in names:
            if n in dft_vals:
                rel_dH[n] = (dft_vals[n] - best_e) * 627.5094740631
                dft_based[n] = True
    # xTB fallback (and for members without DFT yet) -- rank via E_react (additive
    # constant cancels within a fixed (n,family) group, so plain differences work)
    xtb_vals = {n: float(dhf[n]["e_react_kcalmol_xtb"]) for n in names}
    xtb_best = min(xtb_vals.values())
    for n in names:
        if n not in rel_dH:
            rel_dH[n] = xtb_vals[n] - xtb_best
            dft_based[n] = False

# --- bond > 1.7 A flag ---
long_bond = set()
for r in csv.DictReader(open("/home/gilles/polyN/orca_jobs/results_bonds.csv")):
    if float(r["distance_A"]) > 1.7:
        long_bond.add(r["name"])

def make_table(family, label, fname):
    names = sorted(groups.get, key=lambda x: 0) if False else None
    rows_by_n = defaultdict(list)
    for (n, fam), names in groups.items():
        if fam != family:
            continue
        for name in names:
            rows_by_n[n].append(name)
    lines = [r"\begin{longtable}{@{}p{4.7cm}ccrccp{1.2cm}@{}}", r"\scriptsize",
             r"\caption{Compos\'es \textbf{" + label + r"}, class\'es par nombre d'atomes croissant puis par stabilit\'e (isom\`ere le plus stable = $\Delta H=0$). "
             r"$\Delta H$ au niveau DFT quand disponible, sinon GFN2-xTB (indiqu\'e en colonne). ``$>1{,}7$''~: la structure contient au moins une distance N--N sup\'erieure \`a 1,7~\AA. R\'ef\'erence~: num\'ero d'article (voir Annexe~R\'ef\'erences) ou \emph{our work} si absente de la biblio actuelle.}",
             r"\label{tab:provenance-" + family + r"}\\",
             r"\toprule",
             r"\textbf{Nom} & \textbf{N} & \textbf{PG} & \textbf{$\Delta H$ (kcal/mol)} & \textbf{niveau} & \textbf{$>1{,}7$\AA} & \textbf{R\'ef.} \\",
             r"\midrule\endfirsthead",
             r"\multicolumn{7}{c}{\small (suite)}\\ \toprule\endhead",
             r"\bottomrule\endfoot", r"\bottomrule\endlastfoot"]
    for n in sorted(rows_by_n):
        lines.append(r"\addlinespace")
        names_sorted = sorted(rows_by_n[n], key=lambda nm: rel_dH[nm])
        for name in names_sorted:
            s = summary.get(name)
            pg = s["point_group"] if s and s.get("electronic_Eh") else "--"
            niveau = "DFT" if dft_based.get(name) else "xtb"
            flag = r"\checkmark" if name in long_bond else ""
            ref = reference_str(name)
            lines.append(f"\\texttt{{{esc(name)}}} & {n} & {pg} & {rel_dH[name]:.2f} & {niveau} & {flag} & {ref} \\\\")
    lines.append(r"\end{longtable}")
    with open(f"{REPORT_DIR}/{fname}", "w") as fh:
        fh.write("\n".join(lines))
    print(fname, "->", sum(len(v) for v in rows_by_n.values()), "lignes")

make_table("neutral", "neutres", "table_provenance_neutre.tex")
make_table("cation", "cationiques", "table_provenance_cation.tex")
make_table("anion", "anioniques", "table_provenance_anion.tex")

# --- Annexe references ---
lines = [r"\begin{description}[leftmargin=1.6cm,itemsep=3pt,style=nextline]"]
for code, num in sorted(code_to_num.items(), key=lambda kv: kv[1]):
    lines.append(f"\\item[[{num}]] {FULL_REFS.get(code, code)}")
lines.append(r"\end{description}")
with open(f"{REPORT_DIR}/annexe_references.tex", "w") as fh:
    fh.write("\n".join(lines))
print("references used:", code_to_num)

# expose rel_dH / dft_based / long_bond / reference_str-materials for the appendix script
import pickle
pickle.dump({"rel_dH": rel_dH, "dft_based": dft_based, "long_bond": long_bond,
             "code_to_num": code_to_num, "name_to_ref": name_to_ref, "topo": topo},
            open("/tmp/provenance_state.pkl", "wb"))
