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
        for nm in r["all_biblio_names"].split(";") if r["all_biblio_names"] else []:
            code = name_to_ref.get(nm)
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
    """Every row is one of our own 193 candidates, so 'our work' always
    applies; published-article match(es) are cited alongside it, not
    instead -- a topology independently reported in more than one article
    cites all of them."""
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

# --- fragmentation (>1.7 A splits the structure into >1 connected
# component, i.e. it's really two-or-more separate molecular species, not
# one bound cluster -- e.g. an "N5-" that's actually N2 + N3-) ---
frag_rows = {r["name"]: r for r in csv.DictReader(
    open("/home/gilles/polyN/orca_jobs/results_fragmentation.csv"))}
fragmented = {n for n, r in frag_rows.items() if int(r["n_fragments"]) > 1}

# --- relative dH (kcal/mol) within each (n, family) group, best = 0 ---
# Only a genuine confirmed minimum (not fragmented, no imaginary frequency)
# can anchor a group's zero-point; every other member still gets a dH
# computed and printed, just never picked as the anchor.
imaginary_names = {r["name"] for r in csv.DictReader(
    open("/home/gilles/polyN/orca_jobs/results_imaginary_modes.csv"))}

groups = defaultdict(list)
for name, r in dhf.items():
    groups[(int(r["n"]), r["family"])].append(name)

rel_dH = {}
dft_based = {}
for key, names in groups.items():
    intact = [n for n in names if n not in fragmented and n not in imaginary_names]
    # prefer DFT electronic energy where available
    dft_vals = {n: float(summary[n]["electronic_Eh"]) for n in intact
                if n in summary and summary[n].get("electronic_Eh")}
    if dft_vals:
        best_n = min(dft_vals, key=dft_vals.get)
        best_e = dft_vals[best_n]
        for n in names:
            if n in summary and summary[n].get("electronic_Eh"):
                rel_dH[n] = (float(summary[n]["electronic_Eh"]) - best_e) * 627.5094740631
                dft_based[n] = True
    # xTB fallback (and for members without DFT yet) -- rank via E_react (additive
    # constant cancels within a fixed (n,family) group, so plain differences work)
    xtb_pool = intact or names  # if every member is fragmented, fall back to all
    xtb_best = min(float(dhf[n]["e_react_kcalmol_xtb"]) for n in xtb_pool)
    for n in names:
        if n not in rel_dH:
            rel_dH[n] = float(dhf[n]["e_react_kcalmol_xtb"]) - xtb_best
            dft_based[n] = False

def make_table(family, label, fname):
    rows_by_n = defaultdict(list)
    for (n, fam), names in groups.items():
        if fam != family:
            continue
        for name in names:
            if name in fragmented:
                continue  # moved to Table S1 (Annexe) instead
            rows_by_n[n].append(name)
    lines = [r"{\scriptsize", r"\begin{longtable}{@{}p{4.7cm}ccrcp{1.6cm}@{}}",
             r"\caption{Compos\'es \textbf{" + label + r"}, class\'es par nombre d'atomes croissant puis par stabilit\'e (isom\`ere le plus stable = $\Delta H=0$). "
             r"$\Delta H$ au niveau DFT quand disponible, sinon GFN2-xTB (indiqu\'e en colonne). N'inclut que les clusters li\'es (Tableau~S1~: candidats fragment\'es en plusieurs esp\`eces mol\'eculaires distinctes). R\'ef\'erence(s)~: num\'ero(s) d'article (Annexe~R\'ef\'erences) ou \emph{our work} si absente de la biblio actuelle.}",
             r"\label{tab:provenance-" + family + r"}\\",
             r"\toprule",
             r"\textbf{Nom} & \textbf{N} & \textbf{PG} & \textbf{$\Delta H$ (kcal/mol)} & \textbf{niveau} & \textbf{R\'ef.} \\",
             r"\midrule\endfirsthead",
             r"\multicolumn{6}{c}{\small (suite)}\\ \toprule\endhead",
             r"\bottomrule\endfoot", r"\bottomrule\endlastfoot"]
    for n in sorted(rows_by_n):
        lines.append(r"\addlinespace")
        names_sorted = sorted(rows_by_n[n], key=lambda nm: rel_dH[nm])
        for name in names_sorted:
            s = summary.get(name)
            pg = s["point_group"] if s and s.get("electronic_Eh") else "--"
            niveau = "DFT" if dft_based.get(name) else "xtb"
            ref = reference_str(name)
            lines.append(f"\\texttt{{{esc(name)}}} & {n} & {pg} & {rel_dH[name]:.2f} & {niveau} & {ref} \\\\")
    lines.append(r"\end{longtable}")
    lines.append(r"}")
    with open(f"{REPORT_DIR}/{fname}", "w") as fh:
        fh.write("\n".join(lines))
    print(fname, "->", sum(len(v) for v in rows_by_n.values()), "lignes (fragmentees exclues)")

make_table("neutral", "neutres", "table_provenance_neutre.tex")
make_table("cation", "cationiques", "table_provenance_cation.tex")
make_table("anion", "anioniques", "table_provenance_anion.tex")

# --- Table S1 (Annexe): structures fragmentees ---
# Manual "Table S1." heading instead of \caption's auto-numbered "Table N":
# keeps Supporting-Information numbering (S1, S2...) independent of the main
# body's Table 1,2,3... sequence, with no extra package/counter machinery.
lines = [r"\noindent\textbf{Table S1.} Structures fragment\'ees~: le candidat se s\'epare en plusieurs esp\`eces mol\'eculaires distinctes (aucune paire N--N restante sous 1,7~\AA\ entre les morceaux), pas un cluster N$_x$ li\'e unique. "
         r"Retir\'ees des tableaux principaux (\S\ref{sec:provenance}) et de la comparaison de stabilit\'e. Repr\'esentations avant/apr\`es en Figure~S4.\par\vspace{4pt}",
         r"\begin{longtable}{@{}p{4.4cm}ccp{1.6cm}cc@{}}", r"\scriptsize",
         r"\label{tab:fragmented}\\",
         r"\toprule",
         r"\textbf{Nom} & \textbf{N} & \textbf{Charge} & \textbf{Fragments (tailles)} & \textbf{Niveau} & \textbf{R\'ef.} \\",
         r"\midrule\endfirsthead",
         r"\multicolumn{6}{c}{\small (suite)}\\ \toprule\endhead",
         r"\bottomrule\endfoot", r"\bottomrule\endlastfoot"]
frag_sorted = sorted(fragmented, key=lambda nm: (int(dhf[nm]["n"]), dhf[nm]["family"]))
for name in frag_sorted:
    r = dhf[name]
    fr = frag_rows[name]
    sign = {"cation": "+", "anion": "-", "neutral": "0"}[r["family"]]
    sizes = fr["fragment_sizes"].replace("+", "$+$")
    ref = reference_str(name)
    lines.append(f"\\texttt{{{esc(name)}}} & {r['n']} & {sign} & {sizes} & {fr['level']} & {ref} \\\\")
lines.append(r"\end{longtable}")
with open(f"{REPORT_DIR}/table_S1_fragmented.tex", "w") as fh:
    fh.write("\n".join(lines))
print("table_S1_fragmented.tex:", len(frag_sorted), "lignes")

# --- Annexe references ---
lines = [r"\begin{description}[leftmargin=1.6cm,itemsep=3pt,style=nextline]"]
for code, num in sorted(code_to_num.items(), key=lambda kv: kv[1]):
    lines.append(f"\\item[[{num}]] {FULL_REFS.get(code, code)}")
lines.append(r"\end{description}")
with open(f"{REPORT_DIR}/annexe_references.tex", "w") as fh:
    fh.write("\n".join(lines))
print("references used:", code_to_num)

# expose rel_dH / dft_based / fragmented / reference_str-materials for the appendix script
import pickle
pickle.dump({"rel_dH": rel_dH, "dft_based": dft_based, "fragmented": fragmented,
             "frag_rows": frag_rows, "dhf": dhf,
             "code_to_num": code_to_num, "name_to_ref": name_to_ref, "topo": topo},
            open("/tmp/provenance_state.pkl", "wb"))
