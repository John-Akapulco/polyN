import csv, os
from collections import defaultdict

REPORT_DIR = "/home/gilles/polyN/orca_jobs/report"

def esc(s):
    return str(s).replace("_", r"\_")

frag_rows_all = {r["name"]: r for r in csv.DictReader(
    open("/home/gilles/polyN/orca_jobs/results_fragmentation.csv"))}
FRAGMENTED = {n for n, r in frag_rows_all.items() if int(r["n_fragments"]) > 1}

# ---------------------------------------------------------------------------
# 2) Tableaux de classement DFT vs xTB, un par famille de charge
# ---------------------------------------------------------------------------
for fam, label in [("neutral","neutres"), ("cation","cationiques"), ("anion","anioniques")]:
    rows = list(csv.DictReader(open(f"/home/gilles/polyN/orca_jobs/compare_{fam}.csv")))
    lines = []
    lines.append(r"\begin{longtable}{@{}p{4.4cm}rrrrrc@{}}")
    lines.append(r"\scriptsize")
    lines.append(r"\caption{Classement DFT vs xTB, compos\'es \textbf{" + label + r"}. Tri\'e par rang DFT (ground state DFT = rang 1) au sein de chaque formule ; le rang xTB tel que g\'en\'er\'e \`a l'origine est rappel\'e pour voir si les deux classements concordent.}")
    lines.append(r"\label{tab:ranking-" + fam + r"}\\")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Nom} & \textbf{N} & \textbf{rang xtb} & \textbf{rang DFT} & \textbf{$E_{rel}$ xtb} & \textbf{$E_{rel}$ DFT} & \textbf{accord GS} \\")
    lines.append(r" & & & & (kcal/mol) & (kcal/mol) & \\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\multicolumn{7}{c}{\small (suite)}\\ \toprule")
    lines.append(r"\textbf{Nom} & \textbf{N} & \textbf{rang xtb} & \textbf{rang DFT} & \textbf{$E_{rel}$ xtb} & \textbf{$E_{rel}$ DFT} & \textbf{accord GS} \\")
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\bottomrule\endfoot")
    lines.append(r"\bottomrule\endlastfoot")
    last_n = None
    for r in rows:
        if r["n_count"] != last_n:
            lines.append(r"\addlinespace")
            last_n = r["n_count"]
        agree = "\\checkmark" if r["xtb_and_dft_agree_on_gs"] == "True" else ""
        lines.append(f"\\texttt{{{esc(r['name'])}}} & {r['n_count']} & {r['xtb_rank_as_generated']} & {r['dft_rank']} & {r['xtb_Erel_kcal']} & {r['dft_Erel_kcal']} & {agree} \\\\")
    lines.append(r"\end{longtable}")
    with open(os.path.join(REPORT_DIR, f"table_ranking_{fam}.tex"), "w") as fh:
        fh.write("\n".join(lines))
    print(f"table_ranking_{fam}.tex: {len(rows)} lignes")

# ---------------------------------------------------------------------------
# 3) HOMO/LUMO
# ---------------------------------------------------------------------------
summ = list(csv.DictReader(open("/home/gilles/polyN/orca_jobs/results_summary.csv")))
lines = [r"\begin{longtable}{@{}p{4.6cm}p{1.6cm}rrr@{}}", r"\scriptsize",
         r"\caption{Orbitales frontières (WB97X-D4/aug-cc-pVTZ) des candidats termin\'es.}",
         r"\label{tab:homolumo}\\",
         r"\toprule",
         r"\textbf{Nom} & \textbf{PG} & \textbf{HOMO (eV)} & \textbf{LUMO (eV)} & \textbf{Gap (eV)} \\",
         r"\midrule\endfirsthead",
         r"\multicolumn{5}{c}{\small (suite)}\\ \toprule",
         r"\textbf{Nom} & \textbf{PG} & \textbf{HOMO (eV)} & \textbf{LUMO (eV)} & \textbf{Gap (eV)} \\",
         r"\midrule\endhead", r"\bottomrule\endfoot", r"\bottomrule\endlastfoot"]
seen_n = None
for r in summ:
    if not r.get("homo_eV") or r["name"] in FRAGMENTED:
        continue
    if r["n_count"] != seen_n:
        lines.append(r"\addlinespace"); seen_n = r["n_count"]
    homo, lumo = float(r["homo_eV"]), float(r["lumo_eV"])
    lines.append(f"\\texttt{{{esc(r['name'])}}} & {r['point_group']} & {homo:.4f} & {lumo:.4f} & {lumo-homo:.4f} \\\\")
lines.append(r"\end{longtable}")
with open(os.path.join(REPORT_DIR, "table_homolumo.tex"), "w") as fh:
    fh.write("\n".join(lines))
print("table_homolumo.tex")

# ---------------------------------------------------------------------------
# 4) Distances / ordres de liaison N-N : corrélation avec dHf pour les neutres
# ---------------------------------------------------------------------------
bonds = list(csv.DictReader(open("/home/gilles/polyN/orca_jobs/results_bonds.csv")))
by_name = defaultdict(list)
for b in bonds:
    by_name[b["name"]].append(b)
dhf = {r["name"]: r for r in csv.DictReader(open("/home/gilles/polyN/orca_jobs/results_dHf_xtb.csv"))}

lines = [r"\begin{longtable}{@{}p{4.6cm}rrrp{2.4cm}r@{}}", r"\scriptsize",
         r"\caption{Compos\'es \textbf{neutres} termin\'es : proportion de liaisons N--N courtes (triple/double/d\'elocalis\'ees) parmi les liaisons r\'eelles (hors contacts non-liants), et $\Delta H_f$ GFN2-xTB. Seuils de classification en \S\ref{sec:limites} de \texttt{harvest\_results.py}.}",
         r"\label{tab:bonds-dhf}\\",
         r"\toprule",
         r"\textbf{Nom} & \textbf{liaisons} & \textbf{BO Mayer moy.} & \textbf{\% courtes\footnotemark[1]} & \textbf{classes pr\'esentes} & \textbf{$\Delta H_f^{xtb}$} \\",
         r"\midrule\endfirsthead",
         r"\multicolumn{6}{c}{\small (suite)}\\ \toprule",
         r"\textbf{Nom} & \textbf{liaisons} & \textbf{BO Mayer moy.} & \textbf{\% courtes} & \textbf{classes pr\'esentes} & \textbf{$\Delta H_f^{xtb}$} \\",
         r"\midrule\endhead", r"\bottomrule\endfoot", r"\bottomrule\endlastfoot"]
neutral_names = sorted([n for n in dhf if dhf[n]["family"] == "neutral" and n in by_name
                         and n not in FRAGMENTED],
                        key=lambda n: (dhf[n]["n"], int(dhf[n]["rank"])))
for name in neutral_names:
    bl = [b for b in by_name[name] if b["distance_class"] != "non-bonded"]
    if not bl:
        continue
    bos = [float(b["mayer_bond_order"]) for b in bl]
    short = sum(1 for b in bl if b["distance_class"] in ("triple","double","delocalized_1.5"))
    classes = sorted(set(b["distance_class"] for b in bl))
    lines.append(f"\\texttt{{{esc(name)}}} & {len(bl)} & {sum(bos)/len(bos):.3f} & {100*short/len(bl):.0f}\\% & {', '.join(c.replace('_',' ') for c in classes)} & {dhf[name]['dHf_kcalmol_xtb']} \\\\")
lines.append(r"\end{longtable}")
lines.append(r"\footnotetext[1]{Triple, double ou d\'elocalis\'ee (ordre ${\sim}1{,}5$), \`a l'exclusion des liaisons simples et des contacts non-liants.}")
with open(os.path.join(REPORT_DIR, "table_bonds_dhf.tex"), "w") as fh:
    fh.write("\n".join(lines))
print("table_bonds_dhf.tex:", len(neutral_names), "structures neutres")

# ---------------------------------------------------------------------------
# 5) Table S2 (Annexe) : structures avec au moins une frequence imaginaire
# ---------------------------------------------------------------------------
import pickle
prov_state = pickle.load(open("/tmp/provenance_state.pkl", "rb"))
rel_dH_all, dft_based_all = prov_state["rel_dH"], prov_state["dft_based"]

imag_rows = list(csv.DictReader(open("/home/gilles/polyN/orca_jobs/results_imaginary_modes.csv")))
by_imag = defaultdict(list)
for r in imag_rows:
    by_imag[r["name"]].append(r)

summ_by_name = {r["name"]: r for r in summ}
lines = [r"\noindent\textbf{Table S2.} Structures avec au moins une fr\'equence imaginaire (pas encore de vrai minimum). "
         r"``Max.\ imaginaire''~: la fr\'equence de plus grande amplitude parmi les modes imaginaires "
         r"(mode dominant de la coordonn\'ee de r\'eaction vers le vrai minimum). $\Delta H$~: \'energie "
         r"relative au ground-state DFT de la m\^eme composition (kcal/mol, niveau DFT si disponible "
         r"sinon xTB) -- \`a rapprocher de la fr\'equence pour rep\'erer une \'eventuelle corr\'elation "
         r"entre instabilit\'e vibrationnelle et haute \'energie relative. \'A reprendre manuellement "
         r"(distorsion le long de ce mode puis r\'eoptimisation) avant validation finale.\par\vspace{4pt}",
         r"\begin{longtable}{@{}p{4.0cm}ccrrp{1.7cm}c@{}}", r"\scriptsize",
         r"\label{tab:imaginary}\\",
         r"\toprule",
         r"\textbf{Nom} & \textbf{N} & \textbf{n modes} & \textbf{Max.\ imag.\ (cm$^{-1}$)} & \textbf{Intensit\'e (km/mol)} & \textbf{$\Delta H$ (kcal/mol)} & \textbf{Frag.} \\",
         r"\midrule\endfirsthead",
         r"\multicolumn{7}{c}{\small (suite)}\\ \toprule\endhead",
         r"\bottomrule\endfoot", r"\bottomrule\endlastfoot"]
names_sorted = sorted(by_imag, key=lambda n: (int(summ_by_name[n]["n_count"]) if n in summ_by_name else 0, n))
for name in names_sorted:
    modes = by_imag[name]
    worst = min(modes, key=lambda m: float(m["freq_cm1"]))  # most negative = dominant
    n_count = summ_by_name.get(name, {}).get("n_count", "?")
    intensity = worst["intensity_km_mol"] if worst["intensity_km_mol"] else "--"
    frag_mark = r"\checkmark" if name in FRAGMENTED else ""
    if name in rel_dH_all:
        lvl = "DFT" if dft_based_all.get(name) else "xtb"
        dh = f"{rel_dH_all[name]:.2f} ({lvl})"
    else:
        dh = "--"
    lines.append(f"\\texttt{{{esc(name)}}} & {n_count} & {len(modes)} & {float(worst['freq_cm1']):.1f} & {intensity} & {dh} & {frag_mark} \\\\")
lines.append(r"\end{longtable}")
with open(os.path.join(REPORT_DIR, "table_S2_imaginary.tex"), "w") as fh:
    fh.write("\n".join(lines))
print("table_S2_imaginary.tex:", len(names_sorted), "structures")
