import csv, os

REPORT_DIR = "/home/gilles/polyN/orca_jobs/report"
os.makedirs(REPORT_DIR, exist_ok=True)

def esc(s):
    return str(s).replace("_", r"\_")

def charge_sup(fam):
    return {"cation": "$^{+}$", "anion": "$^{-}$", "neutral": ""}[fam]

# ---------------------------------------------------------------------------
# 1) Table la plus importante : provenance / statut vs litterature, 193 lignes
# ---------------------------------------------------------------------------
dhf = {r["name"]: r for r in csv.DictReader(open("/home/gilles/polyN/orca_jobs/results_dHf_xtb.csv"))}
topo = {r["name"]: r for r in csv.DictReader(open("/tmp/topology_match_report.csv"))}
summary = {r["name"]: r for r in csv.DictReader(open("/home/gilles/polyN/orca_jobs/results_summary.csv"))}

names_by_group = {}
for name, r in dhf.items():
    key = (int(r["n"]), r["family"])
    names_by_group.setdefault(key, []).append(name)

lines = []
lines.append(r"\begin{longtable}{@{}p{4.2cm}cp{1.3cm}rrp{1.2cm}p{2.6cm}@{}}")
lines.append(r"\scriptsize")
lines.append(r"\caption{Provenance de chaque candidat : origine (g\'en\'erateur local \texttt{polyN-pipeline} -- voir note en bas de tableau concernant \texttt{polyN\_adapt}), statut DFT, $\Delta H_f$ GFN2-xTB (kcal/mol, ancr\'e sur les valeurs litt\'erature CCSD(T) de N$_5^-$ et N$_5^+$), et correspondance \'eventuelle avec une topologie d\'ej\`a catalogu\'ee dans \texttt{polyN-pipeline/biblio\_polyN}.}")
lines.append(r"\label{tab:provenance}\\")
lines.append(r"\toprule")
lines.append(r"\textbf{Nom} & \textbf{G\'en.} & \textbf{Statut DFT} & \textbf{$\Delta H_f^{xtb}$} & \textbf{rang xtb} & \textbf{PG (DFT)} & \textbf{Vs litt\'erature} \\")
lines.append(r"\midrule")
lines.append(r"\endfirsthead")
lines.append(r"\multicolumn{7}{c}{\small (suite)}\\")
lines.append(r"\toprule")
lines.append(r"\textbf{Nom} & \textbf{G\'en.} & \textbf{Statut DFT} & \textbf{$\Delta H_f^{xtb}$} & \textbf{rang xtb} & \textbf{PG (DFT)} & \textbf{Vs litt\'erature} \\")
lines.append(r"\midrule")
lines.append(r"\endhead")
lines.append(r"\bottomrule")
lines.append(r"\endfoot")
lines.append(r"\bottomrule")
lines.append(r"\endlastfoot")

for (n, fam), names in sorted(names_by_group.items()):
    lines.append(r"\addlinespace")
    for name in sorted(names, key=lambda nm: dhf[nm]["rank"] if False else int(dhf[nm]["rank"])):
        d = dhf[name]
        s = summary.get(name)
        if s and s.get("electronic_Eh"):
            statut = "confirm\\'e" if s["n_imaginary_freq"] == "0" else f"\\textbf{{{s['n_imaginary_freq']} imag.}}"
            pg = s["point_group"] or ""
        else:
            statut = "en file"
            pg = "--"
        t = topo.get(name)
        if t is None:
            vslit = "pas de r\\'ef.\\ \\`a cette formule/charge"
        elif t["status"] == "matches_known":
            vslit = f"= {esc(t['matched_topology'])} ({t['matched_origin'].replace('_',' ')})"
        else:
            vslit = "topologie non catalogu\\'ee"
        lines.append(f"\\texttt{{{esc(name)}}} & polyN & {statut} & {d['dHf_kcalmol_xtb']} & {d['rank']} & {pg} & {vslit} \\\\")

lines.append(r"\end{longtable}")
lines.append(r"\vspace{-6pt}\noindent{\footnotesize Note~: les 193 candidats proviennent tous des g\'en\'erateurs de \texttt{polyN-pipeline} (\texttt{geng\_enumerate.py}, \texttt{cxhx\_to\_nx.py}, \texttt{random\_structure\_generator.py} selon la formule) d'apr\`es la convention de nommage locale -- aucune contribution du second d\'ep\^ot \texttt{polyN\_adapt} (approche surrogate/mutation) n'a \'et\'e identifi\'ee dans ce jeu ; \`a confirmer aupr\`es de l'auteur si un run \texttt{polyN\_adapt} a aliment\'e une partie de ces 193 fichiers.}")

with open(os.path.join(REPORT_DIR, "table_provenance.tex"), "w") as fh:
    fh.write("\n".join(lines))
print("table_provenance.tex:", len(dhf), "lignes")
