import csv, os, pickle

REPORT_DIR = "/home/gilles/polyN/orca_jobs/report"
FIGS_DIR = "figs"

def esc(s):
    return str(s).replace("_", r"\_")

state = pickle.load(open("/tmp/provenance_state.pkl", "rb"))
name_to_ref, topo, code_to_num = state["name_to_ref"], state["topo"], state["code_to_num"]

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

rows = list(csv.DictReader(open("/home/gilles/polyN/orca_jobs/results_nonconverged.csv")))
rows.sort(key=lambda r: (int(r["n"]), r["family"], r["name"]))

# --- Tableau S2 (Annexe) ---
n_frag_last = sum(1 for r in rows if int(r["n_fragments_last_cycle"]) > 1)
lines = [r"\noindent\textbf{Tableau S2.} Structures dont l'optimisation "
         r"g\'eom\'etrique DFT WB97X-D4/aug-cc-pVTZ n'a \textbf{pas "
         r"converg\'e} -- l'optimiseur a atteint sa limite de 50 cycles "
         r"(\texttt{The optimization did not converge but reached the "
         r"maximum}) sans trouver de point stationnaire. \'Energie non "
         r"fiable, structures exclues de tous les tableaux d'analyse "
         r"(\S\ref{sec:provenance}--\ref{sec:bonds}) et de la comparaison "
         r"de stabilit\'e (\S\ref{sec:litterature}) tant qu'elles n'ont "
         r"pas \'et\'e reprises. \textbf{" + str(n_frag_last) + r"~des " +
         str(len(rows)) + r"~n'ont en r\'ealit\'e probablement jamais de "
         r"minimum li\'e} : la colonne \emph{Frag.\ dernier cycle} montre "
         r"qu'elles sont d\'ej\`a s\'epar\'ees en plusieurs composantes "
         r"connexes (seuil 1,70~\AA) \`a la derni\`ere g\'eom\'etrie "
         r"tent\'ee -- l'optimiseur suit un gradient de dissociation qui "
         r"ne s'annule jamais en un nombre fini de cycles, plut\^ot qu'un "
         r"probl\`eme num\'erique de convergence~; les relancer avec "
         r"\texttt{\%geom MaxIter} plus grand ne changera probablement "
         r"rien. Seules les " + str(len(rows) - n_frag_last) + r" restantes "
         r"(compos\'e li\'e unique au dernier cycle) sont des candidates "
         r"cr\'edibles \`a une reprise avec davantage de cycles. "
         r"Repr\'esentations avant/dernier cycle en Figure~S2.\par\vspace{4pt}",
         r"{\scriptsize", r"\begin{longtable}{@{}p{3.8cm}ccrp{1.6cm}c@{}}",
         r"\label{tab:nonconverged}\\",
         r"\toprule",
         r"\textbf{Nom} & \textbf{N} & \textbf{Charge} & \textbf{Cycles} & \textbf{Frag.\ dernier cycle} & \textbf{R\'ef.} \\",
         r"\midrule\endfirsthead",
         r"\multicolumn{6}{c}{\small (suite)}\\ \toprule\endhead",
         r"\bottomrule\endfoot", r"\bottomrule\endlastfoot"]
for r in rows:
    ref = reference_str(r["name"])
    n_frag = int(r["n_fragments_last_cycle"])
    frag_mark = (r"\checkmark~(" + r["fragment_sizes_last_cycle"].replace("+", "$+$") + ")") if n_frag > 1 else "non"
    lines.append(f"\\texttt{{{esc(r['name'])}}} & {r['n']} & {r['charge']} & {r['n_cycles']} & {frag_mark} & {ref} \\\\")
lines.append(r"\end{longtable}")
lines.append(r"}")
with open(f"{REPORT_DIR}/table_S2_nonconverged.tex", "w") as fh:
    fh.write("\n".join(lines))
print("table_S2_nonconverged.tex:", len(rows), "lignes")

# --- Figure S2 (Annexe) : avant (xTB) / dernier cycle tente (DFT) ---
manifest = list(csv.DictReader(open(f"{REPORT_DIR}/{FIGS_DIR}/nonconverged/manifest.csv")))
by_name = {r["name"]: r for r in rows}
items = sorted(manifest, key=lambda r: (int(by_name[r["name"]]["n"]), by_name[r["name"]]["family"]))

s2_lines = [r"\subsection*{Figure S2 -- structures non convergées (avant / dernier cycle)}",
            r"Pour chaque candidat non converg\'e (Tableau~S2)~: \`a gauche "
            r"la g\'eom\'etrie initiale (criblage GFN2-xTB), \`a droite la "
            r"derni\`ere g\'eom\'etrie tent\'ee par l'optimiseur DFT avant "
            r"d'atteindre la limite de cycles -- pas un minimum, juste le "
            r"dernier point de la trajectoire d'optimisation."]
for r in items:
    name = r["name"]
    d = by_name[name]
    sign = {"cation": "+", "anion": "-", "neutral": "0"}[d["family"]]
    s2_lines.append(r"\begin{figure}[H]")
    s2_lines.append(r"\centering")
    s2_lines.append(r"\begin{minipage}{0.46\textwidth}")
    s2_lines.append(r"\centering")
    s2_lines.append(f"\\includegraphics[width=0.8\\linewidth]{{{FIGS_DIR}/nonconverged/{name}_initial.png}}\\\\[2pt]")
    s2_lines.append(r"{\small initiale (xTB, N$_" + d["n"] + "^{" + sign + r"}$)}")
    s2_lines.append(r"\end{minipage}\hfill")
    s2_lines.append(r"\begin{minipage}{0.46\textwidth}")
    s2_lines.append(r"\centering")
    s2_lines.append(f"\\includegraphics[width=0.8\\linewidth]{{{FIGS_DIR}/nonconverged/{name}_last.png}}\\\\[2pt]")
    s2_lines.append(r"{\small dernier cycle DFT (" + r["n_cycles"] + " cycles)}")
    s2_lines.append(r"\end{minipage}")
    s2_lines.append(r"\end{figure}")
    n_frag = int(d["n_fragments_last_cycle"])
    status = (r"en cours de fragmentation, " + d["fragment_sizes_last_cycle"].replace("+", "$+$")) if n_frag > 1 \
        else r"cluster unique -- non-convergence num\'erique"
    s2_lines.append(r"\centerline{\small \texttt{" + esc(name) + r"} -- " + status + "}")
    s2_lines.append(r"\vspace{8pt}")

s2_lines.append(r"\bigskip\noindent\textit{L\'egende g\'en\'erale, Figure~S2~:} "
                 r"sph\`eres bleues reli\'ees par des b\^atonnets = atomes N et "
                 r"liaisons N--N (mod\`ele boules-b\^atonnets, rendu depuis les "
                 r"coordonn\'ees cart\'esiennes .xyz) ; paires group\'ees par "
                 r"candidat, g\'eom\'etrie initiale GFN2-xTB \`a gauche et "
                 r"derni\`ere g\'eom\'etrie de la trajectoire d'optimisation "
                 r"DFT (non converg\'ee) \`a droite, pour les "
                 + str(len(items)) + r" candidats non converg\'es "
                 r"(Tableau~S2).")

with open(f"{REPORT_DIR}/annexe_nonconverged.tex", "w") as fh:
    fh.write("\n".join(s2_lines))
print("annexe_nonconverged.tex (Figure S2):", len(items), "structures")
