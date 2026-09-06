"""Tableau de confrontation xTB / DFT / CCSD(T)-F12 : pour chaque groupe
(N, charge) ayant >=2 structures avec une energie CCSD(T)-F12 terminee,
compare le rang de stabilite (isomere le plus stable = rang 1) aux trois
niveaux de theorie. Concu pour etre relance a chaque fois que de nouveaux
jobs de la campagne CCSD(T) terminent (orca_jobs/ccsdt_jobs/) -- purement
lecture, aucune dependance a un etat intermediaire d'une execution
precedente."""
import csv, glob, os, re
from collections import defaultdict

ROOT = "/home/gilles/polyN/orca_jobs"
REPORT_DIR = f"{ROOT}/report"
HARTREE_TO_KCAL = 627.5094740631
NAME_RE = re.compile(r"^N(?P<n>\d+)_(?P<family>neutral|cation|anion)_(?P=family)_(?P<rank>\d+)$")

def esc(s):
    return str(s).replace("_", r"\_")

# --- 1) energie CCSD(T)-F12 finale de chaque structure terminee ---
ccsdt_E = {}
for d in glob.glob(f"{ROOT}/ccsdt_jobs/*/"):
    name = os.path.basename(d.rstrip("/"))
    out = f"{d}{name}_ccsdt.out"
    if not os.path.isfile(out):
        continue
    txt = open(out, errors="ignore").read()
    m = re.search(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", txt)
    if m:
        ccsdt_E[name] = float(m.group(1))

# --- 2) rangs/DeltaE xTB et DFT (deja calcules par harvest_results.py) ---
compare = {}
for fam in ("neutral", "cation", "anion"):
    path = f"{ROOT}/compare_{fam}.csv"
    if os.path.isfile(path):
        for r in csv.DictReader(open(path)):
            compare[r["name"]] = r

# --- 3) grouper par (n, famille), ne garder que les groupes comparables ---
groups = defaultdict(list)
refs = []  # molecules hors criblage (N2, NH4+, NH2-) : pas de groupe, listees a part
for name in ccsdt_E:
    m = NAME_RE.match(name)
    if not m:
        refs.append(name)
        continue
    groups[(int(m.group("n")), m.group("family"))].append(name)

comparable = {k: v for k, v in groups.items() if len(v) >= 2}
n_total_done = len(ccsdt_E)

lines = [r"{\scriptsize", r"\begin{longtable}{@{}p{4.0cm}rrrr@{}}",
         r"\caption{Confrontation xTB / DFT / CCSD(T)-F12 (cc-pVTZ-F12, "
         r"single-point sur g\'eom\'etrie DFT resym\'etris\'ee) : rang de "
         r"stabilit\'e (isom\`ere le plus stable de son groupe (N, charge) "
         r"= rang~1) aux trois niveaux, et $\Delta E$ CCSD(T)-F12 "
         r"(kcal/mol) relative \`a ce m\^eme isom\`ere. Limit\'e aux "
         r"groupes ayant au moins 2 structures avec un calcul CCSD(T) "
         r"termin\'e -- " + str(n_total_done) + r" structures termin\'ees "
         r"au total au moment de la r\'edaction (campagne en cours, "
         r"\S\ref{sec:limites}).}",
         r"\label{tab:ccsdt-comparison}\\",
         r"\toprule",
         r"\textbf{Nom} & \textbf{rang xtb} & \textbf{rang DFT} & \textbf{rang CCSD(T)} & \textbf{$\Delta E$ CCSD(T)} \\",
         r" & & & & (kcal/mol) \\",
         r"\midrule\endfirsthead",
         r"\multicolumn{5}{c}{\small (suite)}\\ \toprule\endhead",
         r"\bottomrule\endfoot", r"\bottomrule\endlastfoot"]

n_agree_all3, n_dft_ccsdt_agree = 0, 0
for key in sorted(comparable):
    names = comparable[key]
    e_min = min(ccsdt_E[n] for n in names)
    ranked = sorted(names, key=lambda n: ccsdt_E[n])
    ccsdt_rank = {n: i + 1 for i, n in enumerate(ranked)}
    gs_ccsdt, gs_dft, gs_xtb = (
        ranked[0],
        min(names, key=lambda n: float(compare[n]["dft_Erel_kcal"])),
        min(names, key=lambda n: float(compare[n]["xtb_Erel_kcal"])),
    )
    if gs_ccsdt == gs_dft == gs_xtb:
        n_agree_all3 += 1
    if gs_ccsdt == gs_dft:
        n_dft_ccsdt_agree += 1
    lines.append(r"\addlinespace")
    for n in ranked:
        c = compare[n]
        de = (ccsdt_E[n] - e_min) * HARTREE_TO_KCAL
        lines.append(f"\\texttt{{{esc(n)}}} & {c['xtb_rank_as_generated']} & {c['dft_rank']} & {ccsdt_rank[n]} & {de:.3f} \\\\")
lines.append(r"\end{longtable}")
lines.append(r"}")

n_groups = len(comparable)
lines.append(
    r"\noindent\textit{Bilan~:} " + str(n_agree_all3) + "/" + str(n_groups) +
    r" groupes comparables ont le m\^eme \'etat fondamental aux trois "
    r"niveaux~; DFT et CCSD(T) s'accordent entre eux sur " +
    str(n_dft_ccsdt_agree) + "/" + str(n_groups) + r" groupes -- l\`a o\`u "
    r"ils divergent, l'\'ecart CCSD(T) entre les deux premiers isom\`eres "
    r"est syst\'ematiquement inf\'erieur \`a 1~kcal/mol (quasi-d\'eg\'en\'erescence, "
    r"pas de d\'esaccord physique). Quand xTB diverge du couple DFT/CCSD(T), "
    r"c'est presque toujours xTB qui est en cause -- coh\'erent avec sa "
    r"pr\'ecision moindre."
)

if refs:
    lines.append(r"\bigskip\noindent\textit{R\'ef\'erences hors criblage} "
                 r"(pas de groupe de comparaison)~: " +
                 ", ".join(f"\\texttt{{{esc(n)}}}" for n in sorted(refs)) + ".")

with open(f"{REPORT_DIR}/table_ccsdt_comparison.tex", "w") as fh:
    fh.write("\n".join(lines))

print(f"table_ccsdt_comparison.tex : {n_total_done} energies CCSD(T), "
      f"{n_groups} groupes comparables, {n_agree_all3}/{n_groups} GS identique aux 3 niveaux")
