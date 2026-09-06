"""Tableau de couverture du criblage pool 1 : pour chaque groupe (N,
charge) genere par polyN-pipeline, compte le total genere et sa
repartition apres DFT (confirme / fragmente / frequence imaginaire / non
converge). Sert a reperer les groupes absents ou a faible rendement, pour
orienter un complement futur (pool 1 substitution vs pool 2 surrogate)."""
import csv, glob, os, re
from collections import defaultdict

ROOT = "/home/gilles/polyN/orca_jobs"
REPORT_DIR = f"{ROOT}/report"
NAME_RE = re.compile(r"^N(?P<n>\d+)_(?P<family>neutral|cation|anion)_(?P=family)_(?P<rank>\d+)$")
FAM_SIGN = {"neutral": "0", "cation": "+", "anion": "-"}

summ = {r["name"]: r for r in csv.DictReader(open(f"{ROOT}/results_summary.csv"))}
frag = {r["name"] for r in csv.DictReader(open(f"{ROOT}/results_fragmentation.csv")) if int(r["n_fragments"]) > 1}
imag = {r["name"] for r in csv.DictReader(open(f"{ROOT}/results_imaginary_modes.csv"))}
nonconv = {r["name"] for r in csv.DictReader(open(f"{ROOT}/results_nonconverged.csv"))}

total_gen = defaultdict(int)
status = defaultdict(lambda: defaultdict(int))
for f in glob.glob(f"{ROOT}/xyz_gfn2xtb/*.xyz"):
    name = os.path.splitext(os.path.basename(f))[0]
    m = NAME_RE.match(name)
    if not m:
        continue
    key = (int(m.group("n")), m.group("family"))
    total_gen[key] += 1
    if name in nonconv:
        status[key]["non_conv"] += 1
    elif name in frag:
        status[key]["frag"] += 1
    elif name in imag:
        status[key]["imag"] += 1
    elif name in summ:
        status[key]["confirmed"] += 1

# groupes possibles sur toute la grille N4-N16 x {neutral,cation,anion},
# y compris ceux a 0 candidat -- c'est le point de cette analyse.
all_keys = [(n, fam) for n in range(4, 17) for fam in ("neutral", "cation", "anion")]

lines = [r"{\scriptsize", r"\begin{longtable}{@{}ccrrrrrr@{}}",
         r"\caption{Couverture du criblage pool~1 (\texttt{polyN-pipeline}, "
         r"g\'en\'eration combinatoire/substitution, \S\ref{sec:pools}) par "
         r"groupe (N, charge). \emph{Rendement} = confirm\'e / g\'en\'er\'e. "
         r"Groupes \`a 0 candidat surlign\'es par un tiret.}",
         r"\label{tab:coverage}\\",
         r"\toprule",
         r"\textbf{N} & \textbf{Charge} & \textbf{G\'en\'er\'e} & \textbf{Confirm\'e} & "
         r"\textbf{Rendement} & \textbf{Frag.} & \textbf{Imag.} & \textbf{Non conv.} \\",
         r"\midrule\endfirsthead",
         r"\multicolumn{8}{c}{\small (suite)}\\ \toprule\endhead",
         r"\bottomrule\endfoot", r"\bottomrule\endlastfoot"]

def singlet_compatible(n, fam):
    """Regle de core.geng_interface.validate_n_for_charge (pool 2) : un
    etat singulet (7n - charge pair) n'existe que si n est pair a charge
    paire (neutre) ou n impair a charge impaire (cation/anion). Les
    groupes incompatibles sont a 0 candidat POUR TOUJOURS avec un
    generateur singulet -- pool 1 comme pool 2 -- ce n'est pas un trou de
    couverture a combler mais une exclusion chimique (necessiterait un
    doublet, hors perimetre des deux generateurs actuels)."""
    return (n % 2 == 0) if fam == "neutral" else (n % 2 == 1)

missing_excluded, missing_reachable, low_yield = [], [], []
last_n = None
for n, fam in sorted(all_keys, key=lambda k: (k[0], {"neutral": 0, "cation": 1, "anion": 2}[k[1]])):
    key = (n, fam)
    g = total_gen.get(key, 0)
    if g == 0:
        continue  # ligne '0 candidat' omise du corps du tableau, listee a part ci-dessous
    if n != last_n:
        lines.append(r"\addlinespace"); last_n = n
    s = status[key]
    conf = s["confirmed"]
    yield_pct = 100 * conf / g if g else 0
    lines.append(f"{n} & {FAM_SIGN[fam]} & {g} & {conf} & {yield_pct:.0f}\\% & {s['frag']} & {s['imag']} & {s['non_conv']} \\\\")
    if yield_pct < 40:
        low_yield.append((n, fam, g, conf, yield_pct))
for n, fam in all_keys:
    if total_gen.get((n, fam), 0) > 0:
        continue
    if not singlet_compatible(n, fam):
        missing_excluded.append((n, fam))
    else:
        missing_reachable.append((n, fam))

lines.append(r"\end{longtable}")
lines.append(r"}")

with open(f"{REPORT_DIR}/table_coverage.tex", "w") as fh:
    fh.write("\n".join(lines))

def fmt_group(n, fam):
    return f"N$_{{{n}}}^{{{FAM_SIGN[fam].replace('0','')}}}$" if fam != "neutral" else f"N$_{{{n}}}$"

excluded_str = ", ".join(fmt_group(n, f) for n, f in sorted(missing_excluded))
reachable_str = ", ".join(fmt_group(n, f) for n, f in sorted(missing_reachable)) or "aucun"
low_yield_str = ", ".join(f"{fmt_group(n,f)} ({c}/{g}, {p:.0f}\\%)" for n, f, g, c, p in
                          sorted(low_yield, key=lambda x: x[4]))

with open(f"{REPORT_DIR}/coverage_summary.tex", "w") as fh:
    fh.write(
        r"\textbf{Hors p\'erim\`etre singulet} (0 candidat, et cela ne "
        r"changera pas avec un g\'en\'erateur singulet -- pool~1 ou "
        r"pool~2 : $7N-\mathrm{charge}$ impair, aucun \'etat singulet "
        r"n'existe \`a cette taille/charge, il faudrait un doublet)~: "
        + excluded_str + r".\\[4pt]"
        r"\textbf{Trou de couverture r\'eellement accessible} (0 "
        r"candidat mais compatible singulet -- cible l\'egitime pour un "
        r"compl\'ement pool~1 ou pool~2)~: " + reachable_str + r".\\[4pt]"
        r"\textbf{Rendement faible} ($<$40\%, "
        r"beaucoup de candidats g\'en\'er\'es mais peu confirm\'es -- "
        r"g\'eom\'etries de d\'epart chimiquement peu viables plut\^ot "
        r"qu'un manque de volume)~: " + low_yield_str + "."
    )

print(f"table_coverage.tex : {len(all_keys)-len(missing_excluded)-len(missing_reachable)} groupes non-vides, "
      f"{len(missing_excluded)} hors perimetre singulet, {len(missing_reachable)} trous accessibles, "
      f"{len(low_yield)} a rendement <40%")
