import csv
from collections import defaultdict

def esc(s):
    return str(s).replace("_", r"\_")

charges = list(csv.DictReader(open("/home/gilles/polyN/orca_jobs/results_charges.csv")))
by_name = defaultdict(list)
for c in charges:
    by_name[c["name"]].append(c)

summ = {r["name"]: r for r in csv.DictReader(open("/home/gilles/polyN/orca_jobs/results_summary.csv"))}
frag_rows_all = {r["name"]: r for r in csv.DictReader(
    open("/home/gilles/polyN/orca_jobs/results_fragmentation.csv"))}
FRAGMENTED = {n for n, r in frag_rows_all.items() if int(r["n_fragments"]) > 1}

# groupe-best (confirmed minimum, lowest energy, non-fragmente) per (n,family)
groups = defaultdict(list)
for r in summ.values():
    if r["electronic_Eh"] and r["name"] not in FRAGMENTED:
        groups[(r["n_count"], r["family"])].append(r)

lines = [r"{\scriptsize", r"\begin{longtable}{@{}p{4.6cm}rrrp{2.6cm}@{}}",
         r"\caption{Charges atomiques (Mulliken et L\"owdin) du meilleur minimum confirm\'e de chaque groupe (formule, charge) -- min/max sur les atomes N. Charges de Bader/QTAIM non natives \`a ORCA, report\'ees (\S\ref{sec:limites}).}",
         r"\label{tab:charges}\\",
         r"\toprule",
         r"\textbf{Nom} & \textbf{Mulliken min} & \textbf{Mulliken max} & \textbf{L\"owdin min/max} & \\",
         r"\midrule\endfirsthead",
         r"\multicolumn{4}{c}{\small (suite)}\\ \toprule\endhead",
         r"\bottomrule\endfoot", r"\bottomrule\endlastfoot"]
for (n,fam), recs in sorted(groups.items(), key=lambda kv: (int(kv[0][0]), kv[0][1])):
    true_min = [r for r in recs if r["n_imaginary_freq"]=="0"]
    pool = true_min or recs
    best = min(pool, key=lambda r: float(r["electronic_Eh"]))
    cs = by_name.get(best["name"], [])
    if not cs:
        continue
    mull = [float(c["mulliken_charge"]) for c in cs if c["mulliken_charge"]]
    loew = [float(c["loewdin_charge"]) for c in cs if c["loewdin_charge"]]
    if not mull:
        continue
    lines.append(f"\\texttt{{{esc(best['name'])}}} & {min(mull):.3f} & {max(mull):.3f} & {min(loew):.3f} / {max(loew):.3f} & \\\\")
lines.append(r"\end{longtable}")
lines.append(r"}")
with open("/home/gilles/polyN/orca_jobs/report/table_charges.tex", "w") as fh:
    fh.write("\n".join(lines))
print("table_charges.tex done")
