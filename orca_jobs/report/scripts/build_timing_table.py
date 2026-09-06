import re, glob, os, csv

def parse_elapsed_to_seconds(s):
    parts = [float(p) for p in s.strip().split(':')]
    if len(parts) == 3: h, m, sec = parts
    elif len(parts) == 2: h, m, sec = 0, parts[0], parts[1]
    else: h, m, sec = 0, 0, parts[0]
    return h*3600 + m*60 + sec

rows = []
for d in sorted(glob.glob('/home/gilles/polyN/orca_jobs/jobs/*/')):
    d = d.rstrip('/')
    name = os.path.basename(d)
    out, timef = f'{d}/{name}.out', f'{d}/{name}.time'
    if not (os.path.exists(out) and os.path.exists(timef)):
        continue
    if 'TOTAL RUN TIME' not in open(out).read():
        continue
    tt = open(timef).read()
    em = re.search(r'Elapsed \(wall clock\).*?: ([\d:.]+)', tt)
    um = re.search(r'User time \(seconds\): ([\d.]+)', tt)
    sm = re.search(r'System time \(seconds\): ([\d.]+)', tt)
    if not (em and um and sm): continue
    inp = f'{d}/{name}_s1.inp' if os.path.exists(f'{d}/{name}_s1.inp') else f'{d}/{name}.inp'
    ncm = re.search(r'nprocs (\d+)', open(inp).read()) if os.path.exists(inp) else None
    nc = int(ncm.group(1)) if ncm else None
    rows.append((name, nc, parse_elapsed_to_seconds(em.group(1)), float(um.group(1))+float(sm.group(1))))

small = [r for r in rows if r[1] == 8]
big = [r for r in rows if r[1] and r[1] > 8]

def stats(g):
    if not g: return (0,0,0,0,0)
    walls = [r[2] for r in g]; cpus = [r[3]/3600 for r in g]
    return (len(g), sum(walls)/len(g)/60, min(walls)/60, max(walls)/60, sum(cpus)/len(g))

ns, aws, mws, xws, acs = stats(small)
nb, awb, mwb, xwb, acb = stats(big)
nt, awt, mwt, xwt, act = stats(rows)

lines = [r"\begin{table}[H]",
         r"\caption{Temps de calcul (\'ecoul\'e = wall-clock, CPU = temps utilisateur+syst\`eme cumul\'e sur tous les c\oe urs) sur les " + str(nt) + " jobs termin\\'es au moment de la r\\'edaction.}",
         r"\label{tab:timing}",
         r"\centering", r"\small",
         r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
         r"Cat\'egorie & $n$ & Wall moy. (min) & Wall min (min) & Wall max (min) & CPU-h moy. \\",
         r"\midrule",
         f"Petits (8 c\\oe urs, N4--N11) & {ns} & {aws:.1f} & {mws:.1f} & {xws:.1f} & {acs:.2f} \\\\",
         f"Gros (24/32 c\\oe urs, N12--N16) & {nb} & {awb:.1f} & {mwb:.1f} & {xwb:.1f} & {acb:.2f} \\\\",
         r"\midrule",
         f"\\textbf{{Total}} & {nt} & {awt:.1f} & {mwt:.1f} & {xwt:.1f} & {act:.2f} \\\\",
         r"\bottomrule", r"\end{tabular}",
         r"\end{table}"]
with open("/home/gilles/polyN/orca_jobs/report/table_timing.tex", "w") as fh:
    fh.write("\n".join(lines))
print(f"table_timing.tex: n_total={nt} n_small={ns} n_big={nb}")
