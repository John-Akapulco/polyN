import glob, os, re, csv

HARTREE_TO_KCAL = 627.5094740631
E_N2 = -5.763935420175
E_N5_ANION = -14.760899219951      # N5_anion_anion_001, D5h, matches lit. exactly
E_N5_CATION_INTERNAL = -13.816934622594  # N5_cation_cation_004, this run's internal zero (C1)
DHF_N5_ANION = -106.4    # lit. CCSD(T), D5h pentagon
DHF_N5_CATION = 195.9    # lit. CCSD(T)/X-ray, C2v Vchain
CATION_OFFSET_KCAL = -105.63  # E_react(N5_cation_cation_001) as printed -- Vchain vs internal ref

NAME_RE = re.compile(r"^N(?P<n>\d+)_(?P<family>neutral|cation|anion)_(?P=family)_(?P<rank>\d+)$")

def dhf_kcal(n, family, e_react_kcal):
    if family == "neutral":
        return e_react_kcal  # dHf(N2)=0, so dHf(Nx) = E_react directly
    if family == "anion":
        return DHF_N5_ANION + e_react_kcal
    # cation: rescale from the internal (C1) reference to the literature-matched Vchain
    return DHF_N5_CATION + (e_react_kcal - CATION_OFFSET_KCAL)

rows = []
for path in sorted(glob.glob("/home/gilles/polyN/orca_jobs/xyz_gfn2xtb/*.xyz")):
    name = os.path.splitext(os.path.basename(path))[0]
    m = NAME_RE.match(name)
    if not m:
        continue
    n, family, rank = int(m.group("n")), m.group("family"), int(m.group("rank"))
    comment = open(path).read().splitlines()[1]
    em = re.search(r"E_react=(-?[\d.]+)", comment)
    e_react_kcal = float(em.group(1))
    dhf = dhf_kcal(n, family, e_react_kcal)
    rows.append({"name": name, "n": n, "family": family, "rank": rank,
                 "e_react_kcalmol_xtb": e_react_kcal, "dHf_kcalmol_xtb": round(dhf, 2)})

with open("/home/gilles/polyN/orca_jobs/results_dHf_xtb.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["name","n","family","rank","e_react_kcalmol_xtb","dHf_kcalmol_xtb"])
    w.writeheader()
    w.writerows(rows)

print("computed dHf for", len(rows), "structures -> results_dHf_xtb.csv")
# sanity checks
for r in rows:
    if r["name"] in ("N5_anion_anion_001","N5_cation_cation_001","N5_cation_cation_004"):
        print(r)
