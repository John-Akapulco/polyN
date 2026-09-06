import csv, os, shutil

ROOT = "/home/gilles/polyN/orca_jobs"
CCSDT_ROOT = f"{ROOT}/ccsdt_jobs"
XYZ_DIR = f"{ROOT}/xyz_dft_wb97xd4"

RESYM_ROUTE = "! WB97X-D4 aug-cc-pVTZ TightSCF LooseOpt UseSym DEFGRID3"
CCSDT_ROUTE = "! DLPNO-CCSD(T)-F12 cc-pVTZ-F12 cc-pVTZ-F12-CABS cc-pVTZ/C TightPNO TightSCF MORead"

def ncores_for(n):
    if n <= 5:
        return 8
    if n <= 8:
        return 16
    if n <= 11:
        return 24
    return 48

def read_xyz_block(path):
    lines = open(path).read().splitlines()
    n = int(lines[0].strip())
    return lines[2:2 + n]

summ = {r["name"]: r for r in csv.DictReader(open(f"{ROOT}/results_summary.csv"))}

selection = list(csv.DictReader(open("/tmp/ccsdt_selection.csv")))
jobs = []  # (name, n, charge, mult, coords, ncores)

# Molecules de reference / hors criblage N_x, geometrie de depart generique
# (symetrisee des l'etape resym via UseSym) : N2 (reference DeltaHf des
# neutres), NH4+ tetraedrique et NH2- coudee (especes azotees-hydrogenees
# demandees pour etudier des reactions chimiques annexes).
jobs.append(("N2", 2, 0, 1,
             [" N   0.0000000   0.0000000   0.0000000",
              " N   0.0000000   0.0000000   1.0977000"],
             ncores_for(2)))
jobs.append(("NH4_cation", 5, 1, 1,
             [" N   0.0000000   0.0000000   0.0000000",
              " H   0.5894746   0.5894746   0.5894746",
              " H   0.5894746  -0.5894746  -0.5894746",
              " H  -0.5894746   0.5894746  -0.5894746",
              " H  -0.5894746  -0.5894746   0.5894746"],
             ncores_for(5)))
jobs.append(("NH2_anion", 3, -1, 1,
             [" N   0.0000000   0.0000000   0.0000000",
              " H   0.8116511   0.0000000   0.6341313",
              " H  -0.8116511   0.0000000   0.6341313"],
             ncores_for(3)))

for row in selection:
    name = row["name"]
    n = int(row["n"])
    r = summ[name]
    charge, mult = int(r["charge"]), int(r["multiplicity"])
    coords = read_xyz_block(f"{XYZ_DIR}/{name}.xyz")
    jobs.append((name, n, charge, mult, coords, ncores_for(n)))

os.makedirs(CCSDT_ROOT, exist_ok=True)
manifest = []
for name, n, charge, mult, coords, ncores in jobs:
    job_dir = f"{CCSDT_ROOT}/{name}"
    os.makedirs(job_dir, exist_ok=True)

    resym = (f"{RESYM_ROUTE}\n%pal\n  nprocs {ncores}\nend\n"
             f"*xyz {charge} {mult}\n" + "\n".join(coords) + "\n*\n")
    with open(f"{job_dir}/{name}_resym.inp", "w") as fh:
        fh.write(resym)

    # Le step CCSD(T) est ecrit ici avec *xyzfile pointant vers la geometrie
    # que produira le resym (nom fixe, connu a l'avance) -- le fichier n'est
    # valide qu'une fois {name}_resym.xyz/.gbw ecrits par l'etape precedente,
    # ce que le wrapper shell garantit (execution sequentielle, || exit 1).
    ccsdt = (f"{CCSDT_ROUTE}\n%moinp \"{name}_resym.gbw\"\n"
             f"%pal\n  nprocs {ncores}\nend\n"
             f"*xyzfile {charge} {mult} {name}_resym.xyz\n")
    with open(f"{job_dir}/{name}_ccsdt.inp", "w") as fh:
        fh.write(ccsdt)

    manifest.append({"name": name, "n": n, "charge": charge, "mult": mult, "ncores": ncores})

with open(f"{CCSDT_ROOT}/jobs_list_ccsdt.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["name", "n", "charge", "mult", "ncores"])
    w.writeheader()
    w.writerows(manifest)

print(f"{len(jobs)} jobs CCSD(T) prepares dans {CCSDT_ROOT}/ (resym.inp + ccsdt.inp chacun)")
from collections import Counter
print("repartition coeurs:", Counter(j[5] for j in jobs))
