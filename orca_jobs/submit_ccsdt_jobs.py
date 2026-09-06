#!/usr/bin/env python3
"""Soumet les 49 jobs CCSD(T) prepares par build_ccsdt_jobs.py. Chaque job
est une chaine sequentielle resym (DFT WB97X-D4 LooseOpt+UseSym) -> ccsdt
(DLPNO-CCSD(T)-F12 single-point, MORead depuis le gbw du resym), chronometree
separement (deux fichiers .time) pour obtenir les statistiques temps
humain/CPU par etape et par nombre de coeurs. Pas de gestion de noeud
manuelle : sbatch + le scheduler SLURM placent chaque job des que les coeurs
demandes sont libres (aucun besoin d'un dispatcher maison ici, contrairement
a la campagne DFT initiale)."""
import csv, os, subprocess, sys

ROOT = "/home/gilles/polyN/orca_jobs"
CCSDT_ROOT = f"{ROOT}/ccsdt_jobs"
ORCA_DIR = "/opt/ohpc/pub/software/orca_6_1_0_linux_x86-64_shared_openmpi418"
ORCA_BIN = f"{ORCA_DIR}/orca"
MODS = "module load gnu12/12.3.0; module load openmpi4/4.1.6"

rows = list(csv.DictReader(open(f"{CCSDT_ROOT}/jobs_list_ccsdt.csv")))
if len(sys.argv) > 1:
    only = set(sys.argv[1:])
    rows = [r for r in rows if r["name"] in only]
    print(f"filtre applique : {len(rows)}/{len(only)} noms demandes trouves dans le manifeste")

for r in rows:
    name, ncores = r["name"], r["ncores"]
    job_dir = f"{CCSDT_ROOT}/{name}"
    chain = (
        "/usr/bin/time -v {orca} {name}_resym.inp > {name}_resym.out 2> {name}_resym.time; "
        "if grep -q 'THE OPTIMIZATION HAS CONVERGED' {name}_resym.out; then "
        "  /usr/bin/time -v {orca} {name}_ccsdt.inp > {name}_ccsdt.out 2> {name}_ccsdt.time; "
        "else "
        "  echo 'resym did not converge -- ccsdt skipped' > {name}_ccsdt.SKIPPED; "
        "fi"
    ).format(orca=ORCA_BIN, name=name)
    wrap = "{mods}; export PATH={bin_dir}:$PATH; sh -c '{chain}'".format(
        mods=MODS, bin_dir=ORCA_DIR, chain=chain.replace("'", "'\\''"))
    cmd = [
        "sbatch", "--job-name=ccsdt_{}".format(name),
        "--nodes=1", "--ntasks={}".format(ncores),
        "--time=96:00:00",
        "--chdir={}".format(job_dir),
        "--output={}.slurm.log".format(name),
        "--wrap={}".format(wrap),
    ]
    out = subprocess.check_output(cmd).decode().strip()
    print("submitted {} @ {} cores -> {}".format(name, ncores, out))
