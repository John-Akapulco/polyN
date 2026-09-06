#!/usr/bin/env python3
"""Soumet, dans l'ordre, autant de structures de
ccsdt_jobs/remaining_pool_n4_n8.csv que le permettent les coeurs
actuellement libres sur le cluster (colonne 'defq'). Prepare les inputs
(resym + ccsdt) comme submit_ccsdt_jobs.py, les ajoute au manifeste
jobs_list_ccsdt.csv, les soumet, puis retire de la liste d'attente les
structures effectivement soumises. Sans argument : ne fait rien de
destructif, relancable a volonte (idempotent si la liste d'attente est
vide ou si aucun coeur ne suffit)."""
import csv, os, subprocess, sys

ROOT = "/home/gilles/polyN/orca_jobs"
CCSDT_ROOT = f"{ROOT}/ccsdt_jobs"
XYZ_DIR = f"{ROOT}/xyz_dft_wb97xd4"
POOL_FILE = f"{CCSDT_ROOT}/remaining_pool_n4_n8.csv"
ORCA_DIR = "/opt/ohpc/pub/software/orca_6_1_0_linux_x86-64_shared_openmpi418"
ORCA_BIN = f"{ORCA_DIR}/orca"
MODS = "module load gnu12/12.3.0; module load openmpi4/4.1.6"
RESYM_ROUTE = "! WB97X-D4 aug-cc-pVTZ TightSCF LooseOpt UseSym DEFGRID3"
CCSDT_ROUTE = "! DLPNO-CCSD(T)-F12 cc-pVTZ-F12 cc-pVTZ-F12-CABS cc-pVTZ/C TightPNO TightSCF MORead"


def free_cores():
    out = subprocess.check_output(
        "sinfo -N -h -o '%N %C' -p defq | sort -u", shell=True).decode()
    total = 0
    for line in out.splitlines():
        _, cores = line.split()
        idle = int(cores.split("/")[1])
        total += idle
    return total


def read_xyz_block(path):
    lines = open(path).read().splitlines()
    n = int(lines[0].strip())
    return lines[2:2 + n]


def prepare(name, ncores, summ):
    r = summ[name]
    charge, mult = int(r["charge"]), int(r["multiplicity"])
    coords = read_xyz_block(f"{XYZ_DIR}/{name}.xyz")
    job_dir = f"{CCSDT_ROOT}/{name}"
    os.makedirs(job_dir, exist_ok=True)
    with open(f"{job_dir}/{name}_resym.inp", "w") as fh:
        fh.write(f"{RESYM_ROUTE}\n%pal\n  nprocs {ncores}\nend\n"
                  f"*xyz {charge} {mult}\n" + "\n".join(coords) + "\n*\n")
    with open(f"{job_dir}/{name}_ccsdt.inp", "w") as fh:
        fh.write(f"{CCSDT_ROUTE}\n%moinp \"{name}_resym.gbw\"\n"
                  f"%pal\n  nprocs {ncores}\nend\n"
                  f"*xyzfile {charge} {mult} {name}_resym.xyz\n")
    return {"name": name, "n": int(r["n_count"]), "charge": charge,
            "mult": mult, "ncores": ncores}


def submit(name, ncores):
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
    return out


def main():
    if not os.path.isfile(POOL_FILE):
        print("liste d'attente vide ou absente -- rien a faire")
        return
    pool = list(csv.DictReader(open(POOL_FILE)))
    if not pool:
        print("liste d'attente vide -- rien a faire")
        return

    avail = free_cores()
    print(f"{avail} coeurs libres, {len(pool)} structures en attente")

    summ = {r["name"]: r for r in csv.DictReader(open(f"{ROOT}/results_summary.csv"))}
    submitted, remaining = [], []
    for r in pool:
        ncores = int(r["ncores"])
        if ncores <= avail:
            m = prepare(r["name"], ncores, summ)
            existing = list(csv.DictReader(open(f"{CCSDT_ROOT}/jobs_list_ccsdt.csv")))
            with open(f"{CCSDT_ROOT}/jobs_list_ccsdt.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=["name", "n", "charge", "mult", "ncores"])
                w.writeheader()
                w.writerows(existing + [m])
            job_id = submit(r["name"], ncores)
            avail -= ncores
            submitted.append(r["name"])
            print(f"soumis {r['name']} @ {ncores} coeurs -> {job_id} ({avail} coeurs restants)")
        else:
            remaining.append(r)

    with open(POOL_FILE, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["name", "ncores"])
        w.writeheader()
        w.writerows(remaining)

    print(f"\n{len(submitted)} structures soumises, {len(remaining)} restent en attente de coeurs")


if __name__ == "__main__":
    main()
