#!/usr/bin/env python3
"""Fill node07/node08/node15 (big clusters, N12-N16) and node16/node13 (small
clusters, N4-N11) with the 193-structure production campaign, re-polling idle
cores on those five nodes and launching whatever fits until jobs_list.txt is
exhausted. Patches each job's %pal nprocs to match the lane's core count
before submitting (the .inp files all hardcode nprocs 8 from the generator).

Meant to run detached (nohup ... & disown) since the campaign can take a long
time and must keep resubmitting as slots free up. Tracks progress in
dispatch_submitted.txt (names already handed to sbatch -- never resubmitted)
and logs every action to dispatch_events.log.
"""
import os
import re
import subprocess
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
JOBS_LIST = os.path.join(ROOT, "jobs_list.txt")
SUBMITTED_LOG = os.path.join(ROOT, "dispatch_submitted.txt")
EVENT_LOG = os.path.join(ROOT, "dispatch_events.log")

NAME_RE = re.compile(r"^N(?P<n>\d+)_(?P<family>neutral|cation|anion)_(?P=family)_(?P<rank>\d+)$")

# (node, cores-per-job, category). Originally 24/32-core lanes on
# node07/node08/node15 for big clusters (N12-N16) and 8-core lanes on
# node16/node13 for small ones (N4-N11); once the big queue emptied, all
# lanes were converted to 8-core small ones (see below) and extended to
# every reachable node, shared or not.
NODE_LANES = [
    # All switched to 8-core small lanes: the big (N12-N16) queue emptied, and
    # 8 cores is the efficient spot for small clusters anyway (see the core
    # benchmark -- N4-N11 gain little past 8 cores, so keeping node07/08 at
    # 24c/job for small jobs would just waste core-hours for no speed benefit).
    ("node07", 8, "small"),
    ("node08", 8, "small"),
    ("node15", 8, "small"),
    ("node16", 8, "small"),
    ("node13", 8, "small"),
    ("node06", 8, "small"),
    # Shared nodes -- other users' jobs live here too, so idle capacity is
    # whatever they aren't using right now. The existing pending-aware
    # throttle (my_pending_names) keeps us from over-submitting into a node
    # that's actually busy; node09 is included even though it showed 0 idle
    # at the time this was added, since the poll re-checks it live anyway.
    ("node01", 8, "small"),
    ("node02", 8, "small"),
    ("node03", 8, "small"),
    ("node04", 8, "small"),
    ("node09", 8, "small"),
    ("node10", 8, "small"),
    ("node11", 8, "small"),
    ("node12", 8, "small"),
    ("node14", 8, "small"),
]
BIG_MIN_N = 12  # N12-N16 => "big", N4-N11 => "small"
PARTITION = "defq"

ORCA_DIR = "/opt/ohpc/pub/software/orca_6_1_0_linux_x86-64_shared_openmpi418"
ORCA_BIN = ORCA_DIR + "/orca"
MODULE_CMDS = "module load gnu12/12.3.0; module load openmpi4/4.1.6"
POLL_SECONDS = 90


def log(msg):
    line = "[{}] {}".format(time.strftime("%Y-%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    with open(EVENT_LOG, "a") as fh:
        fh.write(line + "\n")


def load_queues():
    with open(JOBS_LIST) as fh:
        names = [os.path.basename(l.strip()) for l in fh if l.strip()]
    already = set()
    if os.path.exists(SUBMITTED_LOG):
        with open(SUBMITTED_LOG) as fh:
            already = set(l.strip() for l in fh if l.strip())
    big, small = [], []
    for name in names:
        if name in already:
            continue
        m = NAME_RE.match(name)
        if not m:
            log("WARNING: unrecognized name in jobs_list.txt: {}".format(name))
            continue
        n = int(m.group("n"))
        (big if n >= BIG_MIN_N else small).append(name)
    return big, small


def idle_cores(node):
    out = subprocess.check_output(
        ["sinfo", "-N", "-h", "-p", PARTITION, "--nodes=" + node, "-o", "%C"]
    ).decode().split()
    # first (only, thanks to -p defq) row: "alloc/idle/other/total"
    a, i, o, t = out[0].split("/")
    return int(i)


def my_pending_names():
    """Names (not job IDs) of our own prod_ jobs currently PENDING in squeue.
    sinfo's idle count can lag a submission by more than one poll interval
    on this cluster's scheduler, so once a node has one of our jobs queued
    we hold off adding more there until it actually starts running --
    otherwise the scheduling lag causes runaway over-submission."""
    out = subprocess.check_output(
        ["squeue", "-u", os.environ.get("USER", ""), "-h", "-t", "PD", "-o", "%j"]
    ).decode().splitlines()
    return set(j[len("prod_"):] for j in out if j.startswith("prod_"))


def load_targets():
    """name -> node, reconstructed from our own event log (each submission
    logs 'submitted <name> on <node> @ <n> cores'). Rebuilt fresh each poll
    so a restarted dispatcher recovers the mapping with no extra state file."""
    targets = {}
    if not os.path.exists(EVENT_LOG):
        return targets
    pat = re.compile(r"submitted (\S+) on (\S+) @")
    with open(EVENT_LOG) as fh:
        for line in fh:
            m = pat.search(line)
            if m:
                targets[m.group(1)] = m.group(2)
    return targets


def set_nprocs(inp_path, n):
    with open(inp_path) as fh:
        text = fh.read()
    text = re.sub(r"nprocs \d+", "nprocs {}".format(n), text)
    with open(inp_path, "w") as fh:
        fh.write(text)


def extract_xyz_block(inp_path):
    """(charge, mult, ['el x y z', ...]) from a '*xyz chg mult' ... '*' block."""
    with open(inp_path) as fh:
        lines = fh.read().splitlines()
    for idx, line in enumerate(lines):
        m = re.match(r"\*xyz\s+(-?\d+)\s+(\d+)", line.strip())
        if m:
            coords = []
            for l in lines[idx + 1:]:
                if l.strip() == "*":
                    break
                coords.append(l)
            return int(m.group(1)), int(m.group(2)), coords
    raise ValueError("no *xyz block found in {}".format(inp_path))


PAL_BLOCK = "%pal\n  nprocs {ncores}\nend\n"

# Step 1: plain Opt, no symmetry -- this is the level the whole campaign has
# used so far, kept unchanged so the minimum-finding itself is never biased
# by a symmetry constraint (avoids trapping a Jahn-Teller-distorted system in
# an artificial high-symmetry saddle point).
STEP1_ROUTE = "! WB97X-D4 aug-cc-pVTZ TightSCF Opt DEFGRID3"
# Step 2: quick re-optimization from that already-converged geometry with
# UseSym on -- snaps it onto its exact point group (LooseOpt: few cycles
# needed, we're polishing, not searching).
STEP2_ROUTE = "! WB97X-D4 aug-cc-pVTZ TightSCF LooseOpt UseSym DEFGRID3"
# Step 3: Hessian/Freq on the symmetrized geometry -- full point-group
# diagonalization benefit (all finite point groups, not just D2h subgroups),
# exactly-degenerate frequencies, clean bond distances for the N-N analysis.
STEP3_ROUTE = "! WB97X-D4 aug-cc-pVTZ TightSCF Freq UseSym DEFGRID3"


def build_chained_inputs(name, job_dir, ncores):
    """Write <name>_s1/_s2/_s3.inp implementing Opt -> symmetrize -> Freq.
    Returns the shell command that runs all three in sequence and copies the
    final step's outputs to the plain <name>.* names harvest_results.py
    already expects (nothing about the harvest side has to change)."""
    orig_inp = os.path.join(job_dir, name + ".inp")
    charge, mult, coords = extract_xyz_block(orig_inp)
    pal = PAL_BLOCK.format(ncores=ncores)

    s1 = "{}\n{}*xyz {} {}\n{}\n*\n".format(STEP1_ROUTE, pal, charge, mult, "\n".join(coords))
    with open(os.path.join(job_dir, name + "_s1.inp"), "w") as fh:
        fh.write(s1)

    s2 = "{}\n{}*xyzfile {} {} {}_s1.xyz\n".format(STEP2_ROUTE, pal, charge, mult, name)
    with open(os.path.join(job_dir, name + "_s2.inp"), "w") as fh:
        fh.write(s2)

    s3 = "{}\n{}*xyzfile {} {} {}_s2.xyz\n".format(STEP3_ROUTE, pal, charge, mult, name)
    with open(os.path.join(job_dir, name + "_s3.inp"), "w") as fh:
        fh.write(s3)

    steps = ""
    for step in ("s1", "s2", "s3"):
        steps += "{orca} {name}_{step}.inp > {name}_{step}.out || exit 1\n".format(
            orca=ORCA_BIN, name=name, step=step)
    finalize = "\n".join(
        "cp {name}_s3.{ext} {name}.{ext} 2>/dev/null".format(name=name, ext=ext)
        for ext in ("out", "property.txt", "gbw", "densities", "hess")
    )
    # step 3 is Freq-only (no optimizer), so it never writes its own .xyz --
    # the geometry it computed the Hessian at is step 2's (symmetrized) one.
    finalize += "\ncp {name}_s2.xyz {name}.xyz 2>/dev/null".format(name=name)
    return steps + finalize + "\n"


def submit_single(name, job_dir, node, ncores):
    """Original approach: one combined Opt+Freq job, no symmetry. Used for
    the small structures (N4-N11) -- the Opt->symmetrize->Freq chain nearly
    doubles wall time (measured on N5_anion_anion_001: 7:41 -> 15:35), not
    worth it for these cheap/fast jobs; the imaginary-frequency check in
    harvest_results.py is the quality safety net instead."""
    inp_path = os.path.join(job_dir, name + ".inp")
    set_nprocs(inp_path, ncores)
    wrap = ("{mods}; export PATH={bin_dir}:$PATH; "
            "/usr/bin/time -v {orca} {name}.inp > {name}.out 2> {name}.time").format(
        mods=MODULE_CMDS, bin_dir=ORCA_DIR, orca=ORCA_BIN, name=name,
    )
    return wrap


def submit_chained(name, job_dir, node, ncores):
    """Opt (no sym) -> quick LooseOpt+UseSym symmetrization -> Freq+UseSym.
    Used only for the big structures (N12-N16), where the extra ~2x wall
    time is worth the exact point-group geometry/degenerate frequencies and
    where full-point-group Hessian diagonalization pays off most."""
    chain_cmds = build_chained_inputs(name, job_dir, ncores)
    wrap = "{mods}; export PATH={bin_dir}:$PATH; /usr/bin/time -v sh -c '{chain}' 2> {name}.time".format(
        mods=MODULE_CMDS, bin_dir=ORCA_DIR, chain=chain_cmds.replace("'", "'\\''"), name=name,
    )
    return wrap


def submit(name, node, ncores, category):
    job_dir = os.path.join(ROOT, "jobs", name)
    if category == "big":
        wrap = submit_chained(name, job_dir, node, ncores)
        mode = "Opt->symmetrize->Freq chain"
    else:
        wrap = submit_single(name, job_dir, node, ncores)
        mode = "single Opt+Freq job"
    cmd = [
        "sbatch", "--job-name=prod_{}".format(name),
        "--nodes=1", "--ntasks={}".format(ncores),
        "--nodelist={}".format(node),
        "--time=96:00:00",
        "--chdir={}".format(job_dir),
        "--output={}.slurm.log".format(name),
        "--wrap={}".format(wrap),
    ]
    out = subprocess.check_output(cmd).decode().strip()
    log("submitted {} on {} @ {} cores ({}) -> {}".format(name, node, ncores, mode, out))
    with open(SUBMITTED_LOG, "a") as fh:
        fh.write(name + "\n")


def main():
    log("dispatcher starting (pid {})".format(os.getpid()))
    while True:
        big, small = load_queues()
        if not big and not small:
            log("jobs_list.txt exhausted -- all 193 handed to sbatch. Dispatcher exiting.")
            break
        pending = my_pending_names()
        targets = load_targets()
        for node, ncores, category in NODE_LANES:
            queue = big if category == "big" else small
            if any(targets.get(n) == node for n in pending):
                log("{}: still has a pending job there, skipping this cycle".format(node))
                continue
            try:
                idle = idle_cores(node)
            except Exception as e:
                log("sinfo failed for {}: {}".format(node, e))
                continue
            n_fit = idle // ncores
            for _ in range(n_fit):
                if not queue:
                    break
                name = queue.pop(0)
                try:
                    submit(name, node, ncores, category)
                except Exception as e:
                    log("submit failed for {}: {}".format(name, e))
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
