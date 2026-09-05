#!/usr/bin/env python3
"""Convert GFN2-xTB xyz structures (resultats/seeds_pubchem/xyz_clean) into
ORCA input files, one per job directory, using the method block of test.inp
as template. Charge is read from the family embedded in the filename
(<formula>_<family>_<family>_<rank>.xyz, family in {neutral,cation,anion} ->
charge 0/+1/-1). Multiplicity is the lowest one consistent with the total
electron count (singlet if even, doublet if odd), matching the uhf=0
(automatic minimal spin) setting used for the GFN2-xTB screening.
"""
import argparse
import pathlib
import re
import sys

ATOMIC_NUMBER = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17,
}

CHARGE_BY_FAMILY = {"neutral": 0, "cation": 1, "anion": -1}

NAME_RE = re.compile(r"^(?P<formula>[A-Za-z0-9]+)_(?P<family>neutral|cation|anion)_(?P=family)_(?P<rank>\d+)$")


def split_template(template_text: str) -> str:
    """Return the template text with everything up to (and including) the
    line that opens the '*xyz' block stripped off, keeping only the header
    (route line, %pal block, etc.)."""
    lines = template_text.splitlines()
    header = []
    for line in lines:
        if line.strip().startswith("*xyz"):
            break
        header.append(line)
    return "\n".join(header).rstrip("\n") + "\n"


def charge_and_multiplicity(atoms, family):
    charge = CHARGE_BY_FAMILY[family]
    n_electrons = sum(ATOMIC_NUMBER[el] for el in atoms) - charge
    multiplicity = 2 if n_electrons % 2 else 1
    return charge, multiplicity


def parse_xyz(path):
    lines = path.read_text().splitlines()
    n_atoms = int(lines[0].strip())
    atoms = []
    coords = []
    for line in lines[2:2 + n_atoms]:
        parts = line.split()
        atoms.append(parts[0])
        coords.append(tuple(parts[1:4]))
    return atoms, coords


def build_input(header, charge, multiplicity, atoms, coords):
    body = [f"*xyz {charge} {multiplicity}"]
    for el, (x, y, z) in zip(atoms, coords):
        body.append(f" {el:<3s} {x:>18s} {y:>18s} {z:>18s}")
    body.append("*")
    return header + "\n".join(body) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default="test.inp", type=pathlib.Path)
    ap.add_argument("--xyz-dir", default="xyz_gfn2xtb", type=pathlib.Path)
    ap.add_argument("--jobs-dir", default="jobs", type=pathlib.Path)
    ap.add_argument("--jobs-list", default="jobs_list.txt", type=pathlib.Path)
    args = ap.parse_args()

    header = split_template(args.template.read_text())

    xyz_files = sorted(args.xyz_dir.glob("*.xyz"))
    if not xyz_files:
        sys.exit(f"No xyz files found in {args.xyz_dir}")

    args.jobs_dir.mkdir(parents=True, exist_ok=True)
    job_dirs = []
    for xyz_path in xyz_files:
        name = xyz_path.stem
        m = NAME_RE.match(name)
        if not m:
            print(f"WARNING: skipping unrecognized filename {xyz_path.name}", file=sys.stderr)
            continue
        family = m.group("family")
        atoms, coords = parse_xyz(xyz_path)
        charge, multiplicity = charge_and_multiplicity(atoms, family)

        job_dir = args.jobs_dir / name
        job_dir.mkdir(exist_ok=True)
        inp_text = build_input(header, charge, multiplicity, atoms, coords)
        (job_dir / f"{name}.inp").write_text(inp_text)
        job_dirs.append(job_dir)

    args.jobs_list.write_text("\n".join(str(d) for d in job_dirs) + "\n")
    print(f"Generated {len(job_dirs)} ORCA inputs under {args.jobs_dir}/ "
          f"(list: {args.jobs_list})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
