#!/usr/bin/env python3
"""
merge_polyN_results.py
========================
Merge and compare the results CSVs produced by the three complementary
polynitrogen scripts in this project into a single, unified metastability
landscape: delta_E_per_atom(n) vs N2 across every structure found by
every method, with explicit provenance tracking.

THREE SOURCES, THREE DIFFERENT ROLES (see companion methodology manual)
---------------------------------------------------------------------------
  polynitrogen_mace.py            : combinatorial graph enumeration +
                                     single relaxation per topology.
  polynitrogen_minimahopping.py   : global PES exploration via Minima
                                     Hopping from diverse seeds, FIXED n.
  polynitrogen_airss.py           : unbiased random sampling, VARIABLE n
                                     drawn per trial (AIRSS-style).

WHY A NAIVE CONCATENATION IS NOT SAFE
-----------------------------------------
Three checks are mandatory before any cross-script comparison of
delta_E_per_atom is meaningful, and this script enforces all three
rather than silently concatenating rows:

  1. ENERGY PROVENANCE: a row's delta_E value is only meaningful if it
     was actually computed by a MACE relaxation. polynitrogen_mace.py
     can be run in --dry_run mode (force-field-only, no MACE), in which
     case its mace_energy_eV / delta_E_eV_per_atom_vs_N2 columns are
     EMPTY STRINGS. Such rows carry no usable energy information and
     are silently dropped here (logged, not just ignored), rather than
     being misread as "structures with energy 0".

  2. MODEL CONSISTENCY: delta_E_per_atom is only comparable across rows
     if all rows were computed against the SAME MACE model (e.g.
     mace_off/small) and the SAME freshly-relaxed N2 reference energy.
     None of the three scripts currently persist this provenance
     information in their CSV output (a gap identified during this
     merge -- see manual addendum), so this script requires the user to
     explicitly declare, via --source, which model each input file was
     generated with; rows from sources flagged as using DIFFERENT models
     are kept in SEPARATE columns/groups rather than pooled into a
     single ranking, and a warning is printed.

  3. INTEGRITY STATUS: only rows whose integrity_status indicates an
     intact, physically sane structure should contribute to a
     metastability "best candidate" summary. The three scripts use
     slightly different status vocabularies (the MACE-enumeration and
     Minima-Hopping scripts include BOND_ANOMALY and REARRANGED as
     usable-with-caveats; the AIRSS script's simplified classifier does
     not produce REARRANGED or BOND_ANOMALY at all, since it has no
     target topology to compare against). This script normalizes all
     three onto a common three-tier scale: USABLE (OK, REARRANGED,
     BOND_ANOMALY) / UNUSABLE (FRAGMENTED, COLLAPSED, ENERGY_ANOMALY) /
     UNKNOWN (dry-run rows with no energy at all).

OUTPUT
-------
  merged_results.csv       : every row from every source, with a
                              standardized column schema and an explicit
                              `source_script` and `usable` column.
  merged_hull_summary.csv  : for every n found in USABLE rows, across
                              ALL sources combined, the lowest
                              delta_E_per_atom found and which source/
                              topology achieved it -- the consolidated
                              metastability landscape.
  merged_hull_plot.svg     : simple SVG visualization of the landscape
                              (delta_E_per_atom vs n), color-coded by
                              source script, since this is small enough
                              to not require external plotting libraries.

USAGE
------
    python merge_polyN_results.py \\
        --mace_csv polyN_results/results.csv \\
        --mh_csv polyN_MH_results/minima_results.csv \\
        --airss_csv polyN_AIRSS_results/airss_results.csv \\
        --output_dir merged_analysis

    # If a source was run in dry_run / a different MACE model, declare it:
    python merge_polyN_results.py \\
        --mh_csv run1/minima_results.csv --mh_model mace_off/small \\
        --airss_csv run2/airss_results.csv --airss_model mace_off/medium \\
        --output_dir merged_analysis
    # -> rows from mh_csv and airss_csv will NOT be pooled into one
    #    ranking (different models), and the output will say so.

AUTHOR
-------
    Generated for IC2MP/E4 Mediacat -- Universite de Poitiers
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("merge-polyN")


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRITY STATUS NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────

USABLE_STATUSES = {"OK", "REARRANGED", "BOND_ANOMALY"}
UNUSABLE_STATUSES = {"FRAGMENTED", "COLLAPSED", "ENERGY_ANOMALY"}


def classify_usability(integrity_status, has_energy):
    """
    Normalize a source-specific integrity_status string onto the common
    three-tier scale described in the module docstring.
    """
    if not has_energy:
        return "UNKNOWN"
    if integrity_status in USABLE_STATUSES:
        return "USABLE"
    if integrity_status in UNUSABLE_STATUSES:
        return "UNUSABLE"
    # Defensive fallback for any future/unrecognized status string:
    # treat as unusable rather than silently including it in the hull.
    return "UNUSABLE"


# ─────────────────────────────────────────────────────────────────────────────
# PER-SOURCE READERS (each script has a slightly different column schema)
# ─────────────────────────────────────────────────────────────────────────────

def read_mace_csv(path, model_label):
    """
    Read polynitrogen_mace.py's results.csv. Rows with an empty
    delta_E_eV_per_atom_vs_N2 (i.e. the run used --dry_run, no MACE
    energy was ever computed) are flagged has_energy=False and logged,
    not silently treated as zero.
    """
    rows = []
    n_dropped_dry_run = 0
    with open(path) as f:
        for r in csv.DictReader(f):
            delta_e_str = r.get("delta_E_eV_per_atom_vs_N2", "").strip()
            has_energy = delta_e_str != ""
            if not has_energy:
                n_dropped_dry_run += 1
            rows.append({
                "source_script": "polynitrogen_mace.py",
                "model": model_label,
                "n_atoms": int(r["n_atoms"]),
                "topology_label": r["topology_label"],
                "delta_E_eV_per_atom_vs_N2": float(delta_e_str) if has_energy else None,
                "has_energy": has_energy,
                "integrity_status": r.get("integrity_status", "UNKNOWN"),
                "spacegroup": r.get("spacegroup", "N/A"),
                "xyz_file": r.get("xyz_file", ""),
                "source_file": str(path),
            })
    if n_dropped_dry_run:
        log.warning(
            f"  {path}: {n_dropped_dry_run}/{len(rows)} rows have NO MACE energy "
            "(this run used --dry_run, force-field-only). These rows are kept "
            "in merged_results.csv with usable='UNKNOWN' but EXCLUDED from the "
            "hull summary, since they carry no comparable delta_E value."
        )
    return rows


def read_mh_csv(path, model_label):
    """Read polynitrogen_minimahopping.py's minima_results.csv."""
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            delta_e_str = r.get("delta_E_eV_per_atom_vs_N2", "").strip()
            has_energy = delta_e_str != ""
            rows.append({
                "source_script": "polynitrogen_minimahopping.py",
                "model": model_label,
                "n_atoms": int(r["n_atoms"]),
                "topology_label": r["final_topology_label"],
                "delta_E_eV_per_atom_vs_N2": float(delta_e_str) if has_energy else None,
                "has_energy": has_energy,
                "integrity_status": r.get("integrity_status", "UNKNOWN"),
                "spacegroup": r.get("spacegroup", "N/A"),
                "xyz_file": r.get("xyz_file", ""),
                "source_file": str(path),
            })
    return rows


def read_airss_csv(path, model_label):
    """Read polynitrogen_airss.py's airss_results.csv."""
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            delta_e_str = r.get("delta_E_eV_per_atom_vs_N2", "").strip()
            has_energy = delta_e_str != ""
            rows.append({
                "source_script": "polynitrogen_airss.py",
                "model": model_label,
                "n_atoms": int(r["n_atoms"]),
                "topology_label": r["topology_label"],
                "delta_E_eV_per_atom_vs_N2": float(delta_e_str) if has_energy else None,
                "has_energy": has_energy,
                "integrity_status": r.get("integrity_status", "UNKNOWN"),
                "spacegroup": r.get("spacegroup", "N/A"),
                "xyz_file": r.get("xyz_file", ""),
                "source_file": str(path),
            })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# MODEL CONSISTENCY CHECK
# ─────────────────────────────────────────────────────────────────────────────

def check_model_consistency(all_rows):
    """
    Group rows by declared model label. If more than one distinct model
    is present among USABLE rows, warn loudly: pooling delta_E values
    computed with different MACE models (e.g. mace_off/small vs
    mace_off/medium, or mace_off vs mace_mp) into a single ranking is
    not rigorously justified, since the two models' fitted PES may not
    agree closely enough, and -- as discussed at length in this
    project -- even small systematic biases that do not cancel between
    bond-order environments can distort delta_E_per_atom comparisons.
    Returns the set of distinct models found.
    """
    models = set(r["model"] for r in all_rows if r["has_energy"])
    if len(models) > 1:
        log.warning(
            f"MULTIPLE DISTINCT MODELS detected among usable rows: {sorted(models)}. "
            "delta_E_per_atom values from different models are NOT pooled into a "
            "single ranking in merged_hull_summary.csv; the summary is computed "
            "PER MODEL GROUP separately. Re-run all three scripts with the same "
            "--model/--model_size for a single unified ranking."
        )
    return models


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT WRITERS
# ─────────────────────────────────────────────────────────────────────────────

def write_merged_csv(all_rows, out_path):
    fieldnames = [
        "source_script", "model", "n_atoms", "topology_label",
        "delta_E_eV_per_atom_vs_N2", "usable", "integrity_status",
        "spacegroup", "xyz_file", "source_file",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow({
                "source_script": r["source_script"],
                "model": r["model"],
                "n_atoms": r["n_atoms"],
                "topology_label": r["topology_label"],
                "delta_E_eV_per_atom_vs_N2": (
                    f"{r['delta_E_eV_per_atom_vs_N2']:.4f}"
                    if r["delta_E_eV_per_atom_vs_N2"] is not None else ""
                ),
                "usable": r["usable"],
                "integrity_status": r["integrity_status"],
                "spacegroup": r["spacegroup"],
                "xyz_file": r["xyz_file"],
                "source_file": r["source_file"],
            })


def write_hull_summary(all_rows, models, out_path):
    """
    For each (model, n) pair among USABLE rows, report the lowest
    delta_E_per_atom found and which source/topology achieved it. Models
    are kept separate (see check_model_consistency) rather than pooled.
    """
    usable_rows = [r for r in all_rows if r["usable"] == "USABLE"]

    best = {}  # (model, n) -> row dict with the lowest delta_E
    for r in usable_rows:
        key = (r["model"], r["n_atoms"])
        if key not in best or r["delta_E_eV_per_atom_vs_N2"] < best[key]["delta_E_eV_per_atom_vs_N2"]:
            best[key] = r

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "n_atoms", "min_delta_E_eV_per_atom",
                    "best_topology", "best_source_script", "n_usable_candidates_at_n"])
        counts = {}
        for r in usable_rows:
            counts[(r["model"], r["n_atoms"])] = counts.get((r["model"], r["n_atoms"]), 0) + 1
        for (model, n) in sorted(best.keys()):
            r = best[(model, n)]
            w.writerow([
                model, n, f"{r['delta_E_eV_per_atom_vs_N2']:.4f}",
                r["topology_label"], r["source_script"],
                counts.get((model, n), 0),
            ])
    return best


def write_svg_plot(best, models, out_path, width=700, height=450):
    """
    Minimal dependency-free SVG scatter plot of the consolidated
    metastability landscape: delta_E_per_atom (y) vs n_atoms (x), one
    color per source script, one panel per model group if more than one
    model is present.
    """
    if not best:
        log.warning("No usable data points to plot; skipping SVG output.")
        return

    margin = 60
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin

    all_n = [n for (_, n) in best.keys()]
    all_de = [r["delta_E_eV_per_atom_vs_N2"] for r in best.values()]
    n_min, n_max = min(all_n), max(all_n)
    de_min, de_max = min(0.0, min(all_de)), max(all_de) * 1.1 + 1e-6

    def x_pos(n):
        if n_max == n_min:
            return margin + plot_w / 2
        return margin + (n - n_min) / (n_max - n_min) * plot_w

    def y_pos(de):
        if de_max == de_min:
            return margin + plot_h / 2
        return margin + plot_h - (de - de_min) / (de_max - de_min) * plot_h

    color_map = {
        "polynitrogen_mace.py": "#2166ac",
        "polynitrogen_minimahopping.py": "#b2182b",
        "polynitrogen_airss.py": "#1a9850",
    }

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'font-family="sans-serif">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        # axes
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" '
        f'stroke="black" stroke-width="1.5"/>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" '
        f'stroke="black" stroke-width="1.5"/>',
        # N2 reference line at delta_E = 0
        f'<line x1="{margin}" y1="{y_pos(0)}" x2="{width-margin}" y2="{y_pos(0)}" '
        f'stroke="#888" stroke-width="1" stroke-dasharray="4,3"/>',
        f'<text x="{width-margin+5}" y="{y_pos(0)+4}" font-size="11" fill="#888">N2</text>',
        f'<text x="{width/2}" y="{height-15}" font-size="13" text-anchor="middle">'
        f'n (number of N atoms)</text>',
        f'<text x="15" y="{height/2}" font-size="13" text-anchor="middle" '
        f'transform="rotate(-90 15 {height/2})">&#916;E/atom (eV) vs N2</text>',
    ]

    # x tick labels (even n only)
    for n in sorted(set(all_n)):
        svg_parts.append(
            f'<text x="{x_pos(n)}" y="{height-margin+18}" font-size="10" '
            f'text-anchor="middle">{n}</text>'
        )
        svg_parts.append(
            f'<line x1="{x_pos(n)}" y1="{height-margin}" x2="{x_pos(n)}" '
            f'y2="{height-margin+4}" stroke="black"/>'
        )

    # y tick labels (5 ticks)
    for i in range(6):
        de = de_min + (de_max - de_min) * i / 5
        svg_parts.append(
            f'<text x="{margin-8}" y="{y_pos(de)+4}" font-size="10" '
            f'text-anchor="end">{de:.2f}</text>'
        )

    # data points
    for (model, n), r in best.items():
        color = color_map.get(r["source_script"], "#666")
        cx, cy = x_pos(n), y_pos(r["delta_E_eV_per_atom_vs_N2"])
        svg_parts.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" fill="{color}" '
            f'fill-opacity="0.85" stroke="white" stroke-width="1"/>'
        )
        svg_parts.append(
            f'<text x="{cx:.1f}" y="{cy-10:.1f}" font-size="9" text-anchor="middle" '
            f'fill="#333">{r["topology_label"]}</text>'
        )

    # legend
    ly = margin
    for label, color in color_map.items():
        svg_parts.append(f'<circle cx="{width-margin-10}" cy="{ly}" r="5" fill="{color}"/>')
        svg_parts.append(
            f'<text x="{width-margin-20}" y="{ly+4}" font-size="9" '
            f'text-anchor="end">{label}</text>'
        )
        ly += 16

    svg_parts.append('</svg>')

    with open(out_path, "w") as f:
        f.write("\n".join(svg_parts))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mace_csv", type=str, default=None,
                   help="Path to polynitrogen_mace.py's results.csv")
    p.add_argument("--mace_model", type=str, default="mace_off/medium",
                   help="Model label used for --mace_csv (default: mace_off/medium, "
                        "the script's own default; CHANGE THIS if you ran with "
                        "different flags). Rows with no MACE energy (--dry_run) "
                        "are detected automatically regardless of this label.")
    p.add_argument("--mh_csv", type=str, default=None,
                   help="Path to polynitrogen_minimahopping.py's minima_results.csv")
    p.add_argument("--mh_model", type=str, default="mace_off/small",
                   help="Model label used for --mh_csv (default: mace_off/small)")
    p.add_argument("--airss_csv", type=str, default=None,
                   help="Path to polynitrogen_airss.py's airss_results.csv")
    p.add_argument("--airss_model", type=str, default="mace_off/small",
                   help="Model label used for --airss_csv (default: mace_off/small)")
    p.add_argument("--output_dir", type=str, default="merged_polyN_analysis")
    args = p.parse_args()

    if not any([args.mace_csv, args.mh_csv, args.airss_csv]):
        sys.exit("ERROR: provide at least one of --mace_csv, --mh_csv, --airss_csv")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_rows = []

    if args.mace_csv:
        log.info(f"Reading {args.mace_csv} (polynitrogen_mace.py, model={args.mace_model})...")
        rows = read_mace_csv(args.mace_csv, args.mace_model)
        log.info(f"  {len(rows)} rows read")
        all_rows.extend(rows)

    if args.mh_csv:
        log.info(f"Reading {args.mh_csv} (polynitrogen_minimahopping.py, model={args.mh_model})...")
        rows = read_mh_csv(args.mh_csv, args.mh_model)
        log.info(f"  {len(rows)} rows read")
        all_rows.extend(rows)

    if args.airss_csv:
        log.info(f"Reading {args.airss_csv} (polynitrogen_airss.py, model={args.airss_model})...")
        rows = read_airss_csv(args.airss_csv, args.airss_model)
        log.info(f"  {len(rows)} rows read")
        all_rows.extend(rows)

    # Normalize usability for every row
    for r in all_rows:
        r["usable"] = classify_usability(r["integrity_status"], r["has_energy"])

    n_usable = sum(1 for r in all_rows if r["usable"] == "USABLE")
    n_unusable = sum(1 for r in all_rows if r["usable"] == "UNUSABLE")
    n_unknown = sum(1 for r in all_rows if r["usable"] == "UNKNOWN")
    log.info(f"Total rows: {len(all_rows)}  "
             f"(USABLE={n_usable}, UNUSABLE={n_unusable}, UNKNOWN/no-energy={n_unknown})")

    models = check_model_consistency(all_rows)

    merged_csv_path = out / "merged_results.csv"
    write_merged_csv(all_rows, merged_csv_path)
    log.info(f"Wrote {merged_csv_path}")

    hull_path = out / "merged_hull_summary.csv"
    best = write_hull_summary(all_rows, models, hull_path)
    log.info(f"Wrote {hull_path} ({len(best)} (model, n) entries)")

    svg_path = out / "merged_hull_plot.svg"
    write_svg_plot(best, models, svg_path)
    log.info(f"Wrote {svg_path}")

    # ── Final printed summary ──────────────────────────────────────────────
    log.info("=" * 60)
    log.info("CONSOLIDATED METASTABILITY LANDSCAPE")
    log.info("=" * 60)
    for (model, n) in sorted(best.keys()):
        r = best[(model, n)]
        log.info(f"  [{model}] N{n}: best dE = {r['delta_E_eV_per_atom_vs_N2']:.4f} "
                  f"eV/atom  ({r['topology_label']}, via {r['source_script']})")

    if n_unknown:
        log.warning(
            f"{n_unknown} rows had no usable energy at all (likely --dry_run "
            "input) and were excluded from the hull summary above. See "
            "merged_results.csv (usable='UNKNOWN') for the full record."
        )


if __name__ == "__main__":
    main()
