#!/usr/bin/env python3
"""
plot_polyN_pes.py
===================
Produce a publication-style plot of the polynitrogen metastability
landscape ("PES vs n" / convex-hull-style view) from the consolidated
output of merge_polyN_results.py.

WHAT IS PLOTTED
-----------------
For every USABLE structure found (across all source scripts), a point
at (n_atoms, delta_E_per_atom_vs_N2) is drawn, colored by which script
discovered it and labeled by its topology. N2 itself is marked as the
reference at (2, 0) by construction. A lower envelope ("hull line") is
drawn connecting, for each n, the single lowest-energy isomer found --
this is the "best known metastable candidate per size" curve discussed
in the companion methodology manual (Section 7.7): NOT a rigorously
exhaustive thermodynamic convex hull (that would require provably
complete sampling), but the empirical lower bound from whatever
sampling has been done so far, which can only improve (go down) with
more trials/seeds/steps.

Within a single n, multiple isomers are shown side by side (jittered
slightly in x for readability) so that the LOCAL part of the PES --
how many distinct, energetically competitive isomers exist at a given
size -- is visible, not just the single best one per size.

USAGE
------
    python plot_polyN_pes.py --merged_csv merged_analysis/merged_results.csv \\
        --output pes_plot.png

    # Or directly from one script's raw CSV (skip the merge step):
    python plot_polyN_pes.py --raw_csv polyN_AIRSS_results/airss_results.csv \\
        --raw_source airss --output pes_plot.png

DEPENDENCIES
-------------
    pip install matplotlib numpy
    pip install adjustText   # optional but recommended: automatic
                              # collision-free label placement when many
                              # isomers cluster at similar energies (a
                              # fixed jitter alone is not always enough);
                              # the script falls back gracefully to
                              # jittered-only labels if not installed.

AUTHOR
-------
    Generated for IC2MP/E4 Mediacat -- Universite de Poitiers
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe backend, no display required
import matplotlib.pyplot as plt
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# COLOR / MARKER CONVENTIONS (consistent with merge_polyN_results.py's SVG)
# ─────────────────────────────────────────────────────────────────────────────

SOURCE_STYLE = {
    "polynitrogen_mace.py": dict(color="#2166ac", marker="s", label="Graph enumeration"),
    "polynitrogen_minimahopping.py": dict(color="#b2182b", marker="^", label="Minima Hopping"),
    "polynitrogen_airss.py": dict(color="#1a9850", marker="o", label="AIRSS-style random"),
}
DEFAULT_STYLE = dict(color="#666666", marker="x", label="Other")


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

# Defensive sanity bound applied at PLOT TIME, independent of whatever
# integrity_status the upstream generator script assigned. This is a
# deliberate second line of defense: a real production artifact was
# observed where a Minima Hopping "minimum" carried delta_E_atom ~ +31
# eV/atom (two orders of magnitude above any physically meaningful
# polynitrogen candidate) yet was still marked USABLE upstream (its
# geometric status was REARRANGED/BOND_ANOMALY, not COLLAPSED, even
# though the underlying energy was a clear MLIP extrapolation
# artifact). Rather than only fixing this at the source (which does not
# help re-plotting an EXISTING merged_results.csv produced before the
# fix), this bound is enforced here too, so the plot itself can never
# be visually dominated by a single nonsensical outlier regardless of
# which CSV or script version produced the input.
PLOT_SANITY_MIN_DELTA_E = -1.0   # eV/atom
PLOT_SANITY_MAX_DELTA_E = 5.0    # eV/atom


def _filter_sane_points(points, e_n2_per_atom):
    """Drop any point whose delta_E vs N2 falls outside a generous
    sanity window, logging how many (and which) were excluded."""
    kept, dropped = [], []
    for p in points:
        delta_e = p["e_per_atom"] - e_n2_per_atom
        if PLOT_SANITY_MIN_DELTA_E <= delta_e <= PLOT_SANITY_MAX_DELTA_E:
            kept.append(p)
        else:
            dropped.append((p, delta_e))
    if dropped:
        print(f"WARNING: excluded {len(dropped)} point(s) outside the "
              f"[{PLOT_SANITY_MIN_DELTA_E}, {PLOT_SANITY_MAX_DELTA_E}] "
              f"eV/atom sanity window (likely MLIP extrapolation "
              f"artifacts upstream):")
        for p, de in dropped:
            print(f"    N{p['n']} {p['topology']} ({p['source']}): "
                  f"delta_E = {de:.2f} eV/atom")
    return kept


def load_merged_csv(path, e_n2_per_atom):
    """
    Load merge_polyN_results.py's merged_results.csv output.

    NOTE on e_n2_per_atom: the three generator scripts only persist
    delta_E_eV_per_atom_vs_N2 (already referenced to N2) in their CSV
    output, not the absolute E(Nn)/n value. To support the
    reference="dataset_min" plotting mode (which needs to compare
    ABSOLUTE per-atom energies across structures, not pre-subtracted
    ones), this loader reconstructs e_per_atom = delta_e + e_n2_per_atom.
    This means e_n2_per_atom must be supplied EVEN when using
    reference="dataset_min" -- it is needed here purely to undo the
    pre-existing N2 subtraction baked into the input file, not to define
    the plot's zero point (that distinction is made later, in plot_pes).
    """
    points = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["usable"] != "USABLE":
                continue
            delta_e = float(r["delta_E_eV_per_atom_vs_N2"])
            points.append({
                "n": int(r["n_atoms"]),
                "e_per_atom": delta_e + e_n2_per_atom,
                "topology": r["topology_label"],
                "source": r["source_script"],
            })
    return _filter_sane_points(points, e_n2_per_atom)


def load_raw_csv(path, source_key, e_n2_per_atom):
    """
    Load one of the three individual scripts' raw CSV directly (when the
    user has not run merge_polyN_results.py first). source_key selects
    which column-name convention and which integrity-status vocabulary
    to use. See load_merged_csv() docstring for why e_n2_per_atom is
    required regardless of the chosen plotting reference.
    """
    source_map = {
        "mace": ("polynitrogen_mace.py", "topology_label",
                 {"OK", "REARRANGED", "BOND_ANOMALY"}),
        "mh": ("polynitrogen_minimahopping.py", "final_topology_label",
               {"OK", "REARRANGED", "BOND_ANOMALY"}),
        "airss": ("polynitrogen_airss.py", "topology_label", {"OK"}),
    }
    if source_key not in source_map:
        sys.exit(f"ERROR: --raw_source must be one of {list(source_map)}")
    source_label, topo_col, usable_statuses = source_map[source_key]

    points = []
    with open(path) as f:
        for r in csv.DictReader(f):
            delta_e_str = r.get("delta_E_eV_per_atom_vs_N2", "").strip()
            if not delta_e_str:
                continue  # dry-run row, no energy
            if r.get("integrity_status", "") not in usable_statuses:
                continue
            delta_e = float(delta_e_str)
            points.append({
                "n": int(r["n_atoms"]),
                "e_per_atom": delta_e + e_n2_per_atom,
                "topology": r[topo_col],
                "source": source_label,
            })
    return _filter_sane_points(points, e_n2_per_atom)


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_pes(points, output_path, title=None, annotate=True,
             figsize=(11, 7), dpi=150, reference="n2", e_n2_per_atom=None):
    """
    Plot the polynitrogen energy landscape with explicit on-hull
    (filled black circle, overlaid on the source-colored marker) vs
    above-hull (open marker, source-colored only) distinction.

    reference : str
        "n2"          -- y-axis is E(Nn)/n - E(N2)/2 (eV/atom), the
                         physically interpretable "distance from
                         complete dissociation into N2" used throughout
                         the companion manual. Requires e_n2_per_atom.
        "dataset_min" -- y-axis is E(Nn)/n - min_over_all_points(E/atom),
                         i.e. relative to whichever structure in THIS
                         dataset has the lowest energy per atom,
                         regardless of size n. This removes any
                         dependency on a possibly-uncertain external N2
                         energy value, at the cost of an axis whose
                         zero point is internal to the current dataset
                         and will shift if a lower-energy structure is
                         later added to it (see companion manual
                         addendum for the full discussion of this
                         trade-off).

    ON-HULL DEFINITION USED HERE
    -------------------------------
    A point is "on the hull" if it is the lowest-energy (on whichever
    y-axis convention is active) structure found AT ITS OWN SIZE n --
    i.e. it lies on the lower envelope across n. This is the same lower
    envelope already drawn as a dashed line in earlier versions of this
    plot; the only change here is that the point ACHIEVING the envelope
    at each n is now also given a distinct marker (filled black,
    enlarged) so it is identifiable individually, not just implied by
    the envelope line passing near it.
    """
    if not points:
        sys.exit("ERROR: no USABLE data points to plot.")

    if reference == "n2" and e_n2_per_atom is None:
        sys.exit("ERROR: reference='n2' requires --e_n2_per_atom "
                 "(eV/atom). Pass the value logged as 'E(N2)/atom' by "
                 "any of the three generator scripts, or use "
                 "--reference dataset_min to avoid needing it.")

    try:
        from adjustText import adjust_text
        HAVE_ADJUSTTEXT = True
    except ImportError:
        HAVE_ADJUSTTEXT = False

    # ── Compute the active y-axis value for every point, under the
    #    chosen reference convention ──────────────────────────────────────
    if reference == "dataset_min":
        e_min = min(p["e_per_atom"] for p in points)
        for p in points:
            p["y"] = p["e_per_atom"] - e_min
        y_label = (r"$E_{\mathrm{rel}}$ = E(N$_n$)/$n$ $-$ "
                   r"$\min$[E(N$_m$)/$m$]  (eV/atom)")
        title_suffix = ("relative to the lowest-energy structure found "
                        "in this dataset")
        ref_point_label = "Dataset minimum (this run's lowest E/atom)"
    else:  # "n2"
        for p in points:
            p["y"] = p["e_per_atom"] - e_n2_per_atom
        y_label = (r"$\Delta E_{\mathrm{atom}}$ = E(N$_n$)/$n$ $-$ "
                   r"E(N$_2$)/2  (eV/atom)")
        title_suffix = r"relative to molecular N$_2$, $\Delta E = 0$ by definition"
        ref_point_label = r"N$_2$ (reference, $\Delta E = 0$)"

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # ── Shaded "metastability candidate" band (heuristic, see manual) ──────
    ax.axhspan(0, 1.0, color="#fff3cd", alpha=0.35, zorder=0,
               label=r"Heuristic screening band (0 $\leq$ y < 1 eV/atom)")

    # ── Reference point/anchor ───────────────────────────────────────────────
    if reference == "n2":
        ax.scatter([2], [0.0], s=160, marker="*", color="black", zorder=5,
                   label=ref_point_label)

    # ── Group points by n; determine the on-hull point at each n ───────────
    by_n = {}
    for p in points:
        by_n.setdefault(p["n"], []).append(p)

    hull_n = sorted(by_n.keys())
    hull_y = []
    on_hull_points = set()  # ids (by object identity) of the per-n minimum
    for n in hull_n:
        best = min(by_n[n], key=lambda p: p["y"])
        hull_y.append(best["y"])
        on_hull_points.add(id(best))

    if reference == "n2":
        hull_n_full = [2] + hull_n
        hull_y_full = [0.0] + hull_y
    else:
        hull_n_full = hull_n
        hull_y_full = hull_y

    # ── Lower envelope line ──────────────────────────────────────────────────
    ax.plot(hull_n_full, hull_y_full, color="#444444", linewidth=1.6,
            linestyle="--", zorder=2,
            label="Lower envelope (on-hull candidate per size)")

    # ── Per-source scatter, with x-jitter sized to the LOCAL point density
    #    at each n, so isomers very close in energy remain visually
    #    separated rather than overlapping into a single blob ──────────────
    sources_present = sorted(set(p["source"] for p in points))
    text_objects = []
    hull_marker_handle = None
    above_hull_marker_handle = None

    for source in sources_present:
        style = SOURCE_STYLE.get(source, DEFAULT_STYLE)
        src_points = [p for p in points if p["source"] == source]

        grouped = {}
        for p in src_points:
            grouped.setdefault(p["n"], []).append(p)

        xs_hull, ys_hull, labels_hull = [], [], []
        xs_above, ys_above, labels_above = [], [], []
        for n, plist in grouped.items():
            k = len(plist)
            n_total_at_this_n = len(by_n[n])
            half_width = min(0.32, 0.10 * max(1, n_total_at_this_n - 1))
            offsets = (np.linspace(-half_width, half_width, k)
                       if k > 1 else [0.0])
            for p, off in zip(plist, offsets):
                if id(p) in on_hull_points:
                    xs_hull.append(n + off)
                    ys_hull.append(p["y"])
                    labels_hull.append(p["topology"])
                else:
                    xs_above.append(n + off)
                    ys_above.append(p["y"])
                    labels_above.append(p["topology"])

        # Above-hull: open marker (face color = white, edge = source color)
        if xs_above:
            h = ax.scatter(xs_above, ys_above, s=70, marker=style["marker"],
                           facecolor="white", edgecolor=style["color"],
                           linewidth=1.6, alpha=0.95, zorder=3,
                           label=f"{style['label']} (above hull)")
            above_hull_marker_handle = above_hull_marker_handle or h

        # On-hull: filled marker, source color, with a black ring overlay
        # to make hull membership unambiguous regardless of source color
        if xs_hull:
            ax.scatter(xs_hull, ys_hull, s=95, marker=style["marker"],
                       color=style["color"], edgecolor="black",
                       linewidth=1.8, alpha=1.0, zorder=4,
                       label=f"{style['label']} (on hull)")

        if annotate:
            for x, y, lab in zip(xs_hull + xs_above, ys_hull + ys_above,
                                  labels_hull + labels_above):
                t = ax.annotate(lab, (x, y), fontsize=7.5, color="#333333",
                                 ha="center", va="bottom")
                text_objects.append(t)

    # ── Collision-free label placement ───────────────────────────────────────
    if annotate and text_objects:
        if HAVE_ADJUSTTEXT:
            adjust_text(
                text_objects, ax=ax,
                arrowprops=dict(arrowstyle="-", color="#999999", lw=0.6),
                expand_points=(1.4, 1.6),
                force_text=(0.3, 0.5),
            )

    # ── Axis formatting ──────────────────────────────────────────────────────
    anchor_n = [2] if reference == "n2" else []
    all_n = sorted(set(anchor_n + list(by_n.keys())))
    ax.set_xticks(all_n)
    ax.set_xlabel("Number of nitrogen atoms, $n$", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(
        title or f"Polynitrogen metastability landscape\n({title_suffix})",
        fontsize=13
    )
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5, zorder=1)
    ax.grid(True, alpha=0.25, linestyle=":")

    # Custom legend: collapse the many per-source "(above hull)"/"(on hull)"
    # entries down to one illustrative pair plus the per-source color key,
    # so the legend stays compact regardless of how many sources/sizes
    # are present.
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor="#444444", markeredgewidth=1.6,
               markersize=8, label="Above hull (open marker)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#444444",
               markeredgecolor="black", markeredgewidth=1.8,
               markersize=9, label="On hull (filled, black ring)"),
        Line2D([0], [0], color="#444444", linewidth=1.6, linestyle="--",
               label="Lower envelope"),
    ]
    if reference == "n2":
        legend_handles.append(
            Line2D([0], [0], marker="*", color="w", markerfacecolor="black",
                   markeredgecolor="black", markersize=12, label=ref_point_label)
        )
    for source in sources_present:
        style = SOURCE_STYLE.get(source, DEFAULT_STYLE)
        legend_handles.append(
            Line2D([0], [0], marker=style["marker"], color="w",
                   markerfacecolor=style["color"], markeredgecolor=style["color"],
                   markersize=8, label=style["label"])
        )

    ax.legend(handles=legend_handles, loc="center left",
              bbox_to_anchor=(1.01, 0.5), fontsize=9,
              framealpha=0.95, ncol=1, borderaxespad=0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"Wrote {output_path}")

    svg_path = Path(output_path).with_suffix(".svg")
    fig.savefig(svg_path, bbox_inches="tight")
    print(f"Wrote {svg_path}")

    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--merged_csv", type=str, default=None,
                   help="Path to merge_polyN_results.py's merged_results.csv")
    p.add_argument("--raw_csv", type=str, default=None,
                   help="Path to one script's raw results CSV (use with --raw_source)")
    p.add_argument("--raw_source", type=str, default=None,
                   choices=["mace", "mh", "airss"],
                   help="Which script produced --raw_csv")
    p.add_argument("--output", type=str, default="polyN_pes_plot.png")
    p.add_argument("--title", type=str, default=None)
    p.add_argument("--no_annotate", action="store_true",
                   help="Disable topology-label annotations on each point")
    p.add_argument("--reference", type=str, default="n2",
                   choices=["n2", "dataset_min"],
                   help="Energy reference convention for the y-axis. "
                        "'n2' (default): y = E(Nn)/n - E(N2)/2, the "
                        "physically interpretable distance from complete "
                        "dissociation, used throughout the companion "
                        "manual. 'dataset_min': y = E(Nn)/n - "
                        "min(E/atom found in this dataset), which removes "
                        "any dependency on the external N2 reference "
                        "value -- useful if that value is itself "
                        "uncertain or unavailable -- at the cost of a "
                        "zero point that is internal to (and will shift "
                        "with) the current dataset. See module docstring "
                        "of plot_pes() for the full trade-off discussion.")
    p.add_argument("--e_n2_per_atom", type=float, default=None,
                   help="E(N2)/2 in eV, as logged by any of the three "
                        "generator scripts (line 'E(N2)/atom = ...'). "
                        "REQUIRED for both --reference modes: even "
                        "'dataset_min' needs this to undo the N2 "
                        "subtraction already baked into the input CSV "
                        "before recomputing energies relative to the "
                        "dataset's own minimum.")
    args = p.parse_args()

    if args.e_n2_per_atom is None:
        sys.exit(
            "ERROR: --e_n2_per_atom is required (e.g. --e_n2_per_atom "
            "-1490.489442), regardless of --reference mode. Find this "
            "value in the log output of whichever generator script "
            "produced your data (line 'E(N2)/atom = ...')."
        )

    if args.merged_csv:
        points = load_merged_csv(args.merged_csv, args.e_n2_per_atom)
    elif args.raw_csv:
        if not args.raw_source:
            sys.exit("ERROR: --raw_source is required when using --raw_csv")
        points = load_raw_csv(args.raw_csv, args.raw_source, args.e_n2_per_atom)
    else:
        sys.exit("ERROR: provide either --merged_csv or --raw_csv/--raw_source")

    print(f"Loaded {len(points)} USABLE structures across "
          f"{len(set(p['n'] for p in points))} sizes.")

    plot_pes(points, args.output, title=args.title,
              annotate=not args.no_annotate,
              reference=args.reference, e_n2_per_atom=args.e_n2_per_atom)


if __name__ == "__main__":
    main()
