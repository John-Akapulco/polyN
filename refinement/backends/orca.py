"""
Génération d'entrées ORCA (.inp), un fichier par étape. Le chaînage entre
étapes se fait via `* xyzfile` pointant vers le `.xyz` optimisé qu'ORCA
écrit automatiquement à l'issue d'une étape Opt (convention ORCA :
`<basename_input>.xyz`) -- pas de parsing de sortie nécessaire ici non
plus.
"""

from __future__ import annotations

from ..config import CalculationStep, Resources
from ..xyz_io import Atom

_JOB_KEYWORD = {
    "geometry_optimization": "Opt",
    "frequency": "Freq",
    "single_point": "",
}


def render_orca_input(
    step: CalculationStep,
    atoms: list[Atom] | None,
    charge: int,
    multiplicity: int,
    resources: Resources,
    previous_xyz_path: str | None = None,
) -> str:
    """`atoms` est requis si `previous_xyz_path` est None (première étape,
    géométrie issue de l'archive) ; ignoré sinon (l'étape relit la
    géométrie de l'étape précédente)."""
    mult = step.multiplicity_override or multiplicity
    keyword_tokens = [step.method, step.basis]
    job_kw = _JOB_KEYWORD[step.job_type]
    if job_kw:
        keyword_tokens.append(job_kw)
    if step.aux_basis:
        keyword_tokens.append(step.aux_basis)
    keyword_tokens.extend(step.extra_keywords)

    per_core_mem = resources.memory_mb // max(resources.n_proc, 1)
    lines = [
        "! " + " ".join(keyword_tokens),
        f"%pal nprocs {resources.n_proc} end",
        f"%maxcore {per_core_mem}",
    ]
    if previous_xyz_path is not None:
        lines.append(f"* xyzfile {charge} {mult} {previous_xyz_path}")
    else:
        if atoms is None:
            raise ValueError(
                f"etape '{step.name}': geometrie de depart manquante "
                f"(atoms=None et previous_xyz_path=None)"
            )
        lines.append(f"* xyz {charge} {mult}")
        for sym, x, y, z in atoms:
            lines.append(f"{sym:2s} {x:14.8f} {y:14.8f} {z:14.8f}")
        lines.append("*")

    return "\n".join(lines) + "\n"
