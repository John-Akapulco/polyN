"""
Génération d'entrées Gaussian (.com) chaînées via Link1 : toutes les
étapes d'un même candidat partagent un seul fichier et un seul checkpoint
(%chk) -- Gaussian relit la géométrie/fonction d'onde de l'étape précédente
via `Geom=Check Guess=Read`, donc aucun parsing de sortie intermédiaire
n'est nécessaire ici.
"""

from __future__ import annotations

from ..config import CalculationStep, Resources
from ..xyz_io import Atom

_JOB_KEYWORD = {
    "geometry_optimization": "Opt",
    "frequency": "Freq",
    "single_point": "",
}


def render_gaussian_input(
    steps: list[CalculationStep],
    atoms: list[Atom],
    charge: int,
    multiplicity: int,
    resources: Resources,
    chk_name: str,
    title_prefix: str = "",
) -> str:
    """Un seul fichier .com avec une section --Link1-- par étape."""
    blocks = []
    for i, step in enumerate(steps):
        route_tokens = [f"{step.method}/{step.basis}"]
        job_kw = _JOB_KEYWORD[step.job_type]
        if job_kw:
            route_tokens.append(job_kw)
        route_tokens.extend(step.extra_keywords)
        if i > 0:
            route_tokens += ["Geom=Check", "Guess=Read"]
        route = "# " + " ".join(route_tokens)

        mult = step.multiplicity_override or multiplicity
        section = [
            f"%chk={chk_name}.chk",
            f"%nprocshared={resources.n_proc}",
            f"%mem={resources.memory_mb}MB",
            route,
            "",
            f"{title_prefix}step {i + 1}: {step.name}",
            "",
            f"{charge} {mult}",
        ]
        if i == 0:
            # Geometrie de depart -- les etapes suivantes la relisent depuis
            # le checkpoint (Geom=Check), pas besoin de la repeter.
            for sym, x, y, z in atoms:
                section.append(f"{sym:2s} {x:14.8f} {y:14.8f} {z:14.8f}")
        section.append("")  # ligne vide terminant la section
        blocks.append("\n".join(section))

    return "\n--Link1--\n".join(blocks) + "\n"
