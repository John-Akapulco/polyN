"""
Orchestrateur : consomme la file de priorité produite par
archive.Archive.export_refinement_queue() et écrit, pour chaque candidat
retenu, les fichiers d'entrée Gaussian ou ORCA correspondant à la séquence
d'étapes définie dans la configuration YAML (config.RefinementConfig).
Aucun paramètre de calcul n'est codé en dur dans ce module : tout vient de
la config.

Portée délibérément limitée : ce module PRÉPARE des jobs (fichiers
d'entrée + script d'exécution local) ; il n'intègre pas de scheduler de
cluster (SLURM/PBS) -- le script `run.sh` généré est un script shell nu,
utilisable tel quel en local ou comme corps d'un script de soumission que
l'utilisateur enveloppe lui-même selon son cluster.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..archive.archive import guess_multiplicity
from .backends.gaussian import render_gaussian_input
from .backends.orca import render_orca_input
from .config import RefinementConfig
from .xyz_io import read_xyz_atoms


@dataclass
class RefinementJob:
    candidate_id: str
    charge: int
    multiplicity: int
    input_paths: list[Path]  # un seul .com (Gaussian) ou un par etape (ORCA)
    run_script: Path


def _candidate_id(entry: dict) -> str:
    xyz = entry.get("optimized_xyz_path")
    if xyz:
        return Path(xyz).stem
    return entry.get("wl_hash") or "candidate"


def _select_applicable_steps(steps, n_atoms: int):
    """Filtre les étapes selon `max_n_atoms` (ex: réserver DLPNO-CCSD(T)-F12
    aux molécules < 10 atomes) et la disponibilité de leur prérequis
    (`depends_on`) : une étape dont le prérequis a été écartée pour ce
    candidat est écartée aussi, en cascade -- le chaînage étant strictement
    linéaire (cf. config.RefinementConfig), ceci écarte simplement tout
    suffixe de la séquence à partir du premier dépassement de taille.
    Retourne (etapes_applicables, etapes_ecartees)."""
    applicable, skipped = [], []
    available_names: set[str] = set()
    for step in steps:
        gated_out = step.max_n_atoms is not None and n_atoms > step.max_n_atoms
        prereq_missing = step.depends_on is not None and step.depends_on not in available_names
        if gated_out or prereq_missing:
            skipped.append(step)
            continue
        applicable.append(step)
        available_names.add(step.name)
    return applicable, skipped


def build_jobs_from_queue(
    queue: list[dict],
    config: RefinementConfig,
    out_dir: str | Path,
    charge: int,
    max_jobs: int | None = None,
) -> list[RefinementJob]:
    """
    queue : sortie de Archive.export_refinement_queue(charge) (déjà triée
        par priorité décroissante).
    charge : redondant avec queue[i]["charge"] si la file en dispose déjà
        (versions récentes d'export_refinement_queue) -- accepté ici
        explicitement pour rester utilisable avec une file construite à la
        main. Doit correspondre à queue[i]["charge"] quand ce champ existe.
    max_jobs : ne traite que les N candidats les plus prioritaires (le
        raffinement CCSD(T)/DFT est réservé à quelques isomères bas en
        énergie, pas au criblage haut-débit -- cf. README du package).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = queue[:max_jobs] if max_jobs is not None else queue

    jobs: list[RefinementJob] = []
    for entry in entries:
        entry_charge = entry.get("charge", charge)
        if entry_charge != charge:
            raise ValueError(
                f"incoherence de charge : parametre charge={charge} vs "
                f"entry['charge']={entry_charge} pour {entry.get('optimized_xyz_path')}"
            )

        xyz_path = entry.get("optimized_xyz_path")
        if not xyz_path:
            continue  # candidat sans geometrie sauvegardee -- rien a soumettre
        atoms = read_xyz_atoms(xyz_path)
        n_atoms = entry.get("n_atoms") or len(atoms)

        multiplicity = entry.get("multiplicity_guess")
        if multiplicity is None:
            multiplicity = guess_multiplicity(len(atoms), charge)

        applicable_steps, skipped_steps = _select_applicable_steps(config.steps, n_atoms)
        if not applicable_steps:
            continue  # aucune etape applicable pour ce candidat (taille au-dela de toute etape)

        cand_id = _candidate_id(entry)
        cand_dir = out_dir / cand_id
        cand_dir.mkdir(parents=True, exist_ok=True)

        if config.backend == "gaussian":
            input_path = cand_dir / f"{cand_id}.com"
            input_path.write_text(render_gaussian_input(
                applicable_steps, atoms, charge, multiplicity, config.resources,
                chk_name=cand_id, title_prefix=f"{cand_id} -- ",
            ))
            input_paths = [input_path]

        elif config.backend == "orca":
            input_paths = []
            previous_xyz: str | None = None
            for step in applicable_steps:
                step_atoms = atoms if previous_xyz is None else None
                content = render_orca_input(
                    step, step_atoms, charge, multiplicity, config.resources,
                    previous_xyz_path=previous_xyz,
                )
                step_input = cand_dir / f"{step.name}.inp"
                step_input.write_text(content)
                input_paths.append(step_input)
                if step.job_type == "geometry_optimization":
                    previous_xyz = f"{step.name}.xyz"  # ecrit par ORCA a la fin de l'optimisation

        else:
            raise ValueError(f"backend non gere: {config.backend}")

        if skipped_steps:
            (cand_dir / "SKIPPED_STEPS.txt").write_text(
                f"Etapes non generees pour ce candidat (n_atoms={n_atoms}) :\n"
                + "\n".join(f"- {s.name} (max_n_atoms={s.max_n_atoms})" for s in skipped_steps)
                + "\n"
            )

        run_script = cand_dir / "run.sh"
        run_script.write_text(_render_run_script(config, input_paths))
        run_script.chmod(0o755)

        jobs.append(RefinementJob(
            candidate_id=cand_id, charge=charge, multiplicity=multiplicity,
            input_paths=input_paths, run_script=run_script,
        ))

    return jobs


def _render_run_script(config: RefinementConfig, input_paths: list[Path]) -> str:
    lines = ["#!/bin/bash", "set -euo pipefail", ""]
    if config.submission.mode == "command" and config.submission.command_template:
        for p in input_paths:
            out = p.with_suffix(".out")
            lines.append(config.submission.command_template.format(input=p.name, output=out.name))
    else:
        lines += [
            "# mode dry_run : fichiers d'entree generes, aucune commande d'execution.",
            "# Definir submission.mode: command et submission.command_template",
            "# dans la config pour generer les commandes reelles.",
        ]
    return "\n".join(lines) + "\n"


def submit_jobs(jobs: list[RefinementJob], dry_run: bool = True) -> None:
    """Exécute chaque run.sh séquentiellement (exécution LOCALE, bloquante
    -- pas d'intégration à un scheduler de cluster, hors du cadre de ce
    module, cf. docstring). dry_run=True (défaut) : n'exécute rien, se
    contente d'imprimer ce qui serait lancé."""
    for job in jobs:
        if dry_run:
            print(f"[dry-run] {job.run_script}")
            continue
        subprocess.run(["bash", str(job.run_script)], cwd=job.run_script.parent, check=True)
