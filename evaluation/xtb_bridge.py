"""
Pont vers l'évaluation réelle (embed 3D -> relaxation GFN2-xTB -> fréquences
-> intégrité -> classification), via tblite.ase.TBLite -- remplace
dftb_bridge.py (DFTB+) pour rester cohérent avec le reste de l'écosystème
(polyN_pipeline.py, polyN_crystal utilisent déjà GFN2-xTB/tblite) et pour
la performance en criblage haut-débit (calculateur in-process, pas de
sous-processus par évaluation contrairement au calculateur Dftb d'ASE).

Interface IDENTIQUE à dftb_bridge.py (même signature relax_and_evaluate,
mêmes champs EvaluationResult) : population_loop.py n'a rien à changer pour
basculer de l'un à l'autre. Le paramètre `slako_dir` est conservé dans la
signature pour cette compatibilité d'appel mais IGNORÉ ici -- GFN2-xTB ne
nécessite aucun fichier de paramètres externe (contrairement à DFTB+).

Réutilise toujours embed_graph_3d_ff / check_structural_integrity /
classify_topology de polynitrogen_charged_explore.py (dépendance externe,
non dupliquée, inchangée par ce remplacement -- seule la brique calculateur
change).
"""

from __future__ import annotations

import os

# Contournement d'un conflit OpenMP fréquent sur macOS Apple Silicon quand on
# mélange des paquets pip/conda compilés qui embarquent chacun leur propre
# runtime libomm -- même précaution que dans polyN_crystal/calculators/backends.py.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from dataclasses import dataclass
from pathlib import Path

import networkx as nx
from ase import Atoms
from ase.optimize import LBFGS
from ase.vibrations import Vibrations
from tblite.ase import TBLite

try:
    from polynitrogen_charged_explore import (
        embed_graph_3d_ff,
        check_structural_integrity,
        classify_topology,
    )
except ImportError as exc:
    raise ImportError(
        "polynitrogen_charged_explore.py introuvable sur le PYTHONPATH. "
        "Ce module en dépend explicitement (embed 3D, intégrité, "
        "classification) plutôt que de dupliquer cette logique. Copier "
        "polynitrogen_charged_explore.py à côté de ce package, ou ajouter "
        "son dossier au PYTHONPATH."
    ) from exc


IMAGINARY_FREQ_TOLERANCE_CM1 = 10.0


@dataclass
class EvaluationResult:
    energy_ev: float
    energy_ev_per_atom: float
    integrity_status: str
    topology_label: str
    n_imaginary_freq: int
    has_imaginary_freq: bool
    frequencies_cm1: list
    final_graph: "object"
    structure_rearranged: bool
    final_positions: list | None = None  # coordonnees 3D reelles apres relaxation
    initial_positions: list | None = None  # coordonnees 3D AVANT optimisation (embed_graph_3d_ff)
    n_optimization_steps: int | None = None  # nombre de pas LBFGS -- mesure du cout/efficacite
    initial_xyz_path: str | None = None  # fichier .xyz reel ecrit sur disque
    optimized_xyz_path: str | None = None  # idem, geometrie finale
    error: str | None = None


def _check_imaginary_frequencies(atoms: Atoms, calc, work_dir: Path,
                                   tol_cm1: float = IMAGINARY_FREQ_TOLERANCE_CM1):
    """Identique en logique à la version DFTB+ -- seul le calculateur passé
    en argument change (TBLite au lieu de Dftb)."""
    vib_dir = work_dir / "vib"
    vib_dir.mkdir(parents=True, exist_ok=True)
    atoms.calc = calc
    vib = Vibrations(atoms, name=str(vib_dir / "vib"))
    vib.run()
    energies = vib.get_energies()
    freqs_cm1 = [complex(e).real / 1.23981e-4 if complex(e).imag == 0
                 else -abs(complex(e).imag) / 1.23981e-4
                 for e in energies]
    n_imaginary = sum(1 for f in freqs_cm1 if f < -tol_cm1)
    vib.clean()
    return freqs_cm1, n_imaginary


def relax_and_evaluate(G, charge: int, slako_dir: str, work_dir: str,
                        n_conformers: int = 3, fmax: float = 0.05,
                        max_steps: int = 300, run_frequencies: bool = True,
                        seed: int = 0) -> EvaluationResult:
    """
    Chaîne complète pour un graphe candidat : embed 3D -> relaxation GFN2-xTB
    -> intégrité géométrique -> (si OK) fréquences -> classification.

    `slako_dir` est accepté mais IGNORÉ (conservé uniquement pour que
    population_loop.py appelle evaluate_fn(G, charge, slako_dir) sans
    modification, que le backend soit DFTB+ ou xTB).

    Multiplicité de spin toujours fixée à 1 (singulet) : conforme à la
    portée de cette campagne (charge fixée par famille, seule la parité
    n/charge compatible avec un état singulet est explorée -- cf.
    core.geng_interface.validate_n_for_charge).
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    n = G.number_of_nodes()

    try:
        coords = embed_graph_3d_ff(G, n_conformers=n_conformers, seed=seed)
        atoms = Atoms(["N"] * n, positions=coords)
        initial_positions = atoms.get_positions().tolist()

        # Sauvegarde REELLE de la geometrie initiale, avant toute optimisation --
        # necessaire pour analyser la stabilite/expliquer les isomeres et pour
        # evaluer l'efficacite du workflow (comparaison initial vs optimise).
        initial_xyz_path = work_dir / "initial.xyz"
        import ase.io
        ase.io.write(str(initial_xyz_path), atoms, format="xyz",
                     comment=f"charge={charge:+d} mult=1 label=candidate_initial")

        atoms.calc = TBLite(method="GFN2-xTB", charge=charge, multiplicity=1)
        opt = LBFGS(atoms, logfile=str(work_dir / "opt.log"))
        opt.run(fmax=fmax, steps=max_steps)
        n_steps = opt.get_number_of_steps()
        energy = atoms.get_potential_energy()

        # Sauvegarde REELLE de la geometrie optimisee -- c'est le fichier a lire
        # directement pour le rapport de structures, plutot que de repasser par
        # le JSON de l'archive.
        optimized_xyz_path = work_dir / "optimized.xyz"
        ase.io.write(str(optimized_xyz_path), atoms, format="xyz",
                     comment=f"charge={charge:+d} mult=1 label=optimized "
                             f"energy_ev={energy:.6f} n_steps={n_steps}")

        integ = check_structural_integrity(atoms.get_positions())
        if integ["status"] != "OK":
            rearranged = None  # non pertinent/non fiable si l'intégrité a échoué
            return EvaluationResult(
                energy_ev=energy, energy_ev_per_atom=energy / n,
                integrity_status=integ["status"], topology_label=classify_topology(integ["final_graph"]),
                n_imaginary_freq=-1, has_imaginary_freq=True,
                frequencies_cm1=[], final_graph=integ["final_graph"],
                structure_rearranged=rearranged,
                initial_positions=initial_positions, n_optimization_steps=n_steps,
                initial_xyz_path=str(initial_xyz_path), optimized_xyz_path=str(optimized_xyz_path),
                error=f"intégrité géométrique: {integ['status']}",
            )

        # Indicateur de réarrangement : le graphe candidat original (G, celui
        # que geng/pyxtal a proposé) et le graphe reconstruit géométriquement
        # APRÈS relaxation (integ["final_graph"]) peuvent différer -- la
        # molécule s'est réarrangée (cycle ouvert/fermé, liaison rompue/
        # formée) pendant l'optimisation. Comparaison par isomorphisme, pas
        # par égalité de descripteurs simples, pour être robuste au
        # relabeling des sommets.
        try:
            structure_rearranged = not nx.is_isomorphic(G, integ["final_graph"])
        except Exception:
            structure_rearranged = None  # comparaison impossible (tailles differentes etc.)

        n_imaginary = 0
        freqs = []
        if run_frequencies:
            calc_vib = TBLite(method="GFN2-xTB", charge=charge, multiplicity=1)
            freqs, n_imaginary = _check_imaginary_frequencies(atoms, calc_vib, work_dir)

        label = classify_topology(integ["final_graph"])
        return EvaluationResult(
            energy_ev=energy, energy_ev_per_atom=energy / n,
            integrity_status=integ["status"], topology_label=label,
            n_imaginary_freq=n_imaginary, has_imaginary_freq=(n_imaginary > 0),
            frequencies_cm1=freqs, final_graph=integ["final_graph"],
            structure_rearranged=structure_rearranged,
            final_positions=atoms.get_positions().tolist(),
            initial_positions=initial_positions, n_optimization_steps=n_steps,
            initial_xyz_path=str(initial_xyz_path), optimized_xyz_path=str(optimized_xyz_path),
            error=None,
        )
    except Exception as exc:
        return EvaluationResult(
            energy_ev=float("nan"), energy_ev_per_atom=float("nan"),
            integrity_status="ERROR", topology_label="error",
            n_imaginary_freq=-1, has_imaginary_freq=True,
            frequencies_cm1=[], final_graph=None,
            structure_rearranged=None, final_positions=None,
            initial_positions=None, n_optimization_steps=None,
            initial_xyz_path=None, optimized_xyz_path=None, error=str(exc),
        )


def relax_and_evaluate_multiseed(G, charge: int, slako_dir: str, work_dir: str,
                                    seeds: tuple = (0, 1, 2), **kwargs) -> tuple[EvaluationResult, dict]:
    """
    Évalue le même candidat sur plusieurs seeds d'embedding 3D et garde le
    résultat de plus basse énergie -- un seed unique peut manquer
    silencieusement un minimum différent et plus bas (observé empiriquement :
    sur un échantillon de 5 seeds pour un candidat n=8, 4 convergeaient vers
    un même bassin à -78.25 eV/atome, 1 tombait sur un bassin distinct et
    RÉELLEMENT plus stable à -78.69 eV/atome -- un écart de 0.44 eV/atome,
    bien au-delà du bruit numérique).

    Retourne (meilleur_résultat, diagnostic) où diagnostic contient
    l'étalement d'énergie observé entre seeds -- utile pour détecter les
    candidats particulièrement multimodaux (étalement élevé) qui mériteraient
    encore plus de seeds pour être sûr d'avoir trouvé le vrai minimum.
    """
    results = []
    for s in seeds:
        r = relax_and_evaluate(G, charge, slako_dir, work_dir=f"{work_dir}_seed{s}", seed=s, **kwargs)
        if r.error is None:
            results.append((s, r))

    if not results:
        # tous les seeds ont échoué -- retourner le dernier résultat (en erreur)
        return r, {"n_seeds_ok": 0, "energy_spread_ev_per_atom": None, "best_seed": None}

    results.sort(key=lambda sr: sr[1].energy_ev_per_atom)
    best_seed, best_result = results[0]
    energies = [r.energy_ev_per_atom for _, r in results]
    spread = max(energies) - min(energies)

    diagnostic = {
        "n_seeds_ok": len(results),
        "n_seeds_tried": len(seeds),
        "energy_spread_ev_per_atom": spread,
        "best_seed": best_seed,
        "all_energies": {s: r.energy_ev_per_atom for s, r in results},
    }
    return best_result, diagnostic


def default_multiseed_evaluate_fn(G, charge: int, slako_dir: str, work_dir: str,
                                     seeds: tuple = (0, 1, 2), run_frequencies: bool = False) -> dict:
    """
    Évaluateur par défaut pour run_campaign (population_loop.py), basé sur
    relax_and_evaluate_multiseed plutôt qu'un seed unique -- nécessaire
    depuis l'observation empirique qu'un seed unique peut manquer un minimum
    réellement plus bas (mesuré : 4/5 seeds convergent vers un bassin, 1
    tombe sur un bassin distinct 0.44 eV/atome plus stable, sur un candidat
    n=8 réel).

    Convertit le (EvaluationResult, diagnostic) de relax_and_evaluate_multiseed
    vers le format dict attendu par run_campaign, en conservant le
    diagnostic multiseed (étalement d'énergie observé, seed retenu) sous une
    clé supplémentaire -- ignorée par population_loop.py mais utile pour
    inspection a posteriori.
    """
    best, diag = relax_and_evaluate_multiseed(
        G, charge, slako_dir, work_dir, seeds=seeds, run_frequencies=run_frequencies,
    )
    return {
        "energy_ev": best.energy_ev,
        "energy_ev_per_atom": best.energy_ev_per_atom,
        "integrity_status": best.integrity_status,
        "topology_label": best.topology_label,
        "has_imaginary_freq": best.has_imaginary_freq,
        "structure_rearranged": best.structure_rearranged,
        "final_graph": best.final_graph,
        "final_positions": best.final_positions,
        "initial_positions": best.initial_positions,
        "n_optimization_steps": best.n_optimization_steps,
        "initial_xyz_path": best.initial_xyz_path,
        "optimized_xyz_path": best.optimized_xyz_path,
        "error": best.error,
        "multiseed_diagnostic": diag,
    }
