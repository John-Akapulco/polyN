"""
Orchestrateur de la campagne complète : pour n et charge fixés, enchaîne les
populations de taille croissante, chacune filtrée par le surrogate adaptatif
entraîné sur l'archive des populations précédentes.

Le générateur de candidats (population_source) est injecté en paramètre :
- pour n dans la plage praticable par énumération exhaustive (mesurée sur
  la machine cible, cf. discussion -- jusqu'à n≈14-16 typiquement),
  utiliser core.geng_interface.stream_geng_graphs ;
- au-delà, utiliser generators.mutation pour générer des voisins de
  l'archive plutôt que d'énumérer tout l'espace.
Cette fonction ne décide pas laquelle utiliser -- c'est un choix explicite
de l'appelant selon n et la machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import networkx as nx

from ..archive.archive import Archive, ArchiveEntry
from ..core.graph_descriptors import compute_descriptors
from ..surrogate.adaptive import AdaptiveSurrogate


@dataclass
class GenerationReport:
    generation: int
    n_candidates_seen: int
    n_selected_for_evaluation: int
    n_evaluated_ok: int
    n_rejected_by_surrogate: int
    archive_size_after: int
    best_energy_per_atom: float | None
    n_new_unique: int
    n_duplicate_updated: int
    n_duplicate_rejected: int
    n_pruned_this_generation: int
    n_rearranged: int
    # Correction #2 : instrumentation de l'échantillon de contrôle.
    # Les candidats "contrôle" sont des candidats REJETÉS par le surrogate
    # mais évalués réellement quand même (fraction control_sample_fraction).
    # false_rejection_rate = fraction des contrôles évalués avec succès qui
    # tombent DANS la fenêtre d'énergie -- donc rejetés À TORT. Un taux
    # élevé (>10%) signale un filtre trop agressif : augmenter le budget
    # ou assouplir la fenêtre du surrogate.
    n_control_evaluated: int = 0
    n_control_in_window: int = 0
    false_rejection_rate: float | None = None  # None si aucun contrôle évalué


FALSE_REJECTION_WARNING_THRESHOLD = 0.10


def run_campaign(
    n: int,
    charge: int,
    candidate_source_per_generation: list[Callable[[], Iterator[nx.Graph]]],
    slako_dir: str,
    work_dir: str,
    population_budgets: list[int],
    evaluate_fn: Callable[[nx.Graph, int, str, str], dict] | None = None,
    multiseed_seeds: tuple = (0, 1, 2),
    run_frequencies: bool = False,
    window_ev_per_atom: float = 0.2,
    active_learning: bool = True,
    control_sample_fraction: float = 0.02,
    seed: int = 0,
) -> tuple[Archive, list[GenerationReport]]:
    """
    candidate_source_per_generation : une fonction génératrice par
        population (peut être la même répétée, ou une différente à partir
        d'un certain rang -- ex: geng pour les 3 premières, mutation ensuite).
    evaluate_fn(G, charge, slako_dir, work_dir) -> dict avec au moins les clés
        'energy_ev_per_atom', 'integrity_status', 'topology_label',
        'has_imaginary_freq', 'final_graph'.

        Si None (par défaut) : bascule automatiquement sur
        evaluation.xtb_bridge.default_multiseed_evaluate_fn -- évalue chaque
        candidat sur plusieurs seeds d'embedding (multiseed_seeds) et garde
        le plus bas, plutôt qu'un seed unique. Nécessaire depuis
        l'observation empirique qu'un seed unique peut manquer un minimum
        réellement plus bas (mesuré : écart de 0.44 eV/atome entre seeds sur
        un candidat n=8 réel) -- au prix d'un budget de calcul multiplié par
        len(multiseed_seeds) par candidat évalué.

        Le 4e argument (work_dir) est désormais un dossier UNIQUE par
        candidat (génération + index), pas un chemin fixe partagé entre
        tous les candidats -- permet d'inspecter après coup les artefacts
        de calcul (fichiers ASE/xTB) propres à un candidat donné.

        'final_graph' (le graphe reconstruit géométriquement après
        relaxation) est REQUIS pour un dédoublonnage correct de l'archive
        (cf. Archive._is_duplicate) -- sans lui, deux candidats distincts
        se réarrangeant vers la même structure physique ne seront pas
        reconnus comme doublons. Un evaluate_fn personnalisé reste
        accepté (ex: évaluateur simulé pour tester l'orchestrateur
        indépendamment de xTB), tant qu'il respecte cette signature à 4
        arguments et ce contrat de retour.
    population_budgets : nombre de candidats À ÉVALUER RÉELLEMENT par
        génération (pas le nombre de candidats vus -- celui-ci peut être
        bien plus grand si le générateur est exhaustif). Avec l'évaluateur
        multiseed par défaut, chaque candidat coûte len(multiseed_seeds)
        évaluations xTB, pas une seule.
    control_sample_fraction : fraction des candidats REJETÉS par le
        surrogate, ré-évaluée réellement quand même, pour détecter si le
        filtre écarte à tort de vrais isomères métastables rares.
    """
    import random
    rng_ctrl = random.Random(seed)

    if evaluate_fn is None:
        from ..evaluation.xtb_bridge import default_multiseed_evaluate_fn

        def evaluate_fn(G, charge, slako_dir, cand_work_dir):
            return default_multiseed_evaluate_fn(
                G, charge, slako_dir, cand_work_dir,
                seeds=multiseed_seeds, run_frequencies=run_frequencies,
            )

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    archive = Archive(window_ev_per_atom=window_ev_per_atom)
    surrogate = AdaptiveSurrogate(window_ev_per_atom=window_ev_per_atom)
    reports: list[GenerationReport] = []

    for gen_idx, (source_fn, budget) in enumerate(zip(candidate_source_per_generation, population_budgets), start=1):
        candidates = []
        descriptors_cache = []
        for G in source_fn():
            candidates.append(G)
            descriptors_cache.append(compute_descriptors(G).as_dict())

        n_seen = len(candidates)
        best_known = archive.best_energy_per_atom if archive.best_energy_per_atom is not None else float("inf")

        selected_idx = set(surrogate.select_batch(
            descriptors_cache, best_known, budget=budget, active_learning=active_learning,
        ))

        # Échantillon de contrôle parmi les rejetés (non sélectionnés)
        rejected_idx = [i for i in range(n_seen) if i not in selected_idx]
        n_control = max(0, int(len(rejected_idx) * control_sample_fraction))
        control_idx = set(rng_ctrl.sample(rejected_idx, n_control)) if n_control > 0 else set()

        to_evaluate = selected_idx | control_idx
        n_rejected = n_seen - len(to_evaluate)

        n_ok = 0
        n_new_unique = 0
        n_duplicate_updated = 0
        n_duplicate_rejected = 0
        n_rearranged = 0
        control_energies: list[float] = []  # correction #2 : énergies des contrôles évalués OK
        eval_dir = work_dir / f"gen{gen_idx:03d}"
        for i in to_evaluate:
            G = candidates[i]
            is_control = i in control_idx
            cand_work_dir = str(eval_dir / f"cand{i:05d}")
            result = evaluate_fn(G, charge, slako_dir, cand_work_dir)
            if result.get("error"):
                continue
            entry = ArchiveEntry(
                candidate_graph=G,
                final_graph=result.get("final_graph"),
                descriptors=descriptors_cache[i],
                energy_ev=result["energy_ev"],
                energy_ev_per_atom=result["energy_ev_per_atom"],
                generation=gen_idx,
                integrity_status=result["integrity_status"],
                topology_label=result["topology_label"],
                has_imaginary_freq=result["has_imaginary_freq"],
                structure_rearranged=result.get("structure_rearranged"),
                final_positions=result.get("final_positions"),
                initial_positions=result.get("initial_positions"),
                n_optimization_steps=result.get("n_optimization_steps"),
                initial_xyz_path=result.get("initial_xyz_path"),
                optimized_xyz_path=result.get("optimized_xyz_path"),
                multiseed_diagnostic=result.get("multiseed_diagnostic"),
            )
            if entry.integrity_status == "OK" and not entry.has_imaginary_freq:
                if entry.structure_rearranged:
                    n_rearranged += 1
                if is_control:
                    control_energies.append(entry.energy_ev_per_atom)
                status = archive.add(entry)
                if status == "new":
                    n_new_unique += 1
                elif status == "duplicate_updated":
                    n_duplicate_updated += 1
                else:
                    n_duplicate_rejected += 1
                n_ok += 1

        n_pruned = archive.prune_outside_window(generation=gen_idx)

        # Correction #2 : taux de faux rejets, mesuré contre la référence
        # FINALE de la génération (le meilleur connu peut avoir baissé
        # pendant la boucle -- juger un contrôle contre la référence de
        # début de génération surestimerait le taux).
        n_control_in_window = 0
        false_rejection_rate = None
        if control_energies and archive.best_energy_per_atom is not None:
            n_control_in_window = sum(
                1 for e in control_energies
                if (e - archive.best_energy_per_atom) <= window_ev_per_atom
            )
            false_rejection_rate = n_control_in_window / len(control_energies)
            if false_rejection_rate > FALSE_REJECTION_WARNING_THRESHOLD:
                print(
                    f"[AVERTISSEMENT] Generation {gen_idx}: taux de faux rejets "
                    f"{false_rejection_rate:.0%} ({n_control_in_window}/{len(control_energies)} "
                    f"controles dans la fenetre) -- le surrogate rejette a tort des "
                    f"isomeres metastables. Augmenter le budget ou assouplir le filtre.",
                    flush=True,
                )

        # Correction #1 : entraînement enrichi (régresseur-filtre inchangé
        # + classificateur de réarrangement + règle sur graphes finaux)
        surrogate.fit_from_archive(archive)

        reports.append(GenerationReport(
            generation=gen_idx, n_candidates_seen=n_seen,
            n_selected_for_evaluation=len(to_evaluate), n_evaluated_ok=n_ok,
            n_rejected_by_surrogate=n_rejected, archive_size_after=len(archive),
            best_energy_per_atom=archive.best_energy_per_atom,
            n_new_unique=n_new_unique, n_duplicate_updated=n_duplicate_updated,
            n_duplicate_rejected=n_duplicate_rejected, n_pruned_this_generation=n_pruned,
            n_rearranged=n_rearranged,
            n_control_evaluated=len(control_energies),
            n_control_in_window=n_control_in_window,
            false_rejection_rate=false_rejection_rate,
        ))

    return archive, reports


def run_campaign_mutation(
    n: int,
    charge: int,
    n_generations: int,
    slako_dir: str,
    work_dir: str,
    population_budgets: list[int],
    n_candidates_per_generation: int = 2000,
    max_degree: int = 3,
    multiseed_seeds: tuple = (0, 1, 2),
    run_frequencies: bool = False,
    window_ev_per_atom: float = 0.2,
    active_learning: bool = True,
    control_sample_fraction: float = 0.02,
    seed: int = 0,
) -> tuple[Archive, list[GenerationReport]]:
    """
    Variante de run_campaign pour les GRANDES tailles (n > ~16), où
    l'énumération geng est impraticable : les candidats de chaque
    génération sont produits par MUTATION des graphes de l'archive
    courante (generators.mutation_source.MutationSource), pas par
    énumération exhaustive.

    Différence structurelle avec run_campaign : la source doit être
    resynchronisée avec l'archive ENTRE les générations (les mutants de la
    génération k+1 dérivent des survivants de la génération k) -- d'où
    cette variante dédiée plutôt qu'un simple paramétrage de run_campaign,
    dont les sources sont figées à l'avance.

    IMPORTANT -- ce que cette variante ne garantit PAS : contrairement à
    l'énumération exhaustive, rien n'assure la couverture de l'espace. La
    recherche est locale autour de l'archive ; un bassin topologique jamais
    atteint par mutation depuis les graines restera invisible. C'est le
    prix structurel du passage à l'échelle, à garder en tête pour
    l'interprétation des résultats (cf. discussion : garder un petit
    échantillon aléatoire non guidé comme contrôle est recommandé -- ici
    assuré partiellement par les graines aléatoires de la génération 1).
    """
    from ..generators.mutation_source import MutationSource

    source = MutationSource(
        n=n, max_degree=max_degree,
        n_candidates_per_generation=n_candidates_per_generation,
        seed=seed,
    )

    def evaluate_fn_factory():
        from ..evaluation.xtb_bridge import default_multiseed_evaluate_fn

        def evaluate_fn(G, charge, slako_dir, cand_work_dir):
            return default_multiseed_evaluate_fn(
                G, charge, slako_dir, cand_work_dir,
                seeds=multiseed_seeds, run_frequencies=run_frequencies,
            )
        return evaluate_fn

    evaluate_fn = evaluate_fn_factory()

    import random
    rng_ctrl = random.Random(seed)

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    archive = Archive(window_ev_per_atom=window_ev_per_atom)
    surrogate = AdaptiveSurrogate(window_ev_per_atom=window_ev_per_atom)
    reports: list[GenerationReport] = []

    for gen_idx in range(1, n_generations + 1):
        budget = population_budgets[min(gen_idx - 1, len(population_budgets) - 1)]

        # Resynchroniser la source avec l'archive courante AVANT de générer
        source.set_archive(
            [e.final_graph for e in archive.entries],
            [e.energy_ev_per_atom for e in archive.entries],
        )

        candidates = []
        descriptors_cache = []
        for G in source():
            candidates.append(G)
            descriptors_cache.append(compute_descriptors(G).as_dict())

        n_seen = len(candidates)
        best_known = archive.best_energy_per_atom if archive.best_energy_per_atom is not None else float("inf")

        selected_idx = set(surrogate.select_batch(
            descriptors_cache, best_known, budget=budget, active_learning=active_learning,
        ))
        rejected_idx = [i for i in range(n_seen) if i not in selected_idx]
        n_control = max(0, int(len(rejected_idx) * control_sample_fraction))
        control_idx = set(rng_ctrl.sample(rejected_idx, n_control)) if n_control > 0 else set()
        to_evaluate = selected_idx | control_idx
        n_rejected = n_seen - len(to_evaluate)

        n_ok = n_new_unique = n_duplicate_updated = n_duplicate_rejected = n_rearranged = 0
        control_energies: list[float] = []  # correction #2
        eval_dir = work_dir / f"gen{gen_idx:03d}"
        for i in to_evaluate:
            G = candidates[i]
            is_control = i in control_idx
            cand_work_dir = str(eval_dir / f"cand{i:05d}")
            result = evaluate_fn(G, charge, slako_dir, cand_work_dir)
            if result.get("error"):
                continue
            entry = ArchiveEntry(
                candidate_graph=G,
                final_graph=result.get("final_graph"),
                descriptors=descriptors_cache[i],
                energy_ev=result["energy_ev"],
                energy_ev_per_atom=result["energy_ev_per_atom"],
                generation=gen_idx,
                integrity_status=result["integrity_status"],
                topology_label=result["topology_label"],
                has_imaginary_freq=result["has_imaginary_freq"],
                structure_rearranged=result.get("structure_rearranged"),
                final_positions=result.get("final_positions"),
                initial_positions=result.get("initial_positions"),
                n_optimization_steps=result.get("n_optimization_steps"),
                initial_xyz_path=result.get("initial_xyz_path"),
                optimized_xyz_path=result.get("optimized_xyz_path"),
                multiseed_diagnostic=result.get("multiseed_diagnostic"),
            )
            if entry.integrity_status == "OK" and not entry.has_imaginary_freq:
                if entry.structure_rearranged:
                    n_rearranged += 1
                if is_control:
                    control_energies.append(entry.energy_ev_per_atom)
                status = archive.add(entry)
                if status == "new":
                    n_new_unique += 1
                elif status == "duplicate_updated":
                    n_duplicate_updated += 1
                else:
                    n_duplicate_rejected += 1
                n_ok += 1

        n_pruned = archive.prune_outside_window(generation=gen_idx)

        # Correction #2 (cf. run_campaign pour la justification du choix de référence)
        n_control_in_window = 0
        false_rejection_rate = None
        if control_energies and archive.best_energy_per_atom is not None:
            n_control_in_window = sum(
                1 for e in control_energies
                if (e - archive.best_energy_per_atom) <= window_ev_per_atom
            )
            false_rejection_rate = n_control_in_window / len(control_energies)
            if false_rejection_rate > FALSE_REJECTION_WARNING_THRESHOLD:
                print(
                    f"[AVERTISSEMENT] Generation {gen_idx}: taux de faux rejets "
                    f"{false_rejection_rate:.0%} ({n_control_in_window}/{len(control_energies)} "
                    f"controles dans la fenetre) -- le surrogate rejette a tort des "
                    f"isomeres metastables. Augmenter le budget ou assouplir le filtre.",
                    flush=True,
                )

        # Correction #1 : entraînement enrichi (régresseur-filtre inchangé
        # + classificateur de réarrangement + règle sur graphes finaux)
        surrogate.fit_from_archive(archive)

        reports.append(GenerationReport(
            generation=gen_idx, n_candidates_seen=n_seen,
            n_selected_for_evaluation=len(to_evaluate), n_evaluated_ok=n_ok,
            n_rejected_by_surrogate=n_rejected, archive_size_after=len(archive),
            best_energy_per_atom=archive.best_energy_per_atom,
            n_new_unique=n_new_unique, n_duplicate_updated=n_duplicate_updated,
            n_duplicate_rejected=n_duplicate_rejected, n_pruned_this_generation=n_pruned,
            n_rearranged=n_rearranged,
            n_control_evaluated=len(control_energies),
            n_control_in_window=n_control_in_window,
            false_rejection_rate=false_rejection_rate,
        ))

    return archive, reports
