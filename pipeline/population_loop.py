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

Évaluation parallèle (n_workers, run_campaign et run_campaign_mutation) :
l'évaluation de chaque candidat retenu au sein d'une génération est
indépendante des autres (l'archive n'est mise à jour, et le surrogate
ré-entraîné, qu'après tout le lot) -- embarrassingly parallel par
construction. n_workers > 1 exécute ces évaluations sur un
ProcessPoolExecutor plutôt qu'en série ; voir _evaluate_batch.
"""

from __future__ import annotations

import functools
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import networkx as nx

from ..archive.archive import Archive, ArchiveEntry
from ..core.graph_descriptors import compute_descriptors
from ..surrogate.adaptive import AdaptiveSurrogate


def _pin_single_thread():
    """Initializer d'un worker ProcessPoolExecutor : force chaque process a
    n'utiliser QU'UN SEUL thread pour ses calculs numeriques (OpenMP/BLAS).

    Sans ca, avec n_workers process xTB/tblite lances en parallele sur une
    machine a n_workers coeurs, chaque process essaie par defaut d'utiliser
    TOUS les coeurs disponibles pour ses propres threads OpenMP internes --
    sursouscription massive (n_workers^2 threads en concurrence sur
    n_workers coeurs). Cause racine identifiee empiriquement sur la
    campagne N15- anion (job SLURM 59926, 32 workers) : ~30 candidats
    evalues en 13h30 au lieu des quelques minutes attendues (mesure
    test_timing_n15.py), avec des pas LBFGS individuels a 40-50s pour un
    systeme a 15 atomes qui devrait prendre une fraction de seconde.
    Chaque process n'ayant besoin que d'UN coeur (l'evaluation d'un
    candidat n'est elle-meme pas parallelisee), fixer explicitement les
    variables d'environnement de threading a 1 AVANT tout import de
    tblite/numpy dans le sous-process elimine la contention.

    IMPORTANT -- ceci ne suffit PAS a lui seul avec un ProcessPoolExecutor
    en mode "fork" (methode de demarrage par defaut sous Linux) : le
    process parent a deja importe tblite (xtb_bridge.py, au niveau module)
    AVANT que le pool ne cree ses workers, donc le runtime OpenMP est deja
    initialise -- et son nombre de threads deja fige -- dans le parent au
    moment du fork. Les workers forkes heritent de cet etat deja initialise
    ; positionner les variables d'environnement APRES le fork (ici) arrive
    trop tard pour changer un pool de threads OpenMP deja cree. Verifie
    empiriquement : avec seulement cet initializer et un pool "fork", les 4
    workers d'un test a 4 coeurs consommaient chacun ~1200% CPU (~12
    coeurs) au lieu de 100%. D'ou l'usage de mp_context=spawn dans
    _evaluate_batch (cf. ci-dessous) : un worker "spawn" est un interprete
    Python entierement neuf qui n'a PAS encore importe tblite -- ces
    variables d'environnement, deja en place avant le premier import de
    tblite dans ce process neuf, sont bien prises en compte."""
    import os
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"


def _evaluate_batch(to_evaluate, control_idx, candidates, evaluate_fn, charge,
                     slako_dir, eval_dir, n_workers, progress_label=""):
    """
    Evalue chaque candidat d'indice dans `to_evaluate`, en parallele si
    n_workers > 1 -- chaque evaluation (embed 3D -> relaxation -> frequences)
    est independante des autres au sein d'une meme generation (l'archive
    n'est mise a jour et le surrogate re-entraine qu'APRES tout le lot,
    cf. run_campaign), ecrit dans son propre cand_work_dir, ne touche a
    aucun etat partage : cas d'ecole d'"embarrassingly parallel".

    Retourne la liste (i, is_control, result) triee par indice croissant
    (pas par ordre de completion), pour un comportement -- et une
    reproductibilite a seed fixe -- identiques a la version serie : seul
    l'ORDRE DE SOUMISSION compte pour le dedoublonnage de l'archive
    (Archive.add), jamais l'ordre de retour des workers.

    n_workers > 1 exige que `evaluate_fn` soit picklable -- une fonction de
    niveau module, ou un functools.partial d'une telle fonction (cas de
    l'evaluateur par defaut, cf. run_campaign) ; une closure/lambda locale
    echoue avec un PicklingError explicite plutot que de degrader
    silencieusement en execution serie.

    progress_label : prefixe affiche sur chaque ligne de progression (ex.
    "gen 1") -- lecture des etapes en temps reel plutot qu'un silence total
    jusqu'a la fin du lot complet (defaut precedent, qui rendait un ralenti-
    ssement anormal indiscernable d'un blocage sans aller inspecter les
    horodatages des fichiers sur disque).
    """
    tasks = [(i, i in control_idx, str(eval_dir / f"cand{i:05d}")) for i in sorted(to_evaluate)]
    n_tasks = len(tasks)
    t_batch_start = time.monotonic()

    if n_workers <= 1:
        results = []
        for k, (i, is_ctrl, wd) in enumerate(tasks, start=1):
            t0 = time.monotonic()
            result = evaluate_fn(candidates[i], charge, slako_dir, wd)
            dt = time.monotonic() - t0
            status = result.get("integrity_status", "?") if not result.get("error") else f"REJETE ({result['error']})"
            print(f"[{progress_label}] {k}/{n_tasks} evalue -- cand{i:05d} : "
                  f"{status} ({dt:.1f}s, cumul {time.monotonic()-t_batch_start:.0f}s)", flush=True)
            results.append(result)
        return [(i, is_ctrl, result) for (i, is_ctrl, wd), result in zip(tasks, results)]

    import multiprocessing as mp
    spawn_ctx = mp.get_context("spawn")

    results_by_idx = {}
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=spawn_ctx,
                              initializer=_pin_single_thread) as pool:
        future_to_task = {
            pool.submit(evaluate_fn, candidates[i], charge, slako_dir, wd): (i, is_ctrl)
            for i, is_ctrl, wd in tasks
        }
        n_done = 0
        for fut in as_completed(future_to_task):
            i, is_ctrl = future_to_task[fut]
            result = fut.result()
            results_by_idx[i] = result
            n_done += 1
            status = result.get("integrity_status", "?") if not result.get("error") else f"REJETE ({result['error']})"
            print(f"[{progress_label}] {n_done}/{n_tasks} evalues -- cand{i:05d} : "
                  f"{status} (cumul {time.monotonic()-t_batch_start:.0f}s, "
                  f"{n_workers} workers)", flush=True)

    return [(i, is_ctrl, results_by_idx[i]) for i, is_ctrl, wd in tasks]


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
    n_workers: int = 1,
    checkpoint_dir: str | Path | None = None,
) -> tuple[Archive, list[GenerationReport]]:
    """
    n_workers : nombre de candidats évalués EN PARALLÈLE au sein d'une même
        génération (ProcessPoolExecutor, un process par évaluation
        concurrente). 1 = série (comportement historique, inchangé).
        Embarrassingly parallel par construction (cf. _evaluate_batch) :
        gain quasi-linéaire avec le nombre de cœurs alloués, sans toucher à
        la logique de filtrage/archive. Avec evaluate_fn=None (défaut),
        fonctionne tel quel ; un evaluate_fn personnalisé doit être une
        fonction de niveau module (ou un functools.partial d'une telle
        fonction) pour rester picklable au-delà de n_workers=1.
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
    checkpoint_dir : si fourni, un point de sauvegarde (archive + reports +
        dernière génération complétée) est écrit sur disque (écriture
        atomique) juste APRÈS que chaque génération ait fini (candidats
        évalués, archive élaguée, surrogate ré-entraîné) -- granularité PAR
        GÉNÉRATION, pas par candidat individuel (cf. docstring de
        pipeline.checkpoint pour la justification). Si un checkpoint existe
        déjà à cet emplacement au démarrage, les générations déjà
        complétées sont sautées (source_fn correspondant jamais appelé) et
        la campagne reprend juste après -- permet de relancer une campagne
        interrompue (timeout SLURM, scancel, panne) sans repartir de zéro.
    """
    import random
    rng_ctrl = random.Random(seed)

    if evaluate_fn is None:
        from ..evaluation.xtb_bridge import default_multiseed_evaluate_fn

        # functools.partial d'une fonction de niveau module (pas une closure
        # locale) : reste picklable pour ProcessPoolExecutor quand
        # n_workers > 1 -- une closure imbriquée (def evaluate_fn(...): ...)
        # echouerait a la soumission (PicklingError), le pickle standard ne
        # sachant serialiser que des fonctions accessibles par nom de module.
        evaluate_fn = functools.partial(
            default_multiseed_evaluate_fn,
            seeds=multiseed_seeds, run_frequencies=run_frequencies,
        )

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    archive = Archive(window_ev_per_atom=window_ev_per_atom)
    surrogate = AdaptiveSurrogate(window_ev_per_atom=window_ev_per_atom)
    reports: list[GenerationReport] = []
    start_gen = 1

    if checkpoint_dir is not None:
        from .checkpoint import checkpoint_path, load_checkpoint, save_checkpoint
        ckpt_file = checkpoint_path(checkpoint_dir)
        loaded = load_checkpoint(ckpt_file)
        if loaded is not None:
            archive, report_dicts, last_completed = loaded
            reports = [GenerationReport(**d) for d in report_dicts]
            surrogate.fit_from_archive(archive)
            start_gen = last_completed + 1
            print(f"[reprise] checkpoint trouve ({ckpt_file}) : {last_completed} generation(s) "
                  f"deja completee(s), archive={len(archive)} structures. Reprise a la generation "
                  f"{start_gen}.", flush=True)

    for gen_idx, (source_fn, budget) in enumerate(zip(candidate_source_per_generation, population_budgets), start=1):
        if gen_idx < start_gen:
            continue
        t_gen_start = time.monotonic()
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

        print(f"\n=== Generation {gen_idx}/{len(population_budgets)} : {n_seen} candidats vus, "
              f"{len(to_evaluate)} a evaluer ({len(selected_idx)} selectionnes + {len(control_idx)} "
              f"controle), archive actuelle={len(archive)} ===", flush=True)

        n_ok = 0
        n_new_unique = 0
        n_duplicate_updated = 0
        n_duplicate_rejected = 0
        n_rearranged = 0
        control_energies: list[float] = []  # correction #2 : énergies des contrôles évalués OK
        eval_dir = work_dir / f"gen{gen_idx:03d}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        for i, is_control, result in _evaluate_batch(
            to_evaluate, control_idx, candidates, evaluate_fn, charge,
            slako_dir, eval_dir, n_workers, progress_label=f"gen {gen_idx}",
        ):
            G = candidates[i]
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

        gen_elapsed = time.monotonic() - t_gen_start
        print(f"=== Generation {gen_idx} terminee en {gen_elapsed/60:.1f} min : "
              f"{n_ok}/{len(to_evaluate)} evalues OK, {n_new_unique} nouvelles structures, "
              f"archive={len(archive)}, meilleure energie={archive.best_energy_per_atom} eV/atome ===",
              flush=True)

        if checkpoint_dir is not None:
            save_checkpoint(ckpt_file, archive, reports, last_completed_generation=gen_idx)
            print(f"[checkpoint] sauvegarde -> {ckpt_file} (generation {gen_idx} completee)", flush=True)

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
    n_workers: int = 1,
    checkpoint_dir: str | Path | None = None,
) -> tuple[Archive, list[GenerationReport]]:
    """
    n_workers : voir run_campaign -- même mécanisme (ProcessPoolExecutor,
        embarrassingly parallel par génération), même contrainte de
        picklabilité sur evaluate_fn au-delà de n_workers=1.

    checkpoint_dir : voir run_campaign -- même mécanisme (sauvegarde après
        chaque génération complète, reprise automatique si un checkpoint
        existe déjà à cet emplacement). La resynchronisation de la source
        avec l'archive (source.set_archive, ci-dessous) reste correcte
        après reprise car elle est refaite à CHAQUE génération à partir de
        l'archive rechargée, jamais mise en cache entre générations.

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

    from ..evaluation.xtb_bridge import default_multiseed_evaluate_fn

    # cf. run_campaign : functools.partial d'une fonction de niveau module,
    # picklable pour ProcessPoolExecutor quand n_workers > 1 (une closure
    # locale ne le serait pas).
    evaluate_fn = functools.partial(
        default_multiseed_evaluate_fn,
        seeds=multiseed_seeds, run_frequencies=run_frequencies,
    )

    import random
    rng_ctrl = random.Random(seed)

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    archive = Archive(window_ev_per_atom=window_ev_per_atom)
    surrogate = AdaptiveSurrogate(window_ev_per_atom=window_ev_per_atom)
    reports: list[GenerationReport] = []
    start_gen = 1

    if checkpoint_dir is not None:
        from .checkpoint import checkpoint_path, load_checkpoint, save_checkpoint
        ckpt_file = checkpoint_path(checkpoint_dir)
        loaded = load_checkpoint(ckpt_file)
        if loaded is not None:
            archive, report_dicts, last_completed = loaded
            reports = [GenerationReport(**d) for d in report_dicts]
            surrogate.fit_from_archive(archive)
            start_gen = last_completed + 1
            print(f"[reprise] checkpoint trouve ({ckpt_file}) : {last_completed} generation(s) "
                  f"deja completee(s), archive={len(archive)} structures. Reprise a la generation "
                  f"{start_gen}.", flush=True)

    for gen_idx in range(1, n_generations + 1):
        budget = population_budgets[min(gen_idx - 1, len(population_budgets) - 1)]

        # Resynchroniser la source avec l'archive courante AVANT de générer
        source.set_archive(
            [e.final_graph for e in archive.entries],
            [e.energy_ev_per_atom for e in archive.entries],
        )

        if gen_idx < start_gen:
            continue
        t_gen_start = time.monotonic()

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

        print(f"\n=== Generation {gen_idx}/{n_generations} : {n_seen} candidats vus, "
              f"{len(to_evaluate)} a evaluer ({len(selected_idx)} selectionnes + {len(control_idx)} "
              f"controle), archive actuelle={len(archive)} ===", flush=True)

        n_ok = n_new_unique = n_duplicate_updated = n_duplicate_rejected = n_rearranged = 0
        control_energies: list[float] = []  # correction #2
        eval_dir = work_dir / f"gen{gen_idx:03d}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        for i, is_control, result in _evaluate_batch(
            to_evaluate, control_idx, candidates, evaluate_fn, charge,
            slako_dir, eval_dir, n_workers, progress_label=f"gen {gen_idx}",
        ):
            G = candidates[i]
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

        gen_elapsed = time.monotonic() - t_gen_start
        print(f"=== Generation {gen_idx} terminee en {gen_elapsed/60:.1f} min : "
              f"{n_ok}/{len(to_evaluate)} evalues OK, {n_new_unique} nouvelles structures, "
              f"archive={len(archive)}, meilleure energie={archive.best_energy_per_atom} eV/atome ===",
              flush=True)

        if checkpoint_dir is not None:
            save_checkpoint(ckpt_file, archive, reports, last_completed_generation=gen_idx)
            print(f"[checkpoint] sauvegarde -> {ckpt_file} (generation {gen_idx} completee)", flush=True)

    return archive, reports
