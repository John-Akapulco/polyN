# PolyNAdapt

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/John-Akapulco/polyN/blob/main/LICENSE)

Surrogate adaptatif pour la génération d'isomères métastables de clusters
polyazotés (Nx, charge q fixée) — neutre, mono/di-cation, mono/di-anion...

## Objectif

Pour une composition Nx et une charge q fixées, peupler un ensemble
d'isomères situés dans une fenêtre d'énergie (0.2 eV/atome par défaut,
recalculée dynamiquement par rapport au meilleur minimum connu à ce jour)
en minimisant le nombre de relaxations xTB réellement nécessaires.

Principe : à chaque population de graphes candidats (générés soit par
énumération exhaustive `geng`, soit par mutation de l'archive une fois
l'énumération devenue impraticable), un surrogate à deux volets — un
régresseur gradient boosting et une règle interprétable, tous deux
ré-entraînés à chaque génération sur l'archive accumulée — filtre les
candidats avant tout calcul coûteux. Un apprentissage actif priorise
l'évaluation réelle sur les candidats les plus incertains/proches de la
frontière de la fenêtre plutôt que sur tous les acceptés indistinctement.

## Architecture

```
polyN_adapt/
  core/
    graph_descriptors.py   # descripteurs topologiques sur graphe nu
    geng_interface.py       # streaming nauty-geng, regle de parite charge/n
  generators/
    mutation.py             # add_edge / remove_edge / double_edge_swap,
                              # n ATOMES CONSTANT -- pour n au-dela du seuil
                              # d'enumeration exhaustive praticable
  surrogate/
    adaptive.py              # AdaptiveSurrogate : regresseur + regle
                              # interpretable + selection avec apprentissage actif
  evaluation/
    xtb_bridge.py             # embed 3D -> relaxation GFN2-xTB (tblite) ->
                              # frequences -> integrite -> classification.
                              # Choix delibere de GFN2-xTB plutot que DFTB+
                              # (utilise initialement dans
                              # polynitrogen_charged_explore.py sans
                              # justification particuliere) pour rester
                              # coherent avec polyN_pipeline.py / polyN_crystal
                              # qui utilisent deja GFN2-xTB, et parce que
                              # tblite.ase.TBLite est un calculateur ASE
                              # in-process (pas de sous-processus par
                              # evaluation, contrairement au calculateur
                              # Dftb qui ecrit des fichiers et spawn dftb+).
                              # ORCA/xTB reste reserve a l'etage suivant
                              # (quelques isomeres bas en energie), pas au
                              # criblage haut-debit.
  archive/
    archive.py                 # dedoublonnage par isomorphisme, fenetre
                              # glissante recalculee sur le meilleur connu
  pipeline/
    population_loop.py          # orchestrateur generation par generation
  refinement/
    config.py                  # RefinementConfig charge depuis YAML -- backend
                                 # (gaussian/orca), sequence d'etapes (methode,
                                 # base, type de job), ressources, soumission ;
                                 # AUCUN parametre de calcul code en dur
    backends/gaussian.py         # rendu .com, etapes chainees via Link1/%chk
    backends/orca.py             # rendu .inp, un fichier par etape, chainees
                                 # via `* xyzfile` (geometrie optimisee ecrite
                                 # automatiquement par ORCA)
    submit.py                    # consomme Archive.export_refinement_queue(),
                                 # ecrit les inputs + un run.sh par candidat
    refinement_config_example.yaml
  environments/
    env-adaptive.yml
```

## Dépendance externe : polynitrogen_charged_explore.py

Ce package NE DUPLIQUE PAS `embed_graph_3d_ff`, `check_structural_integrity`,
`classify_topology` -- il les importe directement depuis votre script
existant `polynitrogen_charged_explore.py`, qui doit être présent sur le
`PYTHONPATH` :

```bash
cp /chemin/vers/polynitrogen_charged_explore.py .
# ou
export PYTHONPATH="/chemin/vers/dossier/contenant/le/script:$PYTHONPATH"
```

Le module `evaluation/xtb_bridge.py` échoue explicitement à l'import si ce
fichier est introuvable, avec un message clair plutôt qu'une erreur cryptique.

## Nouveauté ajoutée : calcul de fréquences

`polynitrogen_charged_explore.py` ne calcule aucune fréquence
vibrationnelle (seulement une intégrité géométrique par distance).
`xtb_bridge.relax_and_evaluate` ajoute ce calcul via `ase.vibrations`, pour
confirmer qu'une structure "intègre" est un vrai minimum (aucune fréquence
imaginaire significative), pas un point-selle.

## Conflit OpenMP macOS (Apple Silicon)

`xtb_bridge.py` positionne `KMP_DUPLICATE_LIB_OK=TRUE` en interne (même
précaution que `polyN_crystal/calculators/backends.py`) pour éviter un
crash au chargement quand PyTorch (conda-forge) et des paquets pip
compilés coexistent dans le même process. Si un script externe importe
`tblite`/`mace` directement AVANT d'importer ce module, préfixer la
commande : `KMP_DUPLICATE_LIB_OK=TRUE python votre_script.py`.

## Seuil énumération exhaustive vs mutation

Pas de valeur figée dans le code -- à mesurer sur la machine cible avec
`core.geng_interface.count_geng_graphs(n, max_degree=3)`. Mesuré sur cette
machine de développement (Ubuntu, conteneur) :

| n | graphes (degré≤3) | temps |
|---|---|---|
| 10 | 1 733 | 0.01 s |
| 12 | 19 430 | 0.14 s |
| 14 | 262 044 | 2.08 s |
| 16 | 4 101 318 | 37.3 s |

Praticable confortablement jusqu'à n≈14-16 sur une machine de bureau ; au-delà,
`candidate_source_per_generation` doit passer d'un générateur `geng` à un
générateur basé sur `generators.mutation` (voisinage de l'archive).

## Raffinement post-xTB (Gaussian/ORCA)

GFN2-xTB sert au criblage haut-débit, pas au classement fin des quelques
isomères bas en énergie qui survivent à la fenêtre -- ceux-là méritent une
revérification à un niveau ab initio/DFT/CCSD(T) supérieur. `refinement/`
prépare ces jobs (Gaussian ou ORCA) à partir de
`Archive.export_refinement_queue(charge)`, sans qu'aucun paramètre de
calcul (méthode, base, type de job) ne soit codé en dur -- tout vient d'un
fichier YAML (`refinement_config_example.yaml`) : backend, séquence
d'étapes (optimisation de géométrie -> fréquences -> single-point
CCSD(T), ou toute autre séquence), ressources, mode de soumission.

```python
from polyN_adapt.refinement.config import load_config
from polyN_adapt.refinement.submit import build_jobs_from_queue, submit_jobs

config = load_config("polyN_adapt/refinement/refinement_config_example.yaml")
queue = archive.export_refinement_queue(charge=0)
jobs = build_jobs_from_queue(queue, config, out_dir="refine_out", charge=0, max_jobs=10)
submit_jobs(jobs, dry_run=True)  # dry_run=False pour executer reellement (local, bloquant)
```

Les étapes sont chaînées automatiquement (Gaussian : Link1/`%chk` ; ORCA :
`* xyzfile` vers la géométrie optimisée par l'étape précédente), sans
parsing de sortie intermédiaire. La charge n'est pas stockée dans
`Archive` (une campagne est menée à charge fixée) : elle est passée
explicitement à `export_refinement_queue` et propagée jusqu'aux fichiers
d'entrée ; la multiplicité de spin par défaut est devinée par parité du
nombre d'électrons (`archive.guess_multiplicity`), toujours surchageable
par étape. Portée délibérément limitée : ce module écrit les entrées et un
script `run.sh` par candidat, il n'intègre pas de scheduler de cluster
(SLURM/PBS).

## Ce qui a été testé RÉELLEMENT dans cet environnement de développement

- `nauty-geng` réel (installé via apt) : streaming, comptage, règle de parité
- Tous les opérateurs de mutation (invariants vérifiés : n constant, degré
  respecté, connexité préservée, séquence de degrés inchangée pour le swap)
- `AdaptiveSurrogate` sur données synthétiques à relation connue (discrimine
  correctement cycle tendu vs chaîne stable) + `select_batch` actif
- `Archive` : dédoublonnage par isomorphisme + tolérance d'énergie, fenêtre
  glissante recalculée sur mise à jour de la référence
- `refinement/` : chargement + validation de config YAML (backend invalide,
  `job_type` invalide, `depends_on` non adjacent -- toutes rejetées avec un
  message clair) ; génération d'entrées Gaussian (chaînage Link1/`%chk`,
  charge/multiplicité correctement propagées) et ORCA (chaînage `*
  xyzfile`, `sp_ccsdt` référence bien la géométrie de `opt_dft` à travers
  `freq_dft`) sur une géométrie `.xyz` synthétique ; `guess_multiplicity`
  vérifié sur N4/N4⁺/N5⁺/N5⁻. **Pas de calcul Gaussian/ORCA réel lancé**
  (aucun des deux n'est installé dans cet environnement de développement)
  -- seule la génération des fichiers d'entrée est validée, pas leur
  exécution effective par le solveur.
- `embed_graph_3d_ff` (import réel depuis `polynitrogen_charged_explore.py`)
  sur un vrai graphe `geng`
- **`xtb_bridge.relax_and_evaluate` avec un VRAI calculateur GFN2-xTB
  (tblite réellement installé)** sur le pentazolate N5- : intégrité OK,
  topologie classée `ring-5` correctement, 0 fréquence imaginaire confirmée
  (vrai minimum, pas un point-selle)
- **`run_campaign` bout-en-bout avec `xtb_bridge` réel** (pas un évaluateur
  simulé) sur une mini-campagne n=6, 2 générations : le pipeline complet
  génération -> filtre -> relaxation xTB réelle -> archive -> ré-entraînement
  s'exécute sans erreur

Contrairement à la première version (DFTB+), qui n'avait pu être testée
qu'en mécanique isolée (binaire non disponible dans ce bac à sable), cette
version xTB a été validée en conditions quasi réelles de bout en bout.

**Avant un premier vrai run de production** : lancer `relax_and_evaluate`
sur quelques graphes connus supplémentaires (N7+, N4 neutre) pour élargir
la couverture de validation avant de lancer une campagne complète à grande
échelle.
