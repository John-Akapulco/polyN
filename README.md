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

## Ce qui a été testé RÉELLEMENT dans cet environnement de développement

- `nauty-geng` réel (installé via apt) : streaming, comptage, règle de parité
- Tous les opérateurs de mutation (invariants vérifiés : n constant, degré
  respecté, connexité préservée, séquence de degrés inchangée pour le swap)
- `AdaptiveSurrogate` sur données synthétiques à relation connue (discrimine
  correctement cycle tendu vs chaîne stable) + `select_batch` actif
- `Archive` : dédoublonnage par isomorphisme + tolérance d'énergie, fenêtre
  glissante recalculée sur mise à jour de la référence
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
