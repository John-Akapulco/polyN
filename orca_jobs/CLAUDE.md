# orca_jobs — notes de projet (lues automatiquement en début de session)

Cluster : `yargla.cluster.local`. Ce fichier consigne les faits établis et
décisions prises pendant les campagnes de calcul, pour ne pas les
redécouvrir/refaire à chaque session. Compléter au fil de l'eau, pas
réécrire l'historique — état actuel factuel uniquement.

## ORCA : chemin binaire officiel du cluster

**Toujours utiliser l'installation système**, jamais une copie locale :

```
ORCA_DIR = /opt/ohpc/pub/software/orca_6_1_0_linux_x86-64_shared_openmpi418
ORCA_BIN = ${ORCA_DIR}/orca
```

Modules requis avant lancement : `module load gnu12/12.3.0` puis
`module load openmpi4/4.1.6`. Le binaire local `/home/gilles/polyN/orca_jobs/orca`
(copie/tarball) n'a pas ses exécutables satellites MPI (`orca_startup_mpi`,
etc.) à côté de lui — ORCA les cherche au même chemin absolu que le binaire
invoqué, donc lancer cette copie casse tout calcul parallèle (`mpirun...
could not access or execute orca_startup_mpi`). Toujours invoquer via
`ORCA_DIR`, jamais un chemin local, même si un binaire y traîne.

Voir `production_dispatch.py` (`ORCA_DIR`, `ORCA_BIN`, `MODULE_CMDS`) — ce
fichier a toujours eu le bon chemin ; en cas de doute, le relire plutôt que
de re-chercher le binaire à tâtons.

## Campagne 1 — DFT WB97X-D4/aug-cc-pVTZ (193 candidats polyazotés)

- Statut (2026-09-06) : 189/193 terminés sans erreur ORCA, 4 encore en
  cours (N9_cation_cation_002/004/005/007).
- **31 structures ont buté sur la limite de cycles d'optimisation**
  (`did not converge but reached the maximum` dans le `.out`) — énergie
  non fiable pour ces 31, à exclure de tout classement/sélection tant
  qu'elles n'ont pas été relancées avec plus de cycles (`%geom MaxIter`).
  Liste dans `results_fragmentation.csv` croisée avec un scan `grep` du
  `.out` (pas encore automatisé dans `harvest_results.py`).
- Rapport d'analyse : `report/rapport_campagne_dft_wb97xd4_v1.tex`.

## Campagne 2 — CCSD(T)-F12 (démarrée 2026-09-06)

- But : statistiques temps humain/CPU + nombre de cœurs optimal pour
  DLPNO-CCSD(T)-F12, sur un sous-ensemble représentatif avant d'attaquer
  le reste.
- Sélection : 3 structures DFT les plus stables par groupe (N, charge)
  parmi les confirmées (non fragmentées, sans fréquence imaginaire, Opt
  DFT convergée) + N2 (référence ΔHf neutres, jamais calculée au-delà de
  xTB avant cette campagne) + NH4⁺ tétraédrique et NH2⁻ coudée (espèces
  N-H demandées pour des réactions chimiques annexes). 49 structures au
  total. Liste : `ccsdt_jobs/jobs_list_ccsdt.csv`.
- Cœurs par taille : N2–N5 → 8, N6–N8 → 16, N9–N11 → 24, N12+ → 48.
- Méthode : `! DLPNO-CCSD(T)-F12 aug-cc-pVTZ-F12 TightPNO TightSCF` (choisie
  par Gilles, cf. `test-ccsdt.inp`).
- Protocole par structure, dans `ccsdt_jobs/<nom>/` : (1) `<nom>_resym.inp`
  — réoptimisation DFT WB97X-D4/aug-cc-pVTZ `LooseOpt UseSym` (même niveau
  que la campagne 1) à partir de la géométrie DFT déjà optimisée, pour
  symétriser proprement dans le groupe ponctuel détecté ; (2)
  `<nom>_ccsdt.inp` — single-point CCSD(T)-F12 sur cette géométrie
  symétrisée, `MORead` + `%moinp "<nom>_resym.gbw"` (réutilise la fonction
  d'onde DFT comme guess). Tout est conservé (géométries, `.gbw`, `.out`) :
  la campagne 3 fera l'optimisation complète au niveau CCSD(T).
- Soumission : `sbatch` direct par job (`submit_ccsdt_jobs.py`), pas de
  dispatcher maison — SLURM place chaque job dès que les cœurs demandés
  sont libres. Chaque étape chronométrée séparément (`<nom>_resym.time`,
  `<nom>_ccsdt.time`, via `/usr/bin/time -v`) pour distinguer le coût DFT
  du coût CCSD(T) dans les statistiques.
- Scripts (générés en session, à rapatrier dans le dépôt une fois
  stabilisés) : `/tmp/build_ccsdt_jobs.py`, `/tmp/submit_ccsdt_jobs.py`.
- **Mots-clés CCSD(T) définitifs (validés 2026-09-06, après 2 corrections)** :
  `! DLPNO-CCSD(T)-F12 cc-pVTZ-F12 cc-pVTZ-F12-CABS cc-pVTZ/C TightPNO TightSCF`.
  `aug-cc-pVTZ-F12` (proposé initialement dans `test-ccsdt.inp`) **n'existe
  pas** dans ORCA 6.1 — les bases F12 de Peterson n'ont pas de variante
  `aug-`, c'est `cc-pVTZ-F12` (ou `cc-pVDZ-F12`) sans préfixe. Il manque
  aussi, sans quoi ORCA refuse de lancer le calcul (`DimCABS = 0` /
  `we need at least one auxiliary basis set`) : la base CABS
  (`cc-pVTZ-F12-CABS`) et la base d'ajustement de corrélation (`cc-pVTZ/C`).
- Test de validation sur N2 complet et réussi (2026-09-06, 8 cœurs) :
  - resym DFT : 2:04 (wall), 8:09 CPU (394 %), d(N-N) optimisée = 1,0912 Å.
  - single-point CCSD(T)-F12 : 2:29 (wall), 13:28 CPU (541 %),
    E = -109,412670614 Ha.
  - Pipeline complet : 4:33 wall, ~21:37 CPU cumulé sur 8 cœurs.
- **Les 49 jobs de la campagne sont lancés** (2026-09-06, jobs SLURM
  59840-59888, soumis directement via `sbatch` sans dispatcher maison —
  24 démarrés immédiatement, 27 en attente de cœurs).
- Remplissage automatique des cœurs libres avec des candidats N4-N8
  restants (hors sélection initiale) : `fill_free_cores.py` lit
  `ccsdt_jobs/remaining_pool_n4_n8.csv`, soumet ce qui tient dans les
  cœurs libres, retire de la liste ce qui est soumis. Invoqué par la
  boucle de réveil automatique en place (toutes les 30 min) en même
  temps que la mise à jour du tableau de comparaison xTB/DFT/CCSD(T) du
  rapport (`report/scripts/build_ccsdt_comparison.py`).
- Rapport : nouvelle section "Couverture du criblage et pistes de
  complément" (`report/scripts/build_coverage_analysis.py`) — décompose
  les 193 candidats par (N, charge), repère les groupes à 0 candidat
  (charges paires sur toute la gamme N4-N16, plus N14/N15/N16 aux tailles
  extrêmes → hors de portée pratique du pool 1, cible naturelle du
  pool 2) et les groupes à faible rendement DFT (N5⁻, N7⁺, N9⁺, N11⁻,
  N13⁻ → encore dans la zone confortable du pool 1, ré-échantillonnage
  ciblé plutôt que changement de générateur).

## Correction multi-fidélité pour le surrogate (pool 2)

`surrogate/multifidelity.py` (`FidelityCorrector`) : à partir des paires
xTB/DFT (140 structures confirmées de la campagne 1) et DFT/CCSD(T) (34
et croissant, campagne 2), entraîne une correction Δ(fidélité haute −
fidélité basse) sur les 11 descripteurs topologiques de
`core/graph_descriptors.py`, avec sélection automatique par validation
croisée entre gradient boosting et une simple régression linéaire sur
`n_atoms` (jamais le modèle riche imposé si les données ne le supportent
pas).

**Résultat honnête, pas un gain systématique** : gradient boosting bat la
baseline seulement pour les cations (Δ DFT-xTB, +58% de réduction de MAE
en validation croisée) ; pour les neutres, les anions et pour DFT→CCSD(T)
(jeu encore ≤34 points), la baseline linéaire reste meilleure — les 11
descripteurs purs de connectivité (pas de charge explicite, pas d'ordre
de liaison, pas de groupe ponctuel) ne capturent pas assez de signal
au-delà du simple nombre d'atomes pour ces familles.

Reproductible via `orca_jobs/build_multifidelity_dataset.py` (export des
CSV d'entraînement — nécessite `/usr/bin/python3.11` pour
numpy/networkx/scikit-learn, absents du python3 système 3.6) puis
`orca_jobs/train_multifidelity.py` (validation croisée, rapport texte).
À relancer périodiquement, le jeu DFT/CCSD(T) grossissant avec la
campagne 2.

**Pas encore fait** : intégration dans `pipeline/population_loop.py` (le
module est autonome, prêt à l'emploi, mais rien dans la boucle de
génération ne l'appelle encore) ; enrichissement des descripteurs
(charge, ordre de liaison, groupe ponctuel) qui expliquerait
probablement le signal manquant pour neutres/anions.
