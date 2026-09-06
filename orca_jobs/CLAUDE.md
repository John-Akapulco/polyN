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
