# Scripts de génération du rapport

Scripts utilisés pour produire les fragments `table_*.tex`, `annexe_*.tex`
et les images `figs/*.png` de `rapport_campagne_dft_wb97xd4_v1.tex`.

**Non entièrement autonomes en l'état** : `build_provenance.py`,
`build_charges_table.py` et `build_references.py` lisent des fichiers
intermédiaires générés en session, certains dans `/tmp/` (`topology_match_report.csv`,
`name_to_ref.json`, `table_references.tex` — bibliographie des 22 références
extraites, codes GS/B/A1–A20 — et `table_isodesmic_{neutre,anion,cation}.tex`,
tous dérivés de `polyN-pipeline`), d'autres dans le scratchpad d'une session
antérieure (`build_references.py` lit directement les JSON de
`biblio_polyN/citation_search/` — `pure_n_candidates_gen1.json`,
`curated_gen2.json` — pour le corpus de 156 références non extraites). À
refactorer en scripts autonomes (lisant directement depuis `polyN-pipeline`
ou depuis des CSV/JSON committés ici) si ce rapport doit être régénéré en
dehors de cette session.

Ordre d'exécution :
1. `fragmentation_check.py` — écrit `../results_fragmentation.csv` (193
   structures). Chaque structure est testée **aux deux niveaux**
   indépendamment (xTB de départ et DFT si terminé) : colonnes
   `n_fragments`/`fragment_sizes` (niveau retenu pour le rapport, DFT si
   disponible sinon xTB) et `n_fragments_xtb_initial`/`already_fragmented_at_xtb`
   (statut de la géométrie de départ, avant tout calcul DFT — distingue un
   artefact du criblage GFN2-xTB en amont d'une fragmentation survenue
   pendant l'optimisation DFT elle-même).
2. `compute_dhf.py` — `../results_dHf_xtb.csv`.
3. `render_molecules.py` — Figures S2/S3/S4 (structures confirmées
   non-fragmentées, une figure par famille de charge) dans `figs/`, et
   Figure S1 avant/après (candidats fragmentés) dans `figs/fragmented/`.
4. `build_provenance.py` — Tableaux 2–4, Tableau S1, et
   `/tmp/provenance_state.pkl` (requis par les étapes suivantes : `rel_dH`,
   `dft_based`, `fragmented`, `code_to_num`, `name_to_ref`, `topo`).
   `code_to_num` fixe la numérotation des 22 références extraites
   (GS, B, A1–A20, ordre fixe) — cette numérotation doit rester identique
   à celle utilisée par `build_references.py`.
5. `build_appendix.py` — Figure S1 (fragmentées, avant/après + légende de
   fin de figure) et Figures S2–S4 (structures confirmées).
6. `build_tables2.py` — Tableaux 5–8 (classement DFT/xTB par famille +
   résumé d'accord de rang, orbitales frontières) et Tableau 10 (liaisons
   N–N vs $\Delta H_f$), plus Tableau S2 (fréquences imaginaires).
7. `build_charges_table.py` — Tableau 9.
8. `build_references.py` — Annexe Références : les 22 références extraites
   (numérotées 1–22, statut "structure extraite") suivies du corpus de
   recherche de citations OpenAlex non encore réduit en structures
   (numérotées 23–178, statut "à faire" — Tableau 13, le travail
   bibliographique restant). Titres assainis (`sanitize_title`) : les
   métadonnées OpenAlex contiennent des caractères unicode et parfois du
   balisage TeX déjà présent dans le titre source, incompatibles tels
   quels avec pdflatex (`inputenc` utf8 seul) — aplatis en texte brut.
9. `pdflatex rapport_campagne_dft_wb97xd4_v1.tex` (deux passes, pour les
   références croisées et la table des matières).

`build_provenance.py` produit 3 tableaux séparés (neutres/cations/anions),
triés par nombre d'atomes croissant puis par $\Delta H$ (kcal/mol, isomère
le plus stable **non-fragmenté et sans fréquence imaginaire** de sa
formule = 0, DFT si disponible sinon xTB — l'exclusion des candidats à
fréquence imaginaire du pool d'ancrage évite qu'un point-selle serve de
"ground state" de référence). Convention de référence : un candidat
correspondant à une entrée `biblio_article` de `comparison_table_3sources.csv`
reçoit le(s) numéro(s) d'article (résolu via `table_isodesmic_*.tex` ; si
plusieurs articles distincts partagent la même empreinte topologique, tous
sont cités) ; tout le reste (aucune correspondance, ou correspondance aux
bases externes `N_csp`/`polyN_study`, qui ne sont pas des articles publiés)
reçoit `our work`.

**Fragmentation** (`fragmentation_check.py`) : composantes connexes du
graphe N–N à 1,70 Å (union-find). Un candidat qui se sépare en >1
composante (ex. N₂ + N₃⁻ au lieu d'un N₅⁻ lié) n'apparaît pas dans les
tableaux principaux ni dans la comparaison de stabilité — une espèce
dissociée est souvent artificiellement plus basse en énergie qu'un vrai
cluster lié — et est listé à part dans le Tableau S1 et la Figure S1
(Annexe, Supporting Information), avec son statut de fragmentation à la
géométrie de départ (xTB).

**Numérotation des tableaux/figures Supporting Information** : Tableaux
S1/S2 utilisent un en-tête manuel (`\textbf{Tableau S1.}` etc.) plutôt que
`\caption`, pour garder une numérotation SI (S1, S2…) indépendante de la
séquence Tableau 1, 2, 3… du corps du rapport. Les Figures S1–S4 sont
numérotées dans l'ordre où elles apparaissent en lisant le document (S1 =
fragmentées, en premier car en §A.1 ; S2/S3/S4 = confirmées neutres/
cations/anions, en §A.3) et non par ordre thématique, pour respecter un
ordre croissant à la lecture.

**Piège LaTeX à ne pas réintroduire** : ne jamais placer `\scriptsize`
comme premier token à l'intérieur de `\begin{longtable}{...}` avant
`\caption{}` — casse le mécanisme interne `\noalign` de longtable
(silencieusement : le document compile quand même, mais le titre du
tableau atterrit hors de la mise en page normale de la légende). Toujours
englober la table entière dans un groupe externe, `{\scriptsize
\begin{longtable}...\end{longtable}}`, avec `\usepackage{caption}` +
`\captionsetup{font=small,...}` en préambule pour que la légende garde une
taille de police cohérente avec les tableaux non-longtable (Tableau 1)
malgré le corps de table en `\scriptsize`.
