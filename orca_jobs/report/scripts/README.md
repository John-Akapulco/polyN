# Scripts de génération du rapport

Scripts utilisés pour produire les fragments `table_*.tex`, `annexe_structures.tex`,
`annexe_references.tex` et les images `figs/*.png` de `rapport_campagne_dft_wb97xd4.tex`.

**Non entièrement autonomes en l'état** : `build_provenance.py` et `build_charges_table.py`
lisent quelques fichiers intermédiaires générés en session dans `/tmp/`
(`ref_filtered.csv`, `topology_match_report.csv`, `name_to_ref.json`,
`table_references.tex`, `table_isodesmic_{neutre,anion,cation}.tex` — extraits
de `polyN-pipeline` : le sous-ensemble filtré de
`biblio_polyN/comparison_table_3sources.csv`, le rapport de correspondance
topologique, la bibliographie numérotée, et le mapping structure→référence)
qui ne sont pas encore versionnés ici. À refactorer en scripts autonomes
(lisant directement depuis `polyN-pipeline` ou depuis des CSV committés) si
ce rapport doit être régénéré en dehors de cette session.

Ordre d'exécution : `fragmentation_check.py` (écrit
`../results_fragmentation.csv`, 193 structures, DFT si termin\'e sinon
xTB) → `compute_dhf.py` → `render_molecules.py` (rend Figures S1-S3,
non-fragmentées uniquement, dans `figs/`, et Figure S4 avant/après dans
`figs/fragmented/`) → `build_provenance.py` (produit aussi
`annexe_references.tex`, `table_S1_fragmented.tex` et
`/tmp/provenance_state.pkl`, requis par les étapes suivantes) →
`build_appendix.py` (S1-S3 + S4) → `build_tables2.py` (tableaux de
classement + `table_S2_imaginary.tex`) → `build_charges_table.py`, puis
`pdflatex rapport_campagne_dft_wb97xd4.tex` (deux passes, pour les
références croisées et la table des matières).

`build_provenance.py` produit 3 tableaux séparés (neutres/cations/anions),
triés par nombre d'atomes croissant puis par $\Delta H$ (kcal/mol, isomère
le plus stable **non-fragmenté** de sa formule = 0, DFT si disponible
sinon xTB). Convention de référence : un candidat correspondant à une
entrée `biblio_article` de `comparison_table_3sources.csv` reçoit le(s)
numéro(s) d'article (résolu via `table_isodesmic_*.tex` ; si plusieurs
articles distincts partagent la même empreinte topologique, tous sont
cités) ; tout le reste (aucune correspondance, ou correspondance aux
bases externes `N_csp`/`polyN_study`, qui ne sont pas des articles
publiés) reçoit `our work`.

**Fragmentation** (`fragmentation_check.py`) : composantes connexes du
graphe N–N à 1,70 Å (union-find). Un candidat qui se sépare en >1
composante (ex. N₂ + N₃⁻ au lieu d'un N₅⁻ lié) est exclu de tous les
tableaux principaux et de la comparaison de stabilité — il fausserait le
classement (une espèce dissociée est souvent artificiellement plus basse
en énergie qu'un vrai cluster lié) — et regroupé à part dans le Tableau S1
et la Figure S4 (Annexe, Supporting Information). Les tableaux et figures
S1/S2 (Annexe) utilisent un en-tête manuel (`\textbf{Table S1.}` etc.)
plutôt que `\caption` pour garder une numérotation SI (S1, S2…)
indépendante de la séquence Table 1, 2, 3… du corps du rapport.
