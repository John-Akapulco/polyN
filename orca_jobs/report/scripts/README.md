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

Ordre d'exécution : `compute_dhf.py` → `render_molecules.py` →
`build_provenance.py` (produit aussi `annexe_references.tex` et
`/tmp/provenance_state.pkl`, requis par l'étape suivante) →
`build_appendix.py` → `build_tables2.py` → `build_charges_table.py`, puis
`pdflatex rapport_campagne_dft_wb97xd4.tex` (deux passes, pour les
références croisées et la table des matières).

`build_provenance.py` produit 3 tableaux séparés (neutres/cations/anions),
triés par nombre d'atomes croissant puis par $\Delta H$ (kcal/mol, isomère
le plus stable de sa formule = 0, DFT si disponible sinon xTB). Convention
de référence : un candidat correspondant à une entrée `biblio_article` de
`comparison_table_3sources.csv` reçoit un numéro d'article (résolu via
`table_isodesmic_*.tex`) ; tout le reste (aucune correspondance, ou
correspondance aux bases externes `N_csp`/`polyN_study`, qui ne sont pas
des articles publiés) reçoit `our work`.
