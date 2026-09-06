# Scripts de génération du rapport

Scripts utilisés pour produire les fragments `table_*.tex`, `annexe_structures.tex`
et les images `figs/*.png` de `rapport_campagne_dft_wb97xd4.tex`.

**Non entièrement autonomes en l'état** : `build_tables.py` et `build_charges_table.py`
lisent quelques fichiers intermédiaires générés en session dans `/tmp/`
(`ref_filtered.csv`, `topology_match_report.csv` — le sous-ensemble filtré de
`biblio_polyN/comparison_table_3sources.csv` et le rapport de correspondance
topologique de `polyN-pipeline`) qui ne sont pas encore versionnés ici. À
refactorer en scripts autonomes (lisant directement depuis `polyN-pipeline`
ou depuis des CSV committés) si ce rapport doit être régénéré en dehors de
cette session.

Ordre d'exécution : `compute_dhf.py` → `render_molecules.py` → `build_tables.py`
→ `build_tables2.py` → `build_charges_table.py` → `build_appendix.py`, puis
`pdflatex rapport_campagne_dft_wb97xd4.tex` (deux passes, pour les
références croisées et la table des matières).
