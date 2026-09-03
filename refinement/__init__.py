"""
Préparation et soumission de jobs de raffinement post-xTB (Gaussian ou
ORCA), entièrement pilotée par un fichier de configuration YAML -- aucune
méthode/base/type de calcul n'est codé en dur dans ce package.

Consomme directement archive.Archive.export_refinement_queue() : cet étage
est réservé à la revérification de quelques isomères bas en énergie à un
niveau de théorie supérieur (ab initio/DFT/CCSD(T)), pas au criblage
haut-débit (cf. README.md du package parent).
"""
