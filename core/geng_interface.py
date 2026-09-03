"""
Interface vers nauty-geng : génère les graphes connectés à n sommets, degré
max imposé directement par geng (-D3, coordination physique de l'azote),
et les convertit en objets networkx via graph6.

Règle de parité charge/n (généralisation de celle déjà validée dans
polynitrogen_charged_explore.py pour charge=-1) :
    n_electrons = 7*n - charge
    singulet (couche fermée) <=> n_electrons pair <=> (7n - charge) pair
    7 est impair, donc (7n - charge) pair <=> (n - charge) pair
    <=> n et charge de meme parite
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Iterator

import networkx as nx


def check_geng_available() -> None:
    if shutil.which("nauty-geng") is None and shutil.which("geng") is None:
        sys.exit(
            "ERREUR : ni 'nauty-geng' ni 'geng' trouve sur le PATH.\n"
            "Installer nauty, ex. : conda install -c conda-forge nauty\n"
            "ou (Debian/Ubuntu) : apt-get install nauty"
        )


def _geng_binary() -> str:
    return shutil.which("nauty-geng") or shutil.which("geng")


def required_parity(charge: int) -> str:
    """Retourne 'even' ou 'odd' -- la parité de n requise pour un singulet
    à cette charge (electron_count = 7n - charge doit etre pair)."""
    return "even" if charge % 2 == 0 else "odd"


def validate_n_for_charge(n: int, charge: int) -> None:
    parity = required_parity(charge)
    n_parity = "even" if n % 2 == 0 else "odd"
    if parity != n_parity:
        raise ValueError(
            f"n={n} incompatible avec charge={charge:+d} pour un etat singulet "
            f"(7n - charge = {7*n - charge} electrons, impair -- pas de couche "
            f"fermee possible). n doit etre {parity} pour cette charge."
        )


def stream_geng_graphs(n: int, max_degree: int = 3, min_degree: int = 1) -> Iterator[nx.Graph]:
    """
    Génère en streaming tous les graphes connectés à n sommets, degré dans
    [min_degree, max_degree], via nauty-geng -c -D{max_degree} -d{min_degree}.

    Streaming (pas de liste complète en mémoire) : pour les grandes valeurs
    de n (jusqu'a ~14-16 mesure praticable, cf. discussion), le nombre de
    graphes peut atteindre plusieurs millions -- charger tout en mémoire
    d'un coup serait inutilement coûteux.
    """
    check_geng_available()
    binary = _geng_binary()
    cmd = [binary, "-c", f"-D{max_degree}", f"-d{min_degree}", str(n)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            g6 = line.strip()
            if not g6 or g6.startswith(">"):
                continue
            yield nx.from_graph6_bytes(g6.encode())
    finally:
        # Terminer explicitement le sous-processus si on arrête de consommer
        # avant la fin du flux (ex: budget de population atteint) -- sans ça,
        # geng continue de tourner en arrière-plan et de remplir le pipe.
        proc.stdout.close()
        proc.terminate()
        proc.wait(timeout=5)


def count_geng_graphs(n: int, max_degree: int = 3, min_degree: int = 1) -> int:
    """Compte rapide sans stocker les graphes -- utile pour dimensionner une
    population (ex: decider si l'enumeration exhaustive est praticable a ce n)."""
    check_geng_available()
    binary = _geng_binary()
    cmd = [binary, "-c", f"-D{max_degree}", f"-d{min_degree}", str(n)]
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    # nauty-geng ecrit le compte final sur stderr, format ">Z NNN graphs generated..."
    for line in proc.stderr.splitlines():
        if "graphs generated" in line:
            return int(line.split()[1])
    return -1
