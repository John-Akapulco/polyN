"""
Lecture minimaliste de fichiers .xyz (format standard : compte / commentaire
/ lignes "symbole x y z"). IO générique sans logique de chimie
computationnelle -- non concernée par la règle de non-duplication du
README principal (qui porte sur embed_graph_3d_ff / check_structural_integrity
/ classify_topology, importés depuis polynitrogen_charged_explore.py).
"""

from __future__ import annotations

from pathlib import Path

Atom = tuple[str, float, float, float]


def read_xyz_atoms(path: str | Path) -> list[Atom]:
    lines = Path(path).read_text().splitlines()
    n = int(lines[0].split()[0])
    atoms: list[Atom] = []
    for ln in lines[2 : 2 + n]:
        parts = ln.split()
        atoms.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
    return atoms
