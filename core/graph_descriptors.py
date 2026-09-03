"""
Descripteurs topologiques bon marché, calculables sur un graphe networkx nu
(connectivité pure, sans ordre de liaison, sans embed 3D, sans xTB/DFTB+).

Ces descripteurs alimentent à la fois le surrogate adaptatif (régresseur +
règle interprétable) et, plus tard, le mineur de motifs stabilisants/
déstabilisants sur l'archive multi-tailles.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import networkx as nx


@dataclass
class GraphDescriptors:
    n_atoms: int
    degree_max: int
    degree_mean: float
    frac_degree_1: float   # fraction de sommets terminaux (candidats N=N ou N≡N en bout de chaîne)
    frac_degree_2: float
    frac_degree_3: float
    min_cycle_size: int    # 0 si acyclique (arbre/chaîne) ; jamais None pour rester ML-friendly
    n_cycles_independent: int  # taille de la base de cycles (nx.cycle_basis)
    is_tree: bool
    n_fragments: int       # 1 si connexe -- un graphe fragmenté ne devrait normalement
                            # jamais sortir de geng -c, mais on le garde par robustesse
    max_local_cycle_overlap: int  # nombre max de cycles partageant un même sommet
                                    # (proxy grossier pour "cage"/polycyclique fusionné,
                                    # signal de tension géométrique potentiellement élevée)

    def as_dict(self) -> dict:
        return asdict(self)

    def as_vector(self, feature_order: list[str] | None = None) -> list[float]:
        """Représentation vectorielle pour scikit-learn, ordre de features fixe
        et explicite (jamais l'ordre d'itération d'un dict, qui n'est garanti
        stable qu'à partir de Python 3.7 mais reste fragile si le schéma évolue)."""
        d = self.as_dict()
        order = feature_order or DEFAULT_FEATURE_ORDER
        return [float(d[k]) for k in order]


DEFAULT_FEATURE_ORDER = [
    "n_atoms", "degree_max", "degree_mean",
    "frac_degree_1", "frac_degree_2", "frac_degree_3",
    "min_cycle_size", "n_cycles_independent", "is_tree",
    "n_fragments", "max_local_cycle_overlap",
]


def compute_descriptors(G: nx.Graph) -> GraphDescriptors:
    n = G.number_of_nodes()
    if n == 0:
        raise ValueError("graphe vide")

    degrees = [d for _, d in G.degree()]
    degree_max = max(degrees)
    degree_mean = sum(degrees) / n

    frac_1 = sum(1 for d in degrees if d == 1) / n
    frac_2 = sum(1 for d in degrees if d == 2) / n
    frac_3 = sum(1 for d in degrees if d == 3) / n

    cycles = nx.cycle_basis(G)
    n_cycles = len(cycles)
    min_cycle = min((len(c) for c in cycles), default=0)
    is_tree = (n_cycles == 0)

    n_fragments = nx.number_connected_components(G)

    # Recouvrement local de cycles : pour chaque sommet, combien de cycles de
    # la base passent par lui -- un proxy grossier mais rapide pour repérer
    # les jonctions polycycliques fusionnées (cages), sans devoir énumérer
    # tous les cycles (coûteux) ni faire de la reconnaissance de motif fine.
    if cycles:
        membership = {v: 0 for v in G.nodes()}
        for c in cycles:
            for v in c:
                membership[v] += 1
        max_overlap = max(membership.values())
    else:
        max_overlap = 0

    return GraphDescriptors(
        n_atoms=n,
        degree_max=degree_max,
        degree_mean=degree_mean,
        frac_degree_1=frac_1,
        frac_degree_2=frac_2,
        frac_degree_3=frac_3,
        min_cycle_size=min_cycle,
        n_cycles_independent=n_cycles,
        is_tree=is_tree,
        n_fragments=n_fragments,
        max_local_cycle_overlap=max_overlap,
    )
