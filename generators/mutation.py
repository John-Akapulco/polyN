"""
Mutations de graphe à n ATOMES CONSTANT (jamais d'ajout/retrait de sommet --
seule la connectivité change). Utilisées pour explorer le voisinage
topologique de l'archive une fois que l'énumération exhaustive geng devient
impraticable (n au-delà du seuil mesuré empiriquement sur la machine cible).
"""

from __future__ import annotations

import networkx as nx
import numpy as np


def _degree_ok(G: nx.Graph, node, max_degree: int) -> bool:
    return G.degree(node) < max_degree


def add_edge_mutation(G: nx.Graph, max_degree: int = 3, rng=None, max_attempts: int = 50) -> nx.Graph | None:
    """Ajoute une arête entre deux sommets non connectés, si la contrainte
    de degré le permet des deux côtés. Retourne None si aucun ajout valide
    n'a été trouvé après max_attempts essais (graphe déjà proche du complet
    localement, ou tous les sommets saturés en degré)."""
    rng = rng or np.random.default_rng()
    nodes = list(G.nodes())
    n = len(nodes)
    for _ in range(max_attempts):
        u, v = rng.choice(n, size=2, replace=False)
        u, v = nodes[u], nodes[v]
        if G.has_edge(u, v):
            continue
        if _degree_ok(G, u, max_degree) and _degree_ok(G, v, max_degree):
            H = G.copy()
            H.add_edge(u, v)
            return H
    return None


def remove_edge_mutation(G: nx.Graph, rng=None, max_attempts: int = 50) -> nx.Graph | None:
    """Retire une arête, seulement si le graphe reste connexe après (ne
    fragmente jamais un candidat -- un graphe fragmenté n'a aucune valeur
    ici, contrairement à une structure 3D relaxée qui peut légitimement se
    fragmenter en cours de relaxation)."""
    rng = rng or np.random.default_rng()
    edges = list(G.edges())
    if not edges:
        return None
    idx_order = rng.permutation(len(edges))
    for idx in idx_order[:max_attempts]:
        u, v = edges[idx]
        H = G.copy()
        H.remove_edge(u, v)
        if nx.is_connected(H):
            return H
    return None


def double_edge_swap_mutation(G: nx.Graph, rng=None, max_attempts: int = 50) -> nx.Graph | None:
    """Rewiring préservant EXACTEMENT la séquence de degrés : retire deux
    arêtes (a-b) et (c-d), en recrée deux différentes (a-d) et (c-b), sous
    réserve que ces nouvelles arêtes n'existent pas déjà et que le graphe
    reste connexe. C'est la mutation la plus 'douce' -- elle explore une
    topologie voisine sans changer la distribution de coordination globale,
    contrairement à add/remove qui la modifient."""
    rng = rng or np.random.default_rng()
    edges = list(G.edges())
    if len(edges) < 2:
        return None
    for _ in range(max_attempts):
        i, j = rng.choice(len(edges), size=2, replace=False)
        a, b = edges[i]
        c, d = edges[j]
        if len({a, b, c, d}) < 4:
            continue  # arêtes partageant un sommet -- swap non défini proprement
        if G.has_edge(a, d) or G.has_edge(c, b):
            continue
        H = G.copy()
        H.remove_edge(a, b)
        H.remove_edge(c, d)
        H.add_edge(a, d)
        H.add_edge(c, b)
        if nx.is_connected(H):
            return H
    return None


MUTATION_OPERATORS = {
    "add_edge": add_edge_mutation,
    "remove_edge": remove_edge_mutation,
    "double_edge_swap": double_edge_swap_mutation,
}


def mutate(G: nx.Graph, max_degree: int = 3, rng=None,
           weights: dict[str, float] | None = None) -> nx.Graph | None:
    """Tire un opérateur de mutation au hasard (pondéré) et l'applique.
    Retourne None si l'opérateur choisi n'a trouvé aucune mutation valide
    (l'appelant doit alors réessayer ou passer au candidat suivant)."""
    rng = rng or np.random.default_rng()
    weights = weights or {"add_edge": 1.0, "remove_edge": 1.0, "double_edge_swap": 2.0}
    names = list(weights.keys())
    probs = np.array([weights[k] for k in names], dtype=float)
    probs /= probs.sum()
    op_name = rng.choice(names, p=probs)
    op = MUTATION_OPERATORS[op_name]
    if op_name == "add_edge":
        return op(G, max_degree=max_degree, rng=rng)
    return op(G, rng=rng)
