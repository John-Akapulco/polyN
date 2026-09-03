"""
Source de candidats par MUTATION de l'archive -- le relais de l'énumération
exhaustive geng pour les grandes tailles (n > ~16, seuil mesuré où geng
devient impraticable : ~4.1M graphes en 37s à n=16, x4 par atome au-delà).

Deux problèmes résolus ici, au-delà des opérateurs bruts de mutation.py :

1. AMORÇAGE (génération 1) : l'archive est vide au départ, il n'y a rien à
   muter. On génère des graines diverses par construction directe --
   chaînes, cycles, et graphes aléatoires à séquence de degrés contrainte
   (même esprit que seed_topologies_for_mh de polynitrogen_charged_explore).
   Pas d'énumération geng ici : à ces tailles, même streamer l'espace est
   trop coûteux.

2. DÉDOUBLONNAGE DES CANDIDATS PROPOSÉS : les mutations de graphes proches
   produisent souvent les mêmes voisins ; on filtre par hash Weisfeiler-
   Lehman (rapide, pas d'isomorphisme exact ici -- un hash WL identique
   pour deux graphes non isomorphes est rare et sans gravité à ce stade :
   ça coûte juste un candidat de moins dans le lot, l'archive fait le vrai
   dédoublonnage exact en aval).

Usage dans run_campaign :

    from polyN_adapt.generators.mutation_source import MutationSource

    src = MutationSource(n=20, max_degree=3, n_candidates_per_generation=2000, seed=0)
    # à chaque génération, run_campaign appelle source_fn() ; la source lit
    # l'archive courante via set_archive() appelé entre les générations --
    # cf. run_campaign_mutation ci-dessous qui gère ce câblage.
"""

from __future__ import annotations

from collections.abc import Iterator

import networkx as nx
import numpy as np

from .mutation import mutate


def _random_seed_graph(n: int, max_degree: int, rng) -> nx.Graph | None:
    """Graphe connexe aléatoire à n sommets, degré <= max_degree, par
    séquence de degrés valide + reconstruction, avec repli sur une chaîne
    perturbée si la séquence tirée n'est pas graphique/connectable."""
    for _ in range(20):
        # tirer une séquence de degrés dans {1,2,3}, somme paire
        seq = rng.integers(1, max_degree + 1, size=n)
        if seq.sum() % 2 != 0:
            idx = rng.integers(0, n)
            seq[idx] += 1 if seq[idx] < max_degree else -1
        seq = np.clip(seq, 1, max_degree)
        if seq.sum() % 2 != 0:
            continue
        try:
            G = nx.random_degree_sequence_graph(seq.tolist(), seed=int(rng.integers(0, 2**31)))
            G = nx.Graph(G)  # retirer multi-arêtes éventuelles
            G.remove_edges_from(nx.selfloop_edges(G))
            if nx.is_connected(G) and max(d for _, d in G.degree()) <= max_degree:
                return G
        except (nx.NetworkXError, nx.NetworkXUnfeasible):
            continue
    return None


def generate_seed_graphs(n: int, max_degree: int, n_seeds: int, rng) -> list[nx.Graph]:
    """Graines diverses pour amorcer la génération 1 : chaîne, cycle, et
    graphes aléatoires contraints. Déterministe à seed donné."""
    seeds: list[nx.Graph] = [nx.path_graph(n), nx.cycle_graph(n)]
    attempts = 0
    while len(seeds) < n_seeds and attempts < n_seeds * 10:
        attempts += 1
        G = _random_seed_graph(n, max_degree, rng)
        if G is not None:
            seeds.append(G)
    return seeds[:n_seeds]


class MutationSource:
    """
    Source de candidats pour run_campaign, basée sur la mutation des
    graphes de l'archive courante.

    Génération 1 (archive vide) : graines construites directement.
    Générations suivantes : chaque candidat = un graphe parent tiré de
    l'archive (biais vers les basses énergies), muté 1 à max_mutations fois.
    """

    def __init__(self, n: int, max_degree: int = 3,
                 n_candidates_per_generation: int = 2000,
                 max_mutations_per_candidate: int = 3,
                 seed: int = 0):
        self.n = n
        self.max_degree = max_degree
        self.n_candidates = n_candidates_per_generation
        self.max_mutations = max_mutations_per_candidate
        self.rng = np.random.default_rng(seed)
        self._archive_graphs: list[nx.Graph] = []
        self._archive_energies: list[float] = []

    def set_archive(self, graphs: list[nx.Graph], energies_per_atom: list[float]) -> None:
        """À appeler entre les générations avec le contenu courant de
        l'archive (graphes FINAUX relaxés + leurs énergies) -- ce sont eux
        qui sont mutés, pas les candidats d'origine."""
        self._archive_graphs = [g for g in graphs if g is not None]
        self._archive_energies = [e for g, e in zip(graphs, energies_per_atom) if g is not None]

    def _parent_probabilities(self) -> np.ndarray:
        """Biais de sélection du parent vers les basses énergies (softmax
        sur -E, température fixe modérée) -- exploite sans être glouton."""
        e = np.array(self._archive_energies)
        w = np.exp(-(e - e.min()) / 0.05)  # 0.05 eV/atome de 'température'
        return w / w.sum()

    def __call__(self) -> Iterator[nx.Graph]:
        seen_hashes: set[str] = set()

        if not self._archive_graphs:
            # Génération 1 : amorçage par graines construites
            for G in generate_seed_graphs(self.n, self.max_degree, self.n_candidates, self.rng):
                h = nx.weisfeiler_lehman_graph_hash(G)
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    yield G
            return

        probs = self._parent_probabilities()
        produced = 0
        attempts = 0
        max_attempts = self.n_candidates * 10

        while produced < self.n_candidates and attempts < max_attempts:
            attempts += 1
            parent_idx = self.rng.choice(len(self._archive_graphs), p=probs)
            G = self._archive_graphs[parent_idx]

            n_mut = int(self.rng.integers(1, self.max_mutations + 1))
            for _ in range(n_mut):
                mutated = mutate(G, max_degree=self.max_degree, rng=self.rng)
                if mutated is None:
                    break
                G = mutated
            else:
                # toutes les mutations ont réussi
                if G.number_of_nodes() != self.n:
                    continue  # invariant de taille, ne devrait jamais arriver, garde-fou
                h = nx.weisfeiler_lehman_graph_hash(G)
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    produced += 1
                    yield G
