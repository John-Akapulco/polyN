"""
Archive des isomères survivants, toutes populations confondues.

Point important, discuté explicitement : la fenêtre de rétention (0.2 eV/
atome par défaut) est recalculée par rapport au MEILLEUR MINIMUM CONNU A CE
JOUR, pas figée sur la référence de la population 1 -- si une population
ultérieure révèle un minimum plus bas, toute l'archive est reclassée contre
cette nouvelle référence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx


def guess_multiplicity(n_atoms: int, charge: int, atomic_number: int = 7) -> int:
    """Multiplicite de spin par defaut pour un cluster homonucleaire de
    n_atoms atomes de numero atomique atomic_number (7 = azote) et de
    charge donnee : nombre d'electrons pair -> singulet (1), impair ->
    doublet (2). C'est une estimation de repli, pas une determination
    physique (un vrai etat fondamental ouvert-couche peut differer) --
    utilisee par export_refinement_queue et refinement.submit comme valeur
    par defaut, toujours surchageable (CalculationStep.multiplicity_override)."""
    n_electrons = atomic_number * n_atoms - charge
    return 2 if n_electrons % 2 else 1


@dataclass
class ArchiveEntry:
    candidate_graph: nx.Graph  # graphe d'origine (geng/mutation) -- utilisé UNIQUEMENT
                                # pour recalculer/retrouver les descripteurs du surrogate,
                                # jamais pour l'identité de l'entrée
    final_graph: nx.Graph | None  # graphe reconstruit géométriquement après relaxation --
                                    # c'est LA structure physique réelle, utilisée pour le
                                    # dédoublonnage et l'identité de l'entrée. None si
                                    # l'évaluation a échoué avant d'atteindre l'intégrité.
    descriptors: dict  # calculés sur candidate_graph (cf. remarque ci-dessus)
    energy_ev: float
    energy_ev_per_atom: float
    generation: int
    integrity_status: str
    topology_label: str
    has_imaginary_freq: bool
    structure_rearranged: bool | None = None
    final_positions: list | None = None  # coordonnees 3D reelles (Angstrom),
                                          # pour visualisation -- None pour les
                                          # archives generees avant cet ajout
    initial_positions: list | None = None  # geometrie AVANT optimisation
    n_optimization_steps: int | None = None  # cout du workflow pour ce candidat
    initial_xyz_path: str | None = None  # fichier .xyz reel sur disque
    optimized_xyz_path: str | None = None  # idem, geometrie optimisee
    multiseed_diagnostic: dict | None = None  # etalement d'energie entre seeds etc.
                                               # (correction #4 : conserve au lieu
                                               # d'etre jete -- alimente le score de
                                               # priorite de raffinement ORCA)
    final_graph_wl_hash: str | None = None  # identifiant canonique quasi unique de
                                             # la classe topologique (correction #3 :
                                             # les labels classify_topology sont trop
                                             # grossiers -- le hash WL sert pour toute
                                             # analyse quantitative, le label pour
                                             # l'humain). Auto-calcule si absent.

    def __post_init__(self):
        if self.final_graph_wl_hash is None and self.final_graph is not None:
            try:
                self.final_graph_wl_hash = nx.weisfeiler_lehman_graph_hash(self.final_graph)
            except Exception:
                self.final_graph_wl_hash = None


class Archive:
    def __init__(self, window_ev_per_atom: float = 0.2, energy_tol_ev: float | None = None):
        """
        energy_tol_ev : critère additionnel de dédoublonnage, DÉSACTIVÉ par
        défaut (None). Si None (défaut), deux entrées isomorphes sur
        final_graph sont TOUJOURS considérées comme le même isomère,
        quelle que soit leur différence d'énergie -- seule la meilleure
        énergie par classe topologique est conservée. C'est le comportement
        voulu pour réduire activement le cheptel de structures au fil des
        générations, pas seulement le constater après coup.

        Historique : la version précédente exigeait EN PLUS une énergie
        quasi identique (tolérance 1 meV/atome) pour fusionner -- ce qui
        laissait l'archive se remplir de quasi-doublons conformationnels
        (mesuré : 44 entrées brutes -> seulement 18 topologies réellement
        distinctes pour N11-, un facteur de gonflement de ~2.4x). Passer
        `energy_tol_ev` à une valeur numérique restaure l'ancien
        comportement (utile seulement si l'on veut explicitement garder
        plusieurs conformères distincts d'une même topologie comme des
        entrées séparées).
        """
        self.window = window_ev_per_atom
        self.energy_tol = energy_tol_ev
        self.entries: list[ArchiveEntry] = []
        self.best_energy_per_atom: float | None = None
        # Correction #5 : journal des elagues -- rend le non-determinisme
        # de l'elagage AUDITABLE (hash WL, label, energie, generation).
        self.pruned_log: list[dict] = []

    def _is_duplicate(self, entry: ArchiveEntry) -> int | None:
        """Retourne l'index de l'entrée existante isomorphe, ou None si
        l'entrée est nouvelle.

        Compare final_graph (la structure physique réelle après relaxation),
        PAS candidate_graph -- avec un taux de réarrangement mesuré de 50%+
        à n=8, deux candidats topologiquement différents relaxent souvent
        vers la MÊME structure finale ; comparer les candidats d'origine
        aurait laissé l'archive se remplir de doublons déguisés en entrées
        distinctes.

        Si final_graph est absent (évaluation échouée) sur l'un des deux,
        la comparaison est impossible -- traité comme non-doublon par
        prudence (mieux vaut un doublon occasionnel qu'un rejet à tort).
        """
        for i, existing in enumerate(self.entries):
            if existing.final_graph is None or entry.final_graph is None:
                continue
            if existing.final_graph.number_of_nodes() != entry.final_graph.number_of_nodes():
                continue
            if not nx.is_isomorphic(existing.final_graph, entry.final_graph):
                continue
            if self.energy_tol is not None:
                # Mode "conformères distincts" explicitement demandé : exige
                # en plus une énergie proche pour fusionner.
                if abs(existing.energy_ev_per_atom - entry.energy_ev_per_atom) >= self.energy_tol:
                    continue
            return i
        return None

    def add(self, entry: ArchiveEntry) -> str:
        """Ajoute une entrée si elle n'est pas un doublon structural (garde
        la version la plus basse en énergie en cas de doublon).

        Retourne un statut explicite pour permettre un diagnostic précis
        (au lieu d'un simple booléen qui confondait "nouvelle entrée" et
        "doublon mis à jour") :
          - "new"                : entrée réellement nouvelle, ajoutée
          - "duplicate_updated"   : doublon structural déjà connu, mais
                                    cette version est plus basse en énergie
                                    -> remplace l'ancienne
          - "duplicate_rejected"  : doublon structural déjà connu, moins
                                    bon ou égal -> rejeté sans effet
        """
        if self.best_energy_per_atom is None or entry.energy_ev_per_atom < self.best_energy_per_atom:
            self.best_energy_per_atom = entry.energy_ev_per_atom

        dup_idx = self._is_duplicate(entry)
        if dup_idx is not None:
            if entry.energy_ev_per_atom < self.entries[dup_idx].energy_ev_per_atom:
                self.entries[dup_idx] = entry
                return "duplicate_updated"
            return "duplicate_rejected"

        self.entries.append(entry)
        return "new"

    def prune_outside_window(self, generation: int | None = None) -> int:
        """Retire de l'archive les entrées désormais hors fenêtre par
        rapport au meilleur connu À CE JOUR. Retourne le nombre d'entrées
        retirées. Chaque entrée élaguée est tracée dans self.pruned_log
        (correction #5 : auditabilité du non-déterminisme d'élagage)."""
        if self.best_energy_per_atom is None:
            return 0
        before = len(self.entries)
        kept, pruned = [], []
        for e in self.entries:
            if (e.energy_ev_per_atom - self.best_energy_per_atom) <= self.window:
                kept.append(e)
            else:
                pruned.append(e)
        self.entries = kept
        for e in pruned:
            self.pruned_log.append({
                "wl_hash": e.final_graph_wl_hash,
                "topology_label": e.topology_label,
                "energy_ev_per_atom": e.energy_ev_per_atom,
                "pruned_at_generation": generation,
                "reference_at_pruning": self.best_energy_per_atom,
            })
        return before - len(self.entries)

    def within_window(self) -> list[ArchiveEntry]:
        if self.best_energy_per_atom is None:
            return []
        return [
            e for e in self.entries
            if (e.energy_ev_per_atom - self.best_energy_per_atom) <= self.window
        ]

    def training_data(self) -> tuple[list[dict], list[float]]:
        """Retourne (descripteurs, énergie/atome) pour TOUTES les entrées de
        l'archive (pas seulement celles dans la fenêtre) -- le surrogate a
        besoin d'exemples négatifs (hauts en énergie) autant que positifs
        pour apprendre une frontière utile."""
        X = [e.descriptors for e in self.entries]
        y = [e.energy_ev_per_atom for e in self.entries]
        return X, y

    def export_refinement_queue(self, charge: int) -> list[dict]:
        """
        Correction #4 : file de priorité pour la re-vérification à un niveau
        de théorie supérieur (ORCA/DFT). GFN2-xTB borne la fiabilité du
        classement fin ; les structures à re-vérifier EN PREMIER sont celles
        où xTB est le moins fiable :

          - fort étalement multiseed (bassins multiples -> énergie xTB
            possiblement non représentative du vrai minimum),
          - proximité de la frontière de la fenêtre (un petit décalage
            d'énergie au niveau supérieur peut les faire entrer/sortir),
          - présence d'un petit cycle (tension mal décrite en tight-binding).

        Score = somme pondérée normalisée ; tri décroissant. Retourne une
        liste de dicts consommable directement comme file d'attente (avec
        les chemins .xyz déjà sauvegardés sur disque).

        charge : une campagne polyN_adapt est menée à charge FIXÉE ;
            l'Archive elle-même ne la stocke pas (cf. run_campaign), donc
            elle doit être fournie explicitement ici -- nécessaire pour
            construire des entrées Gaussian/ORCA valides en aval
            (refinement.submit), qui ont besoin de charge ET de
            multiplicité, pas seulement de la géométrie.
        """
        if not self.entries or self.best_energy_per_atom is None:
            return []

        queue = []
        for e in self.entries:
            spread = 0.0
            if e.multiseed_diagnostic and e.multiseed_diagnostic.get("energy_spread_ev_per_atom") is not None:
                spread = float(e.multiseed_diagnostic["energy_spread_ev_per_atom"])
            dist_to_boundary = abs(
                (e.energy_ev_per_atom - self.best_energy_per_atom) - self.window
            )
            boundary_score = max(0.0, 1.0 - dist_to_boundary / self.window)
            min_cycle = e.descriptors.get("min_cycle_size", 0) or 0
            strain_score = 1.0 if 0 < min_cycle <= 4 else 0.0

            priority = 2.0 * spread + 1.0 * boundary_score + 0.5 * strain_score
            n_atoms = e.final_graph.number_of_nodes() if e.final_graph is not None else None
            queue.append({
                "priority_score": priority,
                "energy_ev_per_atom": e.energy_ev_per_atom,
                "topology_label": e.topology_label,
                "wl_hash": e.final_graph_wl_hash,
                "multiseed_spread": spread,
                "optimized_xyz_path": e.optimized_xyz_path,
                "generation": e.generation,
                "charge": charge,
                "n_atoms": n_atoms,
                "multiplicity_guess": (
                    guess_multiplicity(n_atoms, charge) if n_atoms is not None else None
                ),
            })
        queue.sort(key=lambda d: d["priority_score"], reverse=True)
        return queue

    def to_serializable(self) -> list[dict]:
        """
        Représentation JSON-compatible de l'archive pour sauvegarde disque
        (une campagne de production ne doit jamais rester uniquement en
        mémoire). Les graphes networkx sont encodés en graph6 (compact,
        reconstructible via nx.from_graph6_bytes), pas les objets Python
        eux-mêmes.
        """
        out = []
        for e in self.entries:
            out.append({
                "candidate_graph_g6": nx.to_graph6_bytes(e.candidate_graph, header=False).decode().strip(),
                "final_graph_g6": (
                    nx.to_graph6_bytes(e.final_graph, header=False).decode().strip()
                    if e.final_graph is not None else None
                ),
                "descriptors": e.descriptors,
                "energy_ev": e.energy_ev,
                "energy_ev_per_atom": e.energy_ev_per_atom,
                "generation": e.generation,
                "integrity_status": e.integrity_status,
                "topology_label": e.topology_label,
                "has_imaginary_freq": e.has_imaginary_freq,
                "structure_rearranged": e.structure_rearranged,
                "final_positions": e.final_positions,
                "initial_positions": e.initial_positions,
                "n_optimization_steps": e.n_optimization_steps,
                "initial_xyz_path": e.initial_xyz_path,
                "optimized_xyz_path": e.optimized_xyz_path,
                "multiseed_diagnostic": e.multiseed_diagnostic,
                "final_graph_wl_hash": e.final_graph_wl_hash,
            })
        return out

    def __len__(self) -> int:
        return len(self.entries)
