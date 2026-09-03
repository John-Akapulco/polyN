"""
Surrogate adaptatif à TROIS volets (correction #1), ré-entraîné à chaque
population sur l'archive accumulée :

  - un RÉGRESSEUR (gradient boosting) : descripteurs du graphe CANDIDAT ->
    énergie/atome finale. C'est le filtre de rejet. Sémantique assumée :
    avec ~50%+ de réarrangement structural mesuré, il prédit "ce que
    devient énergétiquement ce point de départ", PAS "l'énergie de cette
    topologie" -- c'est exactement ce qu'il faut pour décider d'évaluer.

  - un CLASSIFICATEUR DE RÉARRANGEMENT (gradient boosting binaire) :
    descripteurs du graphe candidat -> probabilité que la structure change
    de topologie en relaxant. Gratuit en données (structure_rearranged est
    déjà archivé), il rend explicite le bruit qui brouille le régresseur :
    un candidat à forte proba de réarrangement a une prédiction d'énergie
    moins fiable.

  - une RÈGLE INTERPRÉTABLE (arbre peu profond) entraînée sur les
    descripteurs des graphes FINAUX relaxés (recalculés depuis final_graph,
    déjà archivé) -> énergie. Correction du défaut sémantique précédent :
    entraînée sur les candidats, ses "motifs stabilisants" étaient des
    motifs de bons POINTS DE DÉPART ; entraînée sur les finaux, ce sont des
    motifs de STRUCTURES STABLES -- ce qu'on veut pour le futur mineur de
    motifs. Elle n'est jamais utilisée pour filtrer (au moment du filtre,
    le graphe final n'existe pas encore).

Apprentissage actif : au lieu d'un simple seuil accepter/rejeter, priorise
l'évaluation réelle sur les candidats dont la prédiction est la plus
incertaine près de la frontière de la fenêtre -- affine la frontière plus
vite avec le même budget de calculs xTB.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor, export_text

from ..core.graph_descriptors import DEFAULT_FEATURE_ORDER, compute_descriptors


MIN_TRAINING_SAMPLES = 20  # sous ce seuil, le surrogate n'est pas encore fiable
                            # (population 1 : pas de surrogate du tout, cf. design)


@dataclass
class SurrogatePrediction:
    energy_per_atom_pred: float
    uncertainty: float  # écart-type inter-arbres du GradientBoostingRegressor
    accept: bool


class AdaptiveSurrogate:
    def __init__(self, window_ev_per_atom: float = 0.2, feature_order: list[str] | None = None,
                 n_estimators: int = 100, max_depth_rule: int = 3, random_state: int = 0):
        self.window = window_ev_per_atom
        self.feature_order = feature_order or DEFAULT_FEATURE_ORDER
        self.n_estimators = n_estimators
        self.max_depth_rule = max_depth_rule
        self.random_state = random_state
        self.regressor: GradientBoostingRegressor | None = None
        self.rule: DecisionTreeRegressor | None = None
        self.is_trained = False
        # Correction #1 : deux modèles supplémentaires, entraînés par
        # fit_from_archive (fit classique ne les touche pas -- il ne reçoit
        # pas les données nécessaires).
        self.rearrangement_classifier: GradientBoostingClassifier | None = None
        self.rearrangement_trained = False
        self.final_rule: DecisionTreeRegressor | None = None  # règle sur graphes FINAUX
        self.final_rule_trained = False

    def _vectorize(self, descriptors_list: list[dict]) -> np.ndarray:
        return np.array([[d[k] for k in self.feature_order] for d in descriptors_list])

    def fit(self, descriptors_list: list[dict], energies_per_atom: list[float]) -> bool:
        """Entraîne les deux modèles. Retourne False (surrogate non entraîné,
        filtre inactif) si l'archive est encore trop petite pour être fiable."""
        if len(descriptors_list) < MIN_TRAINING_SAMPLES:
            self.is_trained = False
            return False

        X = self._vectorize(descriptors_list)
        y = np.array(energies_per_atom)

        self.regressor = GradientBoostingRegressor(
            n_estimators=self.n_estimators, max_depth=3, random_state=self.random_state,
        )
        self.regressor.fit(X, y)

        self.rule = DecisionTreeRegressor(max_depth=self.max_depth_rule, random_state=self.random_state)
        self.rule.fit(X, y)

        self.is_trained = True
        return True

    def _uncertainty(self, x_row: np.ndarray) -> float:
        """Écart-type des prédictions des arbres individuels du gradient
        boosting -- proxy simple de l'incertitude, sans modèle bayésien
        dédié (suffisant pour prioriser l'apprentissage actif)."""
        staged = np.array([pred[0] for pred in self.regressor.staged_predict(x_row.reshape(1, -1))])
        return float(np.std(np.diff(staged))) if len(staged) > 1 else 0.0

    def predict(self, descriptors: dict, best_known_energy_per_atom: float) -> SurrogatePrediction:
        if not self.is_trained:
            # Surrogate pas encore fiable : on accepte tout, comme en population 1
            return SurrogatePrediction(energy_per_atom_pred=float("nan"), uncertainty=float("inf"), accept=True)

        x = np.array([[descriptors[k] for k in self.feature_order]])
        pred = float(self.regressor.predict(x)[0])
        unc = self._uncertainty(x[0])
        accept = (pred - best_known_energy_per_atom) <= self.window
        return SurrogatePrediction(energy_per_atom_pred=pred, uncertainty=unc, accept=accept)

    def select_batch(self, candidates: list[dict], best_known_energy_per_atom: float,
                       budget: int, active_learning: bool = True,
                       uncertainty_margin: float = 0.05) -> list[int]:
        """
        Sélectionne, parmi les candidats, les indices à envoyer à l'évaluation
        réelle (xTB/DFTB+), sous un budget fixe.

        Si active_learning=True : priorise les candidats dont la prédiction
        tombe DANS la fenêtre OU près de sa frontière (incertitude ou marge
        de distance faible à la frontière) plutôt que tous les acceptés
        indistinctement -- affine la frontière plus vite avec le même budget.
        """
        if not self.is_trained:
            # Pas de surrogate encore entraîné : tout le monde passe (dans la
            # limite du budget, ordre d'arrivée -- population 1 typiquement).
            return list(range(min(budget, len(candidates))))

        predictions = [self.predict(c, best_known_energy_per_atom) for c in candidates]

        if not active_learning:
            accepted = [i for i, p in enumerate(predictions) if p.accept]
            return accepted[:budget]

        # Score de priorité : distance à la frontière de la fenêtre (plus
        # proche de 0 = plus près de la frontière = plus informatif),
        # combinée à l'incertitude du modèle sur ce point.
        def priority(p: SurrogatePrediction) -> float:
            dist_to_boundary = abs((p.energy_per_atom_pred - best_known_energy_per_atom) - self.window)
            return dist_to_boundary - uncertainty_margin * p.uncertainty  # incertitude haute -> priorite plus haute (score plus bas)

        # On garde d'abord tous les acceptés (dans la fenêtre prédite),
        # puis on complète avec les plus incertains/proches de la frontière
        # parmi les rejetés, jusqu'au budget.
        accepted_idx = [i for i, p in enumerate(predictions) if p.accept]
        rejected_idx = [i for i, p in enumerate(predictions) if not p.accept]
        rejected_idx.sort(key=lambda i: priority(predictions[i]))

        selected = accepted_idx[:budget]
        remaining = budget - len(selected)
        if remaining > 0:
            selected += rejected_idx[:remaining]
        return selected

    def rule_summary(self) -> str:
        """Représentation textuelle lisible de la règle interprétable
        entraînée sur les graphes CANDIDATS -- attention à la sémantique :
        ses motifs sont des motifs de bons POINTS DE DÉPART, pas de
        structures stables (cf. docstring du module). Pour les motifs de
        structures stables, voir final_rule_summary()."""
        if self.rule is None:
            return "(surrogate non entraîné)"
        return export_text(self.rule, feature_names=self.feature_order)

    # ------------------------------------------------------------------
    # Correction #1 : entraînement enrichi depuis l'archive complète
    # ------------------------------------------------------------------

    def fit_from_archive(self, archive) -> bool:
        """
        Entraîne les trois modèles à partir de l'archive (duck typing :
        tout objet exposant .entries avec les attributs d'ArchiveEntry
        convient). Remplace l'appel fit(X, y) dans la boucle de population.

        1. Régresseur + règle candidat : identiques à fit() -- le filtre ne
           change pas de comportement.
        2. Classificateur de réarrangement : descripteurs candidats ->
           structure_rearranged (booléen archivé). Entraîné seulement si
           les deux classes sont représentées (sinon rien à apprendre).
        3. Règle interprétable FINALE : descripteurs recalculés sur
           final_graph -> énergie. Zéro appel xTB : les graphes finaux
           sont déjà en mémoire.

        Retourne le booléen de fit() (filtre actif ou non) ; les deux
        modèles supplémentaires sont entraînés au mieux, sans jamais
        bloquer le filtre principal.
        """
        entries = archive.entries
        X_cand = [e.descriptors for e in entries]
        y_energy = [e.energy_ev_per_atom for e in entries]

        filter_trained = self.fit(X_cand, y_energy)

        # --- Classificateur de réarrangement (candidat -> proba) ---
        labeled = [(e.descriptors, bool(e.structure_rearranged))
                   for e in entries if e.structure_rearranged is not None]
        self.rearrangement_trained = False
        if len(labeled) >= MIN_TRAINING_SAMPLES:
            X_r = self._vectorize([d for d, _ in labeled])
            y_r = np.array([lab for _, lab in labeled], dtype=int)
            if len(np.unique(y_r)) == 2:  # les deux classes présentes
                self.rearrangement_classifier = GradientBoostingClassifier(
                    n_estimators=self.n_estimators, max_depth=3,
                    random_state=self.random_state,
                )
                self.rearrangement_classifier.fit(X_r, y_r)
                self.rearrangement_trained = True

        # --- Règle interprétable sur les graphes FINAUX relaxés ---
        finals = [(e.final_graph, e.energy_ev_per_atom)
                  for e in entries if e.final_graph is not None]
        self.final_rule_trained = False
        if len(finals) >= MIN_TRAINING_SAMPLES:
            try:
                final_descs = [compute_descriptors(g).as_dict() for g, _ in finals]
                X_f = self._vectorize(final_descs)
                y_f = np.array([en for _, en in finals])
                self.final_rule = DecisionTreeRegressor(
                    max_depth=self.max_depth_rule, random_state=self.random_state,
                )
                self.final_rule.fit(X_f, y_f)
                self.final_rule_trained = True
            except Exception:
                # un graphe final pathologique ne doit jamais bloquer le filtre
                self.final_rule_trained = False

        return filter_trained

    def predict_rearrangement(self, descriptors: dict) -> float | None:
        """Probabilité prédite que ce candidat change de topologie en
        relaxant. None si le classificateur n'est pas (encore) entraîné.
        Usage : pondérer la confiance dans la prédiction d'énergie (une
        proba haute = étiquette d'entraînement probablement issue d'une
        autre structure que celle décrite par les descripteurs)."""
        if not self.rearrangement_trained:
            return None
        x = np.array([[descriptors[k] for k in self.feature_order]])
        return float(self.rearrangement_classifier.predict_proba(x)[0, 1])

    def final_rule_summary(self) -> str:
        """Règle interprétable entraînée sur les descripteurs des graphes
        FINAUX relaxés : ses motifs sont des motifs de STRUCTURES STABLES
        -- c'est celle-ci qu'il faut lire pour comprendre la stabilité, et
        celle qui alimentera le futur mineur de motifs."""
        if not self.final_rule_trained or self.final_rule is None:
            return "(règle finale non entraînée -- archive trop petite ou graphes finaux absents)"
        return export_text(self.final_rule, feature_names=self.feature_order)
