"""
Correction multi-fidelite pour le surrogate adaptatif (pool 2) : au lieu de
filtrer les candidats sur la seule energie GFN2-xTB brute, corrige la
prediction vers une estimation d'energie a plus haut niveau (DFT, voire
CCSD(T)) a partir des paires deja accumulees par la campagne DFT/CCSD(T)
manuelle (orca_jobs/). Objectif : cibler plus efficacement les groupes
(N, charge) identifies comme mal couverts par le pool 1 (voir le rapport
de campagne, section "Couverture du criblage et pistes de complement").

Resultats de validation croisee (5-fold, GradientBoostingRegressor sur les
11 descripteurs de core.graph_descriptors vs. une regression lineaire sur
n_atoms seul) sur le jeu de donnees exporte le 2026-09-06 -- MITIGES, pas
un gain systematique :

  Delta(DFT - xTB), par famille (n = nombre de structures) :
    neutre  (n=68) : GBR PIRE que la baseline n_atoms seul (-11%)
    cation  (n=26) : GBR MEILLEUR que la baseline (+58%)
    anion   (n=46) : GBR PIRE que la baseline (-4%)

  Delta(CCSD(T) - DFT), toutes familles regroupees (n=29, ~8-11 par
  famille -- trop peu pour separer) : GBR PIRE que la baseline (-37%),
  net signe de surapprentissage (11 descripteurs pour <30 points).

Interpretation : les 11 descripteurs purement topologiques (connectivite
a 1,90 A, sans ordre de liaison, sans charge explicite, sans groupe
ponctuel) ne capturent un signal utile au-dela du simple nombre d'atomes
que pour les cations. Pour les neutres/anions et pour CCSD(T)-DFT (jeu
encore trop petit), la baseline lineaire est la meilleure estimation
disponible actuellement -- ne PAS forcer un modele plus riche qu'un jeu
de donnees ne le supporte. D'ou la selection automatique ci-dessous :
FidelityCorrector choisit, par famille et par CV, le meilleur des deux
plutot que d'imposer le gradient boosting partout.

A ameliorer plus tard (pas fait ici, jeu de donnees encore trop petit ou
descripteurs insuffisants) : ajouter charge/multiplicite et un invariant
d'ordre de liaison (Mayer) aux descripteurs, ce qui manque probablement
pour capturer la variance neutre/anion ; ré-évaluer des que le jeu
CCSD(T)-DFT grossit (campagne en cours, mise a jour automatique).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict

MIN_SAMPLES_FOR_GBR_ATTEMPT = 20  # sous ce seuil, GBR n'est meme pas essaye (cf. AdaptiveSurrogate)
CV_FOLDS = 5


@dataclass
class CorrectorDiagnostics:
    n_samples: int
    baseline_mae_ev: float
    gbr_mae_ev: float | None  # None si pas assez de points pour tenter GBR
    chosen_model: str  # "gbr" ou "baseline_linear" ou "constant"


class FidelityCorrector:
    """Corrige une energie/atome de fidelite basse (xTB, ou DFT) vers une
    estimation de fidelite haute (DFT, ou CCSD(T)) : predict_delta(x) ->
    delta a ajouter. Choix automatique, par validation croisee, entre
    gradient boosting sur les descripteurs complets et une simple
    regression lineaire sur n_atoms -- ne jamais imposer un modele plus
    riche que ce que le jeu de donnees supporte (voir docstring module)."""

    def __init__(self, feature_order: list[str], n_estimators: int = 100,
                 max_depth: int = 3, random_state: int = 0):
        self.feature_order = feature_order
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None
        self.chosen_model: str | None = None
        self.diagnostics: CorrectorDiagnostics | None = None
        self.is_trained = False

    def _vectorize(self, descriptors_list: list[dict]) -> np.ndarray:
        return np.array([[d[k] for k in self.feature_order] for d in descriptors_list])

    def fit(self, descriptors_list: list[dict], deltas_ev_per_atom: list[float]) -> CorrectorDiagnostics:
        X = self._vectorize(descriptors_list)
        y = np.array(deltas_ev_per_atom)
        n = len(y)

        if n < 8:
            # Trop peu de points pour toute validation croisee fiable :
            # la seule estimation defendable est la moyenne (correction
            # constante), pas un modele conditionne sur des features.
            self.model = float(np.mean(y))
            self.chosen_model = "constant"
            self.is_trained = True
            self.diagnostics = CorrectorDiagnostics(n, float(np.mean(np.abs(y - np.mean(y)))), None, "constant")
            return self.diagnostics

        n_splits = min(CV_FOLDS, n)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        X_natoms = X[:, [self.feature_order.index("n_atoms")]]
        base_pred = cross_val_predict(LinearRegression(), X_natoms, y, cv=kf)
        base_mae = float(np.mean(np.abs(base_pred - y)))

        gbr_mae = None
        if n >= MIN_SAMPLES_FOR_GBR_ATTEMPT:
            gbr = GradientBoostingRegressor(n_estimators=self.n_estimators,
                                             max_depth=self.max_depth,
                                             random_state=self.random_state)
            gbr_pred = cross_val_predict(gbr, X, y, cv=kf)
            gbr_mae = float(np.mean(np.abs(gbr_pred - y)))

        if gbr_mae is not None and gbr_mae < base_mae:
            self.model = GradientBoostingRegressor(n_estimators=self.n_estimators,
                                                     max_depth=self.max_depth,
                                                     random_state=self.random_state)
            self.model.fit(X, y)
            self.chosen_model = "gbr"
        else:
            self.model = LinearRegression()
            self.model.fit(X_natoms, y)
            self.chosen_model = "baseline_linear"

        self.is_trained = True
        self.diagnostics = CorrectorDiagnostics(n, base_mae, gbr_mae, self.chosen_model)
        return self.diagnostics

    def predict_delta(self, descriptors: dict) -> float:
        if not self.is_trained:
            raise RuntimeError("FidelityCorrector.fit() n'a pas encore ete appele")
        if self.chosen_model == "constant":
            return self.model
        x = np.array([[descriptors[k] for k in self.feature_order]])
        if self.chosen_model == "baseline_linear":
            x = x[:, [self.feature_order.index("n_atoms")]]
        return float(self.model.predict(x)[0])
