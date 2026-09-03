"""
Configuration du raffinement post-xTB, entièrement pilotée par un fichier
YAML (cf. refinement_config_example.yaml) -- aucun paramètre de calcul
(méthode ab initio/DFT/CCSD(T), base, type de job) n'est codé en dur ici
ni dans les backends : ce module se contente de charger et valider la
structure attendue.

Les étapes (CalculationStep) sont chaînées LINÉAIREMENT dans l'ordre de
la liste -- ce module ne résout pas un graphe de dépendances arbitraire.
`depends_on`, quand fourni, sert à la validation/lisibilité : il doit
désigner l'étape immédiatement précédente, sinon la config est rejetée.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

VALID_BACKENDS = ("gaussian", "orca")
VALID_JOB_TYPES = ("geometry_optimization", "frequency", "single_point")


@dataclass
class CalculationStep:
    name: str
    job_type: str  # cf. VALID_JOB_TYPES
    method: str  # ex: "B3LYP", "wB97X-D3", "MP2", "DLPNO-CCSD(T)"
    basis: str  # ex: "def2-TZVP", "cc-pVTZ"
    depends_on: str | None = None  # doit etre l'etape immediatement precedente, ou None
    aux_basis: str | None = None  # base auxiliaire RIJCOSX/DLPNO (ORCA) -- ignoree par le backend gaussian
    extra_keywords: list[str] = field(default_factory=list)  # tokens passes tels quels au backend (ex: "TightSCF", "Int=UltraFine")
    multiplicity_override: int | None = None  # None = utilise la valeur devinee/fournie pour la structure

    def __post_init__(self):
        if self.job_type not in VALID_JOB_TYPES:
            raise ValueError(
                f"etape '{self.name}': job_type invalide '{self.job_type}' "
                f"-- attendu parmi {VALID_JOB_TYPES}"
            )


@dataclass
class Resources:
    n_proc: int = 1
    memory_mb: int = 4000


@dataclass
class SubmissionSettings:
    mode: str = "dry_run"  # "dry_run" (ecrit les inputs seulement) | "command" (genere aussi les commandes d'execution dans run.sh)
    command_template: str | None = None  # ex: "orca {input} > {output}" ou "g16 < {input} > {output}" -- {input}/{output} substitues par etape

    def __post_init__(self):
        if self.mode not in ("dry_run", "command"):
            raise ValueError(f"submission.mode invalide '{self.mode}' -- attendu 'dry_run' ou 'command'")
        if self.mode == "command" and not self.command_template:
            raise ValueError("submission.mode='command' requiert submission.command_template")


@dataclass
class RefinementConfig:
    backend: str
    steps: list[CalculationStep]
    resources: Resources = field(default_factory=Resources)
    submission: SubmissionSettings = field(default_factory=SubmissionSettings)

    def __post_init__(self):
        if self.backend not in VALID_BACKENDS:
            raise ValueError(f"backend invalide '{self.backend}' -- attendu parmi {VALID_BACKENDS}")
        if not self.steps:
            raise ValueError("au moins une etape de calcul est requise (config.steps)")

        names = [s.name for s in self.steps]
        if len(names) != len(set(names)):
            raise ValueError("noms d'etape dupliques dans la configuration")

        for i, s in enumerate(self.steps):
            if s.depends_on is None:
                continue
            expected = self.steps[i - 1].name if i > 0 else None
            if s.depends_on != expected:
                raise ValueError(
                    f"etape '{s.name}': depends_on='{s.depends_on}' doit etre "
                    f"l'etape immediatement precedente ('{expected}') -- ce "
                    f"module chaine les etapes lineairement dans l'ordre de "
                    f"la liste, il ne resout pas un graphe de dependances "
                    f"arbitraire."
                )


def load_config(path: str | Path) -> RefinementConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    steps = [CalculationStep(**s) for s in raw["steps"]]
    resources = Resources(**raw.get("resources", {}))
    submission = SubmissionSettings(**raw.get("submission", {}))
    return RefinementConfig(
        backend=raw["backend"], steps=steps, resources=resources, submission=submission,
    )
