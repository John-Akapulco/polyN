"""
Point de sauvegarde / reprise pour run_campaign et run_campaign_mutation
(population_loop.py).

Granularite : PAR GENERATION COMPLETE (pas par candidat individuel a
l'interieur d'une generation). Choix delibere plutot qu'un raffinement
plus fin : la lenteur anormale qui a motive ce module (campagne N15- anion,
job SLURM 59926 -- ~30 candidats evalues en 13h30 au lieu de quelques
minutes attendues) venait d'une sursouscription OpenMP (cf. _pin_single_thread
dans population_loop.py), pas d'un cout intrinseque par candidat. Une fois
corrigee, une generation complete (jusqu'a budget x len(multiseed_seeds)
evaluations xTB) prend typiquement quelques minutes avec 32 workers -- un
redemarrage qui refait la generation en cours depuis le debut reste bon
marche, ce qui rend inutile la complexite d'un checkpoint intra-generation
(persister les candidats deja evalues, resynchroniser l'executor, etc.)
pour un gain marginal.

Format sur disque : un seul fichier JSON par campagne (pas un fichier par
generation) -- ecriture atomique (fichier temporaire + os.replace) pour ne
jamais laisser un checkpoint corrompu si le job est tue en plein milieu de
l'ecriture.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import networkx as nx

from ..archive.archive import Archive, ArchiveEntry


def _entry_from_dict(d: dict) -> ArchiveEntry:
    candidate_graph = nx.from_graph6_bytes(d["candidate_graph_g6"].encode())
    final_graph = (
        nx.from_graph6_bytes(d["final_graph_g6"].encode())
        if d.get("final_graph_g6") is not None else None
    )
    return ArchiveEntry(
        candidate_graph=candidate_graph,
        final_graph=final_graph,
        descriptors=d["descriptors"],
        energy_ev=d["energy_ev"],
        energy_ev_per_atom=d["energy_ev_per_atom"],
        generation=d["generation"],
        integrity_status=d["integrity_status"],
        topology_label=d["topology_label"],
        has_imaginary_freq=d["has_imaginary_freq"],
        structure_rearranged=d.get("structure_rearranged"),
        final_positions=d.get("final_positions"),
        initial_positions=d.get("initial_positions"),
        n_optimization_steps=d.get("n_optimization_steps"),
        initial_xyz_path=d.get("initial_xyz_path"),
        optimized_xyz_path=d.get("optimized_xyz_path"),
        multiseed_diagnostic=d.get("multiseed_diagnostic"),
        final_graph_wl_hash=d.get("final_graph_wl_hash"),
    )


def _archive_from_serializable(entries: list[dict], window_ev_per_atom: float) -> Archive:
    """Reconstruit une Archive en rejouant Archive.add() sur chaque entree,
    dans l'ordre d'origine -- garantit exactement le meme etat (meilleure
    energie, doublons resolus) qu'obtenu en direct, plutot qu'une simple
    desererialisation champ a champ qui pourrait diverger si la logique de
    dedoublonnage evolue."""
    archive = Archive(window_ev_per_atom=window_ev_per_atom)
    for d in entries:
        archive.add(_entry_from_dict(d))
    return archive


def save_checkpoint(path: str | Path, archive: Archive, reports: list, last_completed_generation: int) -> None:
    """Ecriture atomique : jamais de fichier tronque/corrompu si le
    processus est tue pendant l'ecriture (SLURM timeout, scancel, OOM)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "last_completed_generation": last_completed_generation,
        "window_ev_per_atom": archive.window,
        "archive_entries": archive.to_serializable(),
        "pruned_log": archive.pruned_log,
        "reports": [r.__dict__ for r in reports],
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload))
    os.replace(tmp_path, path)


def load_checkpoint(path: str | Path):
    """Retourne (archive, reports_as_dicts, last_completed_generation) si un
    checkpoint existe, sinon None. reports_as_dicts est une liste de dict
    (pas de GenerationReport -- reconstruction laissee a l'appelant, qui
    connait le type exact importe depuis population_loop) pour eviter un
    import circulaire checkpoint.py <-> population_loop.py."""
    path = Path(path)
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    archive = _archive_from_serializable(payload["archive_entries"], payload["window_ev_per_atom"])
    archive.pruned_log = payload.get("pruned_log", [])
    return archive, payload["reports"], payload["last_completed_generation"]


def checkpoint_path(work_dir: str | Path) -> Path:
    return Path(work_dir) / "checkpoint.json"
