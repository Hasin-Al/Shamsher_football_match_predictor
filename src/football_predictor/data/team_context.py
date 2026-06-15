from __future__ import annotations

import json
import math
import re
from collections import deque
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ..config import ProjectPaths
from .news import GoogleNewsRssClient
from .name_normalization import canonical_person_name, canonical_team_name


NEWS_VECTOR_DIM = 128
SQUAD_SIZE = 18
SQUAD_FEATURE_NAMES = [
    "start_prob",
    "fitness",
    "recent_start_rate",
    "recent_minutes_avg",
    "lineup_continuity",
    "importance",
    "unavailable_proxy",
]


@dataclass(slots=True)
class PlayerAvailabilityState:
    player_id: int
    player_name: str
    minutes_history: deque[float] = field(default_factory=lambda: deque(maxlen=6))
    start_history: deque[float] = field(default_factory=lambda: deque(maxlen=6))
    last_seen_match_index: int | None = None
    last_start_match_index: int | None = None
    total_appearances: int = 0


def normalize_name(value: str) -> str:
    return canonical_person_name(value)


def hash_news_texts(texts: list[str], dim: int = NEWS_VECTOR_DIM) -> torch.Tensor:
    vector = np.zeros(dim, dtype=np.float32)
    token_pattern = re.compile(r"[a-zA-Z][a-zA-Z0-9'-]+")
    for text in texts:
        tokens = token_pattern.findall(text.lower())
        for token in tokens:
            bucket = hash(token) % dim
            sign = 1.0 if (hash(f"{token}_sign") % 2 == 0) else -1.0
            vector[bucket] += sign
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector /= norm
    return torch.tensor(vector, dtype=torch.float32)


def load_manual_team_context(paths: ProjectPaths) -> dict[str, Any]:
    manual_path = paths.manual_dir / "team_context.json"
    payload = {}
    if manual_path.exists():
        payload = json.loads(manual_path.read_text(encoding="utf-8"))
    merged = {canonical_team_name(team_name): team_payload for team_name, team_payload in payload.items()}

    live_context_paths = [
        paths.external_dir / "live_context",
        Path("new_data/football_prediction_dataset/data/external/live_context"),
    ]
    injuries_by_team: dict[str, list[dict[str, Any]]] = defaultdict(list)
    lineup_by_team: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for live_dir in live_context_paths:
        injury_path = live_dir / "injury_candidates.csv"
        if injury_path.exists():
            frame = pd.read_csv(injury_path)
            for row in frame.to_dict("records"):
                team_key = canonical_team_name(str(row.get("team", "")))
                player_name = str(row.get("player_name", "")).strip()
                if not team_key or not player_name:
                    continue
                injuries_by_team[team_key].append(
                    {
                        "player": player_name,
                        "status": str(row.get("status", "unknown")).strip().lower(),
                        "source": str(row.get("source", "structured-live")),
                    }
                )
        predicted_path = live_dir / "predicted_lineups.csv"
        if predicted_path.exists():
            frame = pd.read_csv(predicted_path)
            for row in frame.to_dict("records"):
                team_key = canonical_team_name(str(row.get("team", "")))
                player_name = str(row.get("player_name", "")).strip()
                if not team_key or not player_name:
                    continue
                lineup_by_team[team_key].append(
                    {
                        "player": player_name,
                        "start_probability": 0.9,
                        "source": str(row.get("source", "structured-live")),
                    }
                )

    for team_key, injuries in injuries_by_team.items():
        team_entry = merged.setdefault(team_key, {})
        existing = team_entry.setdefault("injuries", [])
        seen = {normalize_name(str(row.get("player", ""))) for row in existing}
        for row in injuries:
            if normalize_name(str(row.get("player", ""))) not in seen:
                existing.append(row)

    for team_key, lineup in lineup_by_team.items():
        team_entry = merged.setdefault(team_key, {})
        existing = team_entry.setdefault("probable_xi", [])
        seen = {normalize_name(str(row.get("player", ""))) for row in existing}
        for row in lineup:
            if normalize_name(str(row.get("player", ""))) not in seen:
                existing.append(row)

    return merged


def default_manual_context_template() -> dict[str, Any]:
    return {
        "England": {
            "aliases": ["England national team", "Three Lions"],
            "coach": {
                "name": "Gareth Southgate",
                "known_match_count": 102,
            },
            "tactical_overrides": {
                "pass_completion_rate": 0.89,
                "shots": 14.2,
                "xg": 1.75,
                "progressive_passes": 31.0,
                "final_third_entries": 49.0,
                "direct_speed_proxy": 1.08,
                "pressing_intensity_proxy": 0.91,
            },
            "injuries": [
                {"player": "Bukayo Saka", "status": "fit", "fitness": 92, "source": "staff-report"},
                {"player": "John Stones", "status": "doubtful", "fitness": 58, "source": "press-conference"},
            ],
            "probable_xi": [
                {"player": "Bukayo Saka", "start_probability": 0.94},
                {"player": "Declan Rice", "start_probability": 0.99},
            ],
            "bench": [
                {"player": "Cole Palmer", "start_probability": 0.35},
            ],
        }
    }


def ensure_manual_context_template(paths: ProjectPaths) -> None:
    template_path = paths.manual_dir / "team_context.example.json"
    if not template_path.exists():
        template_path.write_text(json.dumps(default_manual_context_template(), indent=2), encoding="utf-8")


def get_player_state_map(team_player_states: dict[int, PlayerAvailabilityState], team_profile: dict[str, Any] | None) -> dict[int, PlayerAvailabilityState]:
    states = dict(team_player_states)
    if team_profile:
        for player_id, player_name in zip(team_profile.get("player_ids", []), team_profile.get("player_names", []), strict=False):
            if player_id == -1:
                continue
            states.setdefault(int(player_id), PlayerAvailabilityState(player_id=int(player_id), player_name=str(player_name)))
    return states


def build_squad_matrix(
    team_player_states: dict[int, PlayerAvailabilityState],
    match_index: int,
    team_profile: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, list[int], list[str]]:
    states = list(get_player_state_map(team_player_states, team_profile).values())
    scored: list[tuple[float, PlayerAvailabilityState]] = []
    for state in states:
        recent_minutes = float(np.mean(state.minutes_history)) if state.minutes_history else 0.0
        recent_starts = float(np.mean(state.start_history)) if state.start_history else 0.0
        absence = 99 if state.last_seen_match_index is None else max(0, match_index - state.last_seen_match_index)
        continuity = 1.0 if state.last_start_match_index is not None and match_index - state.last_start_match_index <= 1 else 0.0
        importance = min(1.0, (recent_minutes / 90.0) * (1.0 + min(state.total_appearances, 10) / 10.0) / 2.0)
        score = recent_minutes + recent_starts * 30.0 + importance * 20.0 - absence
        scored.append((score, state))
    scored.sort(key=lambda item: item[0], reverse=True)
    top_states = [state for _, state in scored[:SQUAD_SIZE]]

    rows: list[list[float]] = []
    player_ids: list[int] = []
    player_names: list[str] = []
    for state in top_states:
        recent_minutes = float(np.mean(state.minutes_history)) if state.minutes_history else 0.0
        recent_starts = float(np.mean(state.start_history)) if state.start_history else 0.0
        absence = 99 if state.last_seen_match_index is None else max(0, match_index - state.last_seen_match_index)
        fitness = max(0.0, 1.0 - min(absence, 45) / 45.0)
        continuity = 1.0 if state.last_start_match_index is not None and match_index - state.last_start_match_index <= 1 else 0.0
        importance = min(1.0, (recent_minutes / 90.0) * (1.0 + min(state.total_appearances, 10) / 10.0) / 2.0)
        start_prob = min(1.0, recent_starts * (0.65 + 0.35 * fitness) * (0.8 + 0.2 * continuity))
        unavailable_proxy = 1.0 if importance > 0.45 and absence >= 3 else 0.0
        rows.append(
            [
                start_prob,
                fitness,
                recent_starts,
                recent_minutes / 90.0,
                continuity,
                importance,
                unavailable_proxy,
            ]
        )
        player_ids.append(state.player_id)
        player_names.append(state.player_name)

    while len(rows) < SQUAD_SIZE:
        rows.append([0.0] * len(SQUAD_FEATURE_NAMES))
        player_ids.append(-1)
        player_names.append("Unknown")
    return torch.tensor(rows, dtype=torch.float32), player_ids, player_names


def update_player_availability(
    team_player_states: dict[int, PlayerAvailabilityState],
    match_index: int,
    starters: set[int],
    participants: set[int],
    minutes_by_player: dict[int, float],
    player_names: dict[int, str],
) -> None:
    for player_id in participants:
        state = team_player_states.setdefault(
            int(player_id),
            PlayerAvailabilityState(player_id=int(player_id), player_name=player_names.get(int(player_id), "Unknown")),
        )
        state.player_name = player_names.get(int(player_id), state.player_name)
        state.total_appearances += 1
        state.last_seen_match_index = match_index
        minutes = float(minutes_by_player.get(int(player_id), 0.0))
        state.minutes_history.append(minutes)
        started = 1.0 if int(player_id) in starters else 0.0
        state.start_history.append(started)
        if started:
            state.last_start_match_index = match_index


def extract_lineup_player_sets(lineups: list[dict[str, Any]], team_name: str) -> tuple[set[int], set[int], dict[int, str]]:
    starters: set[int] = set()
    participants: set[int] = set()
    player_names: dict[int, str] = {}
    team_key = canonical_team_name(team_name)
    for team_block in lineups:
        if canonical_team_name(str(team_block.get("team_name", ""))) != team_key:
            continue
        for player in team_block.get("lineup", []):
            raw_player_id = player.get("player_id")
            if raw_player_id is None:
                continue
            player_id = int(raw_player_id)
            player_names[player_id] = str(player.get("player_name", "Unknown"))
            participants.add(player_id)
            positions = player.get("positions") or []
            if positions:
                first_position = positions[0] or {}
                if first_position.get("start_reason") == "Starting XI":
                    starters.add(player_id)
    return starters, participants, player_names


def collect_team_news_vectors(
    home_team: str,
    away_team: str,
    manual_context: dict[str, Any],
    limit_per_query: int = 6,
) -> tuple[torch.Tensor, torch.Tensor, list[str], list[str]]:
    client = GoogleNewsRssClient()

    def collect(team_name: str) -> tuple[torch.Tensor, list[str]]:
        team_key = canonical_team_name(team_name)
        team_context = manual_context.get(team_key, {})
        queries = [team_name]
        queries.extend(team_context.get("aliases", []))
        queries = [query for query in queries if query]
        titles: list[str] = []
        try:
            items = client.multi_search(queries[:3], limit_per_query=limit_per_query)
            titles = [item.title for item in items if item.title]
        except Exception:
            titles = []
        return hash_news_texts(titles), titles

    home_vector, home_titles = collect(home_team)
    away_vector, away_titles = collect(away_team)
    return home_vector, away_vector, home_titles, away_titles


def apply_manual_player_context(team_profile: dict[str, Any], squad_x: torch.Tensor, team_context: dict[str, Any]) -> torch.Tensor:
    adjusted = squad_x.clone()
    name_to_index = {normalize_name(name): idx for idx, name in enumerate(team_profile.get("squad_player_names", []))}

    for row in team_context.get("probable_xi", []):
        player_key = normalize_name(str(row.get("player", "")))
        idx = name_to_index.get(player_key)
        if idx is not None and idx < adjusted.size(0):
            adjusted[idx, 0] = float(row.get("start_probability", adjusted[idx, 0]))

    for row in team_context.get("injuries", []):
        player_key = normalize_name(str(row.get("player", "")))
        idx = name_to_index.get(player_key)
        if idx is None or idx >= adjusted.size(0):
            continue
        fitness = float(row.get("fitness", 100.0)) / 100.0
        adjusted[idx, 1] = max(0.0, min(1.0, fitness))
        if row.get("status", "").lower() in {"out", "injured", "suspended"}:
            adjusted[idx, 6] = 1.0
            adjusted[idx, 0] = 0.0
        elif row.get("status", "").lower() in {"doubtful", "questionable"}:
            adjusted[idx, 6] = max(adjusted[idx, 6], 0.5)
            adjusted[idx, 0] = min(adjusted[idx, 0], 0.5)
    return adjusted


def merge_team_context(base: dict[str, Any] | None, override: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(base or {})
    override = override or {}
    for key, value in override.items():
        if key in {"injuries", "probable_xi", "bench", "aliases"} and isinstance(value, list):
            merged[key] = list(value)
        elif key in {"coach", "tactical_overrides"} and isinstance(value, dict):
            existing = dict(merged.get(key, {}))
            existing.update(value)
            merged[key] = existing
        else:
            merged[key] = value
    return merged
