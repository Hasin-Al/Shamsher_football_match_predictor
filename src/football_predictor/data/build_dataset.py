from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ..config import ProjectPaths
from .fifa import FifaRankingClient
from .knowledge_text import (
    KnowledgeTextEncoder,
    KnowledgeTextConfig,
    summarize_team_knowledge_text,
)
from .local_sources import (
    load_data_final_match_rows,
    load_local_match_stat_rows,
    load_updated_player_priors,
)
from .name_normalization import canonical_person_name, canonical_team_name
from .statsbomb import StatsBombClient
from .team_context import (
    SQUAD_FEATURE_NAMES,
    SQUAD_SIZE,
    NEWS_VECTOR_DIM,
    load_manual_team_context,
    build_squad_matrix,
    ensure_manual_context_template,
    extract_lineup_player_sets,
    update_player_availability,
)


BASE_PLAYER_FEATURE_NAMES = [
    "minutes",
    "touches",
    "passes_completed",
    "passes_attempted",
    "pass_xg_chain",
    "shots",
    "goals",
    "shot_xg",
    "pressures",
    "carries",
    "dribbles",
    "duels",
]

PLAYER_FEATURE_NAMES = BASE_PLAYER_FEATURE_NAMES + [
    "club_minutes_prior",
    "club_starts_prior",
    "club_goals_per90_prior",
    "club_assists_per90_prior",
    "club_xg90_prior",
    "club_xa90_prior",
    "club_cards_per90_prior",
    "club_prior_confidence",
]

SQUAD_PRIOR_FEATURE_NAMES = [
    "club_minutes_prior",
    "club_starts_prior",
    "club_goals_per90_prior",
    "club_assists_per90_prior",
    "club_xg90_prior",
    "club_xa90_prior",
    "club_cards_per90_prior",
    "club_prior_confidence",
]

DATASET_SQUAD_FEATURE_NAMES = SQUAD_FEATURE_NAMES + SQUAD_PRIOR_FEATURE_NAMES

CONTEXT_FEATURE_NAMES = [
    "fifa_rank_gap",
    "fifa_points_gap",
    "home_form_points",
    "away_form_points",
    "home_goal_diff_form",
    "away_goal_diff_form",
    "home_rest_days",
    "away_rest_days",
    "neutral_venue",
    "is_tournament",
    "home_pass_completion_rate",
    "away_pass_completion_rate",
    "home_shots_style",
    "away_shots_style",
    "home_xg_style",
    "away_xg_style",
    "home_progressive_passes",
    "away_progressive_passes",
    "home_final_third_entries",
    "away_final_third_entries",
    "home_direct_speed_proxy",
    "away_direct_speed_proxy",
    "home_pressing_intensity_proxy",
    "away_pressing_intensity_proxy",
    "home_coach_match_count",
    "away_coach_match_count",
    "home_coach_known",
    "away_coach_known",
    "home_projected_lineup_strength",
    "away_projected_lineup_strength",
    "home_projected_bench_strength",
    "away_projected_bench_strength",
    "home_availability_index",
    "away_availability_index",
    "home_attacking_potential",
    "away_attacking_potential",
    "home_creative_control",
    "away_creative_control",
    "home_defensive_discipline",
    "away_defensive_discipline",
    "home_lineup_cohesion",
    "away_lineup_cohesion",
    "home_defenders",
    "away_defenders",
    "home_midfielders",
    "away_midfielders",
    "home_forwards",
    "away_forwards",
    "home_back_three",
    "away_back_three",
    "home_double_pivot",
    "away_double_pivot",
]

REGRESSION_TARGET_NAMES = [
    "home_xg",
    "away_xg",
    "home_shots",
    "away_shots",
    "home_passes",
    "away_passes",
    "home_possession_proxy",
    "away_possession_proxy",
    "home_shots_on_target",
    "away_shots_on_target",
]

SCORE_TARGET_NAMES = [
    "home_goals",
    "away_goals",
]


@dataclass(slots=True)
class TeamState:
    recent_points: list[float]
    recent_goal_diff: list[float]
    last_match_date: datetime | None = None
    player_states: dict[int, Any] | None = None

    def __post_init__(self) -> None:
        if self.player_states is None:
            self.player_states = {}


TEAM_TACTICAL_KEYS = [
    "pass_completion_rate",
    "shots",
    "xg",
    "progressive_passes",
    "final_third_entries",
    "direct_speed_proxy",
    "pressing_intensity_proxy",
]

TEAM_EXPERT_FEATURE_KEYS = [
    "projected_lineup_strength",
    "projected_bench_strength",
    "availability_index",
    "attacking_potential",
    "creative_control",
    "defensive_discipline",
    "lineup_cohesion",
]

TEAM_CODE_PATTERN = re.compile(r"\s+[a-z]{2,3}$", re.IGNORECASE)
TEAM_CODE_PREFIX_PATTERN = re.compile(r"^[a-z]{2,3}\s+")


def safe_lower(value: str | None) -> str:
    return (value or "").strip().lower()


def parse_match_date(match_row: dict[str, Any]) -> datetime:
    for key in ("match_date", "kick_off", "match_updated", "last_updated"):
        value = match_row.get(key)
        if value:
            text = str(value).replace("Z", "")
            try:
                return datetime.fromisoformat(text)
            except ValueError:
                pass
    raise ValueError(f"Unable to parse date from match row: {match_row.get('match_id')}")


def squash(values: list[float], size: int = 5) -> tuple[float, float]:
    recent = values[-size:]
    if not recent:
        return 0.0, 0.0
    return float(np.mean(recent)), float(np.sum(recent))


def event_team_name(event: dict[str, Any]) -> str:
    team = event.get("team") or {}
    return str(team.get("name", "")).strip()


def normalize_person_name(name: str | None) -> str:
    return canonical_person_name(name)


def parse_number(value: Any) -> float:
    text = str(value or "").strip().replace(",", "")
    if not text:
        return 0.0
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return 0.0
    return float(match.group(0))


def safe_divide(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-6:
        return 0.0
    return numerator / denominator


def load_team_tactical_knowledge(paths: ProjectPaths) -> dict[str, dict[str, float]]:
    path = paths.external_dir / "statsbomb_enriched" / "team_tactical_summary.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    knowledge: dict[str, dict[str, float]] = {}
    for row in frame.to_dict("records"):
        team_key = normalize_team_name(row.get("team", ""))
        if not team_key:
            continue
        knowledge[team_key] = {metric: float(row.get(metric) or 0.0) for metric in TEAM_TACTICAL_KEYS}
    return knowledge


def load_team_coaching_knowledge(paths: ProjectPaths) -> dict[str, dict[str, Any]]:
    summary_path = paths.external_dir / "statsbomb_enriched" / "coach_team_summary.csv"
    profiles_path = paths.external_dir / "statsbomb_enriched" / "coaching_profiles.csv"
    if not summary_path.exists():
        return {}

    summary = pd.read_csv(summary_path)
    latest_manager: dict[str, str] = {}
    if profiles_path.exists():
        profiles = pd.read_csv(profiles_path)
        if {"team", "match_date", "manager_name"}.issubset(profiles.columns):
            profiles = profiles.sort_values("match_date")
            for row in profiles.to_dict("records"):
                team_key = normalize_team_name(row.get("team", ""))
                manager_name = str(row.get("manager_name") or "").strip()
                if team_key and manager_name:
                    latest_manager[team_key] = manager_name

    best_by_team: dict[str, dict[str, Any]] = {}
    for row in summary.to_dict("records"):
        team_key = normalize_team_name(row.get("team", ""))
        manager_name = str(row.get("manager_name") or "").strip()
        match_count = int(row.get("match_count") or 0)
        if not team_key:
            continue
        record = best_by_team.get(team_key)
        if record is None or match_count > int(record.get("match_count", 0)):
            best_by_team[team_key] = {"manager_name": manager_name, "match_count": match_count}

    for team_key, manager_name in latest_manager.items():
        if team_key not in best_by_team:
            best_by_team[team_key] = {"manager_name": manager_name, "match_count": 0}
        else:
            best_by_team[team_key]["latest_manager_name"] = manager_name
    return best_by_team


def team_knowledge_vector(
    team_key: str,
    tactical_knowledge: dict[str, dict[str, float]],
    coaching_knowledge: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    tactical = tactical_knowledge.get(team_key, {})
    coaching = coaching_knowledge.get(team_key, {})
    return {
        "tactical": {metric: float(tactical.get(metric, 0.0)) for metric in TEAM_TACTICAL_KEYS},
        "manager_name": str(coaching.get("latest_manager_name") or coaching.get("manager_name") or ""),
        "coach_match_count": float(coaching.get("match_count") or 0.0),
        "coach_known": 1.0 if coaching else 0.0,
    }


def build_context_vector(
    *,
    home_rank: float,
    away_rank: float,
    home_points: float,
    away_points: float,
    home_form_sum: float,
    away_form_sum: float,
    home_gd_mean: float,
    away_gd_mean: float,
    home_rest_days: float,
    away_rest_days: float,
    neutral_venue: float,
    is_tournament: float,
    home_knowledge: dict[str, Any],
    away_knowledge: dict[str, Any],
    home_expert_features: dict[str, float] | None = None,
    away_expert_features: dict[str, float] | None = None,
    home_formation: str | None = None,
    away_formation: str | None = None,
) -> torch.Tensor:
    home_tactical = home_knowledge.get("tactical", {})
    away_tactical = away_knowledge.get("tactical", {})
    home_expert_features = home_expert_features or {}
    away_expert_features = away_expert_features or {}
    home_shape = formation_features(home_formation)
    away_shape = formation_features(away_formation)
    return torch.tensor(
        [
            float(away_rank - home_rank),
            float(home_points - away_points),
            float(home_form_sum),
            float(away_form_sum),
            float(home_gd_mean),
            float(away_gd_mean),
            float(home_rest_days),
            float(away_rest_days),
            float(neutral_venue),
            float(is_tournament),
            float(home_tactical.get("pass_completion_rate", 0.0)),
            float(away_tactical.get("pass_completion_rate", 0.0)),
            float(home_tactical.get("shots", 0.0)),
            float(away_tactical.get("shots", 0.0)),
            float(home_tactical.get("xg", 0.0)),
            float(away_tactical.get("xg", 0.0)),
            float(home_tactical.get("progressive_passes", 0.0)),
            float(away_tactical.get("progressive_passes", 0.0)),
            float(home_tactical.get("final_third_entries", 0.0)),
            float(away_tactical.get("final_third_entries", 0.0)),
            float(home_tactical.get("direct_speed_proxy", 0.0)),
            float(away_tactical.get("direct_speed_proxy", 0.0)),
            float(home_tactical.get("pressing_intensity_proxy", 0.0)),
            float(away_tactical.get("pressing_intensity_proxy", 0.0)),
            float(home_knowledge.get("coach_match_count", 0.0)),
            float(away_knowledge.get("coach_match_count", 0.0)),
            float(home_knowledge.get("coach_known", 0.0)),
            float(away_knowledge.get("coach_known", 0.0)),
            float(home_expert_features.get("projected_lineup_strength", 0.0)),
            float(away_expert_features.get("projected_lineup_strength", 0.0)),
            float(home_expert_features.get("projected_bench_strength", 0.0)),
            float(away_expert_features.get("projected_bench_strength", 0.0)),
            float(home_expert_features.get("availability_index", 0.0)),
            float(away_expert_features.get("availability_index", 0.0)),
            float(home_expert_features.get("attacking_potential", 0.0)),
            float(away_expert_features.get("attacking_potential", 0.0)),
            float(home_expert_features.get("creative_control", 0.0)),
            float(away_expert_features.get("creative_control", 0.0)),
            float(home_expert_features.get("defensive_discipline", 0.0)),
            float(away_expert_features.get("defensive_discipline", 0.0)),
            float(home_expert_features.get("lineup_cohesion", 0.0)),
            float(away_expert_features.get("lineup_cohesion", 0.0)),
            home_shape["defenders"],
            away_shape["defenders"],
            home_shape["midfielders"],
            away_shape["midfielders"],
            home_shape["forwards"],
            away_shape["forwards"],
            home_shape["back_three"],
            away_shape["back_three"],
            home_shape["double_pivot"],
            away_shape["double_pivot"],
        ],
        dtype=torch.float32,
    )


def formation_features(formation: str | None) -> dict[str, float]:
    parts = [int(part) for part in str(formation or "").strip().split("-") if part.isdigit()]
    if len(parts) == 3:
        defenders, midfielders, forwards = parts
    elif len(parts) == 4:
        defenders, midfielders, forwards = parts[0], parts[1] + parts[2], parts[3]
    elif len(parts) >= 5:
        defenders, midfielders, forwards = parts[0], sum(parts[1:-1]), parts[-1]
    else:
        defenders, midfielders, forwards = 4, 3, 3
    return {
        "defenders": float(defenders) / 5.0,
        "midfielders": float(midfielders) / 6.0,
        "forwards": float(forwards) / 4.0,
        "back_three": 1.0 if defenders == 3 else 0.0,
        "double_pivot": 1.0 if len(parts) >= 4 and parts[1] == 2 else 0.0,
    }


def summarize_squad_expert_features(squad_x: np.ndarray | torch.Tensor) -> dict[str, float]:
    matrix = np.asarray(squad_x, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return {key: 0.0 for key in TEAM_EXPERT_FEATURE_KEYS}

    top_lineup = matrix[:11]
    bench = matrix[11:]
    start_prob = top_lineup[:, 0] if top_lineup.shape[1] > 0 else np.zeros((top_lineup.shape[0],), dtype=np.float32)
    fitness = top_lineup[:, 1] if top_lineup.shape[1] > 1 else np.zeros((top_lineup.shape[0],), dtype=np.float32)
    continuity = top_lineup[:, 4] if top_lineup.shape[1] > 4 else np.zeros((top_lineup.shape[0],), dtype=np.float32)
    importance = top_lineup[:, 5] if top_lineup.shape[1] > 5 else np.zeros((top_lineup.shape[0],), dtype=np.float32)
    unavailable = top_lineup[:, 6] if top_lineup.shape[1] > 6 else np.zeros((top_lineup.shape[0],), dtype=np.float32)

    goals_per90 = top_lineup[:, len(SQUAD_FEATURE_NAMES) + 2] if top_lineup.shape[1] > len(SQUAD_FEATURE_NAMES) + 2 else np.zeros((top_lineup.shape[0],), dtype=np.float32)
    assists_per90 = top_lineup[:, len(SQUAD_FEATURE_NAMES) + 3] if top_lineup.shape[1] > len(SQUAD_FEATURE_NAMES) + 3 else np.zeros((top_lineup.shape[0],), dtype=np.float32)
    xg90 = top_lineup[:, len(SQUAD_FEATURE_NAMES) + 4] if top_lineup.shape[1] > len(SQUAD_FEATURE_NAMES) + 4 else np.zeros((top_lineup.shape[0],), dtype=np.float32)
    xa90 = top_lineup[:, len(SQUAD_FEATURE_NAMES) + 5] if top_lineup.shape[1] > len(SQUAD_FEATURE_NAMES) + 5 else np.zeros((top_lineup.shape[0],), dtype=np.float32)
    cards_per90 = top_lineup[:, len(SQUAD_FEATURE_NAMES) + 6] if top_lineup.shape[1] > len(SQUAD_FEATURE_NAMES) + 6 else np.zeros((top_lineup.shape[0],), dtype=np.float32)
    prior_confidence = top_lineup[:, len(SQUAD_FEATURE_NAMES) + 7] if top_lineup.shape[1] > len(SQUAD_FEATURE_NAMES) + 7 else np.zeros((top_lineup.shape[0],), dtype=np.float32)

    lineup_strength = np.mean(start_prob * ((0.45 * importance) + (0.3 * fitness) + (0.15 * prior_confidence) + (0.1 * continuity)))
    if bench.size > 0:
        bench_strength = np.mean(
            bench[:, 0]
            * ((0.5 * bench[:, 5]) + (0.2 * bench[:, 1]) + (0.2 * bench[:, -1]) + (0.1 * bench[:, 4]))
        )
    else:
        bench_strength = 0.0
    availability_index = np.mean((1.0 - unavailable) * fitness * start_prob)
    attacking_potential = np.mean(start_prob * (0.55 * goals_per90 + 0.45 * xg90))
    creative_control = np.mean(start_prob * (0.4 * assists_per90 + 0.35 * xa90 + 0.15 * continuity + 0.1 * importance))
    defensive_discipline = np.mean(start_prob * np.clip((0.5 * fitness) + (0.35 * continuity) + (0.25 * prior_confidence) - (0.25 * cards_per90), 0.0, 2.0))
    lineup_cohesion = np.mean(start_prob * continuity)

    return {
        "projected_lineup_strength": float(lineup_strength),
        "projected_bench_strength": float(bench_strength),
        "availability_index": float(availability_index),
        "attacking_potential": float(attacking_potential),
        "creative_control": float(creative_control),
        "defensive_discipline": float(defensive_discipline),
        "lineup_cohesion": float(lineup_cohesion),
    }


def load_fbref_player_priors(paths: ProjectPaths) -> dict[str, np.ndarray]:
    candidates = [
        paths.external_dir / "imports" / "fbref" / "fbref_standard_stats.txt",
        paths.external_dir / "fbref" / "fbref_standard_stats.csv",
    ]
    source_path = next((path for path in candidates if path.exists()), None)
    if source_path is None:
        return {}

    rows: list[list[str]] = []
    if source_path.suffix.lower() == ".txt":
        lines = source_path.read_text(encoding="utf-8").splitlines()
        for line in lines[2:]:
            parts = line.split("\t")
            if len(parts) < 26 or parts[0] == "Rk":
                continue
            rows.append(parts[:26])
    else:
        frame = pd.read_csv(source_path)
        if "Player" not in frame.columns:
            return {}
        rows = frame.fillna("").astype(str).values.tolist()

    weighted_priors: dict[str, list[tuple[np.ndarray, float]]] = defaultdict(list)
    for row in rows:
        if len(row) < 26:
            continue
        player_name = normalize_person_name(row[1])
        if not player_name:
            continue
        minutes = parse_number(row[10])
        starts = parse_number(row[9])
        nineties = max(parse_number(row[11]), safe_divide(minutes, 90.0))
        goals_per90 = parse_number(row[20])
        assists_per90 = parse_number(row[21])
        cards_per90 = safe_divide(parse_number(row[18]) + parse_number(row[19]), max(nineties, 1.0))
        prior = np.array(
            [
                min(minutes / 3000.0, 1.5),
                min(starts / 38.0, 1.5),
                goals_per90,
                assists_per90,
                0.0,
                0.0,
                cards_per90,
                0.5,
            ],
            dtype=np.float32,
        )
        weight = max(minutes, 1.0)
        weighted_priors[player_name].append((prior, weight))

    collapsed: dict[str, np.ndarray] = {}
    for player_name, rows in weighted_priors.items():
        total_weight = float(sum(weight for _, weight in rows))
        if total_weight <= 0:
            continue
        total = np.sum([prior * weight for prior, weight in rows], axis=0)
        collapsed[player_name] = total / total_weight
    return collapsed


def load_understat_player_priors(paths: ProjectPaths) -> dict[str, np.ndarray]:
    weighted_priors: dict[str, list[tuple[np.ndarray, float]]] = defaultdict(list)
    source_candidates = [
        paths.external_dir / "understat" / "player_stats_merged.csv",
        *sorted((paths.external_dir / "understat").glob("*_players.csv")),
        *sorted((paths.external_dir / "imports" / "understat").glob("*_players.csv")),
        *sorted((paths.external_dir / "imports" / "understat").glob("*_players.json")),
    ]

    for source_path in source_candidates:
        if not source_path.exists():
            continue
        if source_path.suffix.lower() == ".json":
            try:
                rows = json.loads(source_path.read_text(encoding="utf-8-sig"))
            except Exception:
                continue
            if not isinstance(rows, list):
                continue
            records = [row for row in rows if isinstance(row, dict)]
        else:
            frame = pd.read_csv(source_path)
            if "Player" in frame.columns:
                records = frame.to_dict("records")
            else:
                normalized_columns = {str(column).strip().lower(): column for column in frame.columns}
                player_col = normalized_columns.get("player")
                if player_col is None:
                    continue
                records = frame.to_dict("records")

        for row in records:
            player_name = normalize_person_name(row.get("Player") or row.get("player"))
            if not player_name:
                continue
            minutes = parse_number(row.get("Min") or row.get("min"))
            apps = parse_number(row.get("Apps") or row.get("apps"))
            goals = parse_number(row.get("G") or row.get("goals"))
            assists = parse_number(row.get("A") or row.get("a"))
            xg90 = parse_number(row.get("xG90") or row.get("xg90"))
            xa90 = parse_number(row.get("xA90") or row.get("xa90"))
            xg_total = parse_number(row.get("xG") or row.get("xg"))
            xa_total = parse_number(row.get("xA") or row.get("xa"))
            nineties = max(minutes / 90.0, 1.0)
            if xg90 <= 0.0 and xg_total > 0.0:
                xg90 = xg_total / nineties
            if xa90 <= 0.0 and xa_total > 0.0:
                xa90 = xa_total / nineties
            cards_per90 = 0.0
            prior = np.array(
                [
                    min(minutes / 3000.0, 1.5),
                    min(apps / 38.0, 1.5),
                    goals / nineties,
                    assists / nineties,
                    xg90,
                    xa90,
                    cards_per90,
                    1.0,
                ],
                dtype=np.float32,
            )
            weight = max(minutes, 1.0)
            weighted_priors[player_name].append((prior, weight))

    collapsed: dict[str, np.ndarray] = {}
    for player_name, rows in weighted_priors.items():
        total_weight = float(sum(weight for _, weight in rows))
        if total_weight <= 0:
            continue
        total = np.sum([prior * weight for prior, weight in rows], axis=0)
        collapsed[player_name] = total / total_weight
    return collapsed


def load_tournament_player_priors() -> dict[str, np.ndarray]:
    processed_dir = Path("new_data/football_prediction_dataset/data/processed")
    files = [
        processed_dir / "world_cup_2022_player_stats.csv",
        processed_dir / "euro_2024_player_stats.csv",
    ]
    weighted_priors: dict[str, list[tuple[np.ndarray, float]]] = defaultdict(list)

    for path in files:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        required = {"player_name", "minutes_played", "goals", "assists", "xg", "xa"}
        if not required.issubset(frame.columns):
            continue
        for row in frame.to_dict("records"):
            player_name = normalize_person_name(row.get("player_name"))
            if not player_name:
                continue
            minutes = parse_number(row.get("minutes_played"))
            if minutes <= 0:
                continue
            nineties = max(minutes / 90.0, 1.0)
            starts_proxy = min(1.0, nineties / 7.0)
            prior = np.array(
                [
                    min(minutes / 3000.0, 1.5),
                    starts_proxy,
                    parse_number(row.get("goals")) / nineties,
                    parse_number(row.get("assists")) / nineties,
                    parse_number(row.get("xg")) / nineties,
                    parse_number(row.get("xa")) / nineties,
                    0.0,
                    0.8,
                ],
                dtype=np.float32,
            )
            weighted_priors[player_name].append((prior, minutes))

    collapsed: dict[str, np.ndarray] = {}
    for player_name, rows in weighted_priors.items():
        total_weight = float(sum(weight for _, weight in rows))
        if total_weight <= 0:
            continue
        total = np.sum([prior * weight for prior, weight in rows], axis=0)
        collapsed[player_name] = total / total_weight
    return collapsed


def combine_prior_vectors(primary: np.ndarray | None, secondary: np.ndarray | None, tertiary: np.ndarray | None = None) -> np.ndarray | None:
    vectors = [vector for vector in [primary, secondary, tertiary] if vector is not None]
    if not vectors:
        return None
    if len(vectors) == 1:
        return vectors[0]
    merged = np.mean(np.stack(vectors, axis=0), axis=0).astype(np.float32)
    merged[-1] = max(vector[-1] for vector in vectors)
    merged[0] = max(vector[0] for vector in vectors)
    merged[1] = max(vector[1] for vector in vectors)
    return merged


def build_player_prior_index(paths: ProjectPaths) -> dict[str, np.ndarray]:
    fbref_priors = load_fbref_player_priors(paths)
    understat_priors = load_understat_player_priors(paths)
    tournament_priors = load_tournament_player_priors()
    updated_priors, _ = load_updated_player_priors(Path.cwd())
    merged: dict[str, np.ndarray] = {}
    for player_name in sorted(set(fbref_priors) | set(understat_priors) | set(tournament_priors) | set(updated_priors)):
        fbref = fbref_priors.get(player_name)
        understat = understat_priors.get(player_name)
        tournament = tournament_priors.get(player_name)
        updated = updated_priors.get(player_name)
        combined = combine_prior_vectors(understat, fbref, tournament)
        combined = combine_prior_vectors(combined, updated)
        if combined is not None:
            merged[player_name] = combined
    return merged


def player_prior_vector(player_name: str, player_priors: dict[str, np.ndarray]) -> np.ndarray:
    return player_priors.get(normalize_person_name(player_name), np.zeros(len(SQUAD_PRIOR_FEATURE_NAMES), dtype=np.float32))


def augment_player_features_with_priors(
    player_features: dict[int, np.ndarray],
    player_names: dict[int, str],
    player_priors: dict[str, np.ndarray],
) -> dict[int, np.ndarray]:
    augmented: dict[int, np.ndarray] = {}
    for player_id, base_vector in player_features.items():
        prior = player_prior_vector(player_names.get(player_id, ""), player_priors)
        augmented[player_id] = np.concatenate([base_vector, prior]).astype(np.float32)
    return augmented


def augment_squad_matrix_with_priors(
    squad_x: torch.Tensor,
    squad_player_names: list[str],
    player_priors: dict[str, np.ndarray],
) -> torch.Tensor:
    prior_rows = [player_prior_vector(player_name, player_priors) for player_name in squad_player_names]
    priors_tensor = torch.tensor(np.vstack(prior_rows), dtype=torch.float32)
    return torch.cat([squad_x, priors_tensor], dim=1)


def build_player_features(events: list[dict[str, Any]], team_name: str) -> tuple[dict[int, np.ndarray], dict[int, str], dict[tuple[int, int], float]]:
    player_features: dict[int, np.ndarray] = {}
    player_names: dict[int, str] = {}
    pass_edges: dict[tuple[int, int], float] = defaultdict(float)

    def ensure_player(player_id: int, player_name: str) -> np.ndarray:
        if player_id not in player_features:
            player_features[player_id] = np.zeros(len(BASE_PLAYER_FEATURE_NAMES), dtype=np.float32)
            player_names[player_id] = player_name
        return player_features[player_id]

    for event in events:
        if safe_lower(event_team_name(event)) != safe_lower(team_name):
            continue
        player = event.get("player") or {}
        player_id = player.get("id")
        player_name = str(player.get("name", "Unknown"))
        if player_id is None:
            continue
        feature_vector = ensure_player(int(player_id), player_name)
        feature_vector[1] += 1.0
        event_type = ((event.get("type") or {}).get("name") or "").lower()

        if event_type == "pass":
            feature_vector[2] += 1.0
            feature_vector[3] += 1.0
            pass_details = event.get("pass") or {}
            recipient = pass_details.get("recipient") or {}
            recipient_id = recipient.get("id")
            if recipient_id is not None:
                ensure_player(int(recipient_id), str(recipient.get("name", "Unknown")))
                pass_edges[(int(player_id), int(recipient_id))] += 1.0
            if pass_details.get("outcome") is not None:
                feature_vector[2] -= 1.0
        elif event_type == "shot":
            feature_vector[5] += 1.0
            shot_details = event.get("shot") or {}
            feature_vector[7] += float(shot_details.get("statsbomb_xg") or 0.0)
            if ((shot_details.get("outcome") or {}).get("name") or "").lower() == "goal":
                feature_vector[6] += 1.0
        elif event_type == "pressure":
            feature_vector[8] += 1.0
        elif event_type == "carry":
            feature_vector[9] += 1.0
        elif event_type == "dribble":
            feature_vector[10] += 1.0
        elif event_type in {"duel", "50/50"}:
            feature_vector[11] += 1.0

    for player_id, vector in player_features.items():
        touches = max(vector[1], 1.0)
        vector[4] = vector[7] / touches
        vector[0] = min(90.0, max(1.0, touches / 1.2))
        player_features[player_id] = vector

    return player_features, player_names, pass_edges


def graph_from_player_features(
    player_features: dict[int, np.ndarray],
    pass_edges: dict[tuple[int, int], float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    player_ids = sorted(player_features)
    if not player_ids:
        x = torch.zeros((1, len(PLAYER_FEATURE_NAMES)), dtype=torch.float32)
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        edge_weight = torch.ones((1,), dtype=torch.float32)
        return x, edge_index, edge_weight, [-1]

    id_to_index = {player_id: idx for idx, player_id in enumerate(player_ids)}
    x = torch.tensor(np.vstack([player_features[player_id] for player_id in player_ids]), dtype=torch.float32)

    if pass_edges:
        edges = []
        weights = []
        for (src, dst), weight in pass_edges.items():
            if src in id_to_index and dst in id_to_index:
                edges.append([id_to_index[src], id_to_index[dst]])
                weights.append(weight)
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_weight = torch.tensor(weights, dtype=torch.float32)
        else:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            edge_weight = torch.ones((1,), dtype=torch.float32)
    else:
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        edge_weight = torch.ones((1,), dtype=torch.float32)

    return x, edge_index, edge_weight, player_ids


def extract_goal_scorers(events: list[dict[str, Any]], team_name: str) -> set[int]:
    scorer_ids: set[int] = set()
    team_key = normalize_team_name(team_name)
    for event in events:
        if normalize_team_name(event_team_name(event)) != team_key:
            continue
        if ((event.get("type") or {}).get("name") or "").lower() != "shot":
            continue
        shot_details = event.get("shot") or {}
        if ((shot_details.get("outcome") or {}).get("name") or "").lower() != "goal":
            continue
        player = event.get("player") or {}
        player_id = player.get("id")
        if player_id is not None:
            scorer_ids.add(int(player_id))
    return scorer_ids


def build_player_binary_targets(player_ids: list[int], positive_ids: set[int]) -> torch.Tensor:
    return torch.tensor([1.0 if player_id in positive_ids else 0.0 for player_id in player_ids], dtype=torch.float32)


def build_player_valid_mask(player_ids: list[int]) -> torch.Tensor:
    return torch.tensor([1.0 if player_id != -1 else 0.0 for player_id in player_ids], dtype=torch.float32)


def build_player_name_targets(player_names: list[str], positive_names: list[str]) -> torch.Tensor:
    positives = {normalize_person_name(name) for name in positive_names if normalize_person_name(name)}
    values = []
    for player_name in player_names:
        normalized = normalize_person_name(player_name)
        last_name = normalized.split()[-1] if normalized.split() else ""
        matched = normalized in positives or any(last_name and last_name in positive.split()[-1:] for positive in positives)
        values.append(1.0 if matched else 0.0)
    return torch.tensor(values, dtype=torch.float32)


def result_class(home_score: int, away_score: int) -> int:
    if home_score > away_score:
        return 0
    if home_score == away_score:
        return 1
    return 2


def normalize_team_name(name: str) -> str:
    return canonical_team_name(name)


def clean_schedule_team_name(value: str) -> str:
    text = " ".join(str(value or "").replace("‏", "").split()).strip()
    text = text.replace("Rep. of Ireland", "Republic of Ireland")
    text = text.replace("N. Macedonia", "North Macedonia")
    text = text.replace("Congo DR", "DR Congo")
    text = TEAM_CODE_PREFIX_PATTERN.sub("", text).strip()
    text = TEAM_CODE_PATTERN.sub("", text).strip()
    return text


def parse_more_matches_file(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    competition_name = ""
    rows: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip("\n")
        compact = " ".join(line.split()).strip()
        if not compact:
            continue
        if compact.startswith("Scores & Fixtures "):
            competition_name = compact
            continue
        if compact.startswith("Day Date Time Home Score Away") or compact.startswith("Day Date Time"):
            continue
        parts = [part.strip() for part in line.split("\t")]
        if len(parts) < 5:
            continue

        date_idx = next(
            (idx for idx, value in enumerate(parts) if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value or "")),
            None,
        )
        score_idx = next(
            (
                idx
                for idx, value in enumerate(parts)
                if re.fullmatch(r"\d+\s*[–—−-]\s*\d+", (value or "").strip())
            ),
            None,
        )
        if date_idx is None or score_idx is None:
            continue
        if score_idx - 1 < 0 or score_idx + 1 >= len(parts):
            continue
        score_text = parts[score_idx].replace("—", "-").replace("–", "-").replace("−", "-").strip()
        score_match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", score_text)
        if not score_match:
            continue
        home_team = clean_schedule_team_name(parts[score_idx - 1])
        away_team = clean_schedule_team_name(parts[score_idx + 1])
        if not home_team or not away_team:
            continue
        rows.append(
            {
                "competition_name": competition_name,
                "date": parts[date_idx],
                "home_team": home_team,
                "away_team": away_team,
                "home_score": int(score_match.group(1)),
                "away_score": int(score_match.group(2)),
                "source": "more_matches_txt",
            }
        )
    return rows


def parse_structured_match_results(path: Path, source_name: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    required = {"match_date", "home_team", "away_team", "home_score", "away_score"}
    if not required.issubset(frame.columns):
        return []
    rows: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        match_date = str(row.get("match_date") or "").strip()
        home_team = clean_schedule_team_name(str(row.get("home_team") or ""))
        away_team = clean_schedule_team_name(str(row.get("away_team") or ""))
        if not match_date or not home_team or not away_team:
            continue
        rows.append(
            {
                "competition_name": str(row.get("competition") or row.get("competition_stage") or source_name),
                "date": match_date,
                "home_team": home_team,
                "away_team": away_team,
                "home_score": int(parse_number(row.get("home_score"))),
                "away_score": int(parse_number(row.get("away_score"))),
                "source": source_name,
            }
        )
    return rows


def default_graph_sample() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], list[str]]:
    return (
        torch.zeros((1, len(PLAYER_FEATURE_NAMES)), dtype=torch.float32),
        torch.tensor([[0], [0]], dtype=torch.long),
        torch.ones((1,), dtype=torch.float32),
        [-1],
        ["Unknown"],
    )


def ensure_prior_augmented_squad(
    squad_x: torch.Tensor,
    squad_names: list[str],
    player_priors: dict[str, np.ndarray],
) -> torch.Tensor:
    if squad_x.size(1) == len(DATASET_SQUAD_FEATURE_NAMES):
        return squad_x
    return augment_squad_matrix_with_priors(squad_x, squad_names, player_priors)


def squad_from_prior_roster(
    roster_rows: list[tuple[str, np.ndarray]],
    player_priors: dict[str, np.ndarray],
) -> tuple[torch.Tensor, list[int], list[str]]:
    rows: list[list[float]] = []
    names: list[str] = []
    ids: list[int] = []
    for idx, (player_name, prior) in enumerate(roster_rows[:SQUAD_SIZE]):
        prior = np.asarray(prior, dtype=np.float32)
        minutes_prior = float(prior[0]) if prior.size > 0 else 0.0
        starts_prior = float(prior[1]) if prior.size > 1 else 0.0
        goal_signal = float(prior[2] + prior[4]) if prior.size > 4 else 0.0
        creative_signal = float(prior[3] + prior[5]) if prior.size > 5 else 0.0
        confidence = float(prior[-1]) if prior.size else 0.0
        start_prob = max(0.05, min(0.98, 0.55 * min(starts_prior, 1.0) + 0.35 * min(minutes_prior, 1.0) + 0.08 * confidence))
        importance = max(0.05, min(1.0, 0.35 * min(minutes_prior, 1.0) + 0.25 * min(starts_prior, 1.0) + 0.25 * goal_signal + 0.15 * creative_signal))
        base = [
            start_prob,
            1.0,
            min(starts_prior, 1.0),
            min(minutes_prior, 1.0),
            0.35,
            importance,
            0.0,
        ]
        rows.append(base + player_prior_vector(player_name, player_priors).tolist())
        names.append(player_name)
        ids.append(-10_000 - idx)

    while len(rows) < SQUAD_SIZE:
        rows.append([0.0] * len(DATASET_SQUAD_FEATURE_NAMES))
        names.append("Unknown")
        ids.append(-1)
    return torch.tensor(rows, dtype=torch.float32), ids, names


def squad_from_named_lineup(
    starters: list[str],
    bench: list[str],
    player_priors: dict[str, np.ndarray],
) -> tuple[torch.Tensor, list[int], list[str]]:
    rows: list[list[float]] = []
    ids: list[int] = []
    names: list[str] = []
    ordered_players = [(name, 1.0) for name in starters[:11]]
    ordered_players.extend((name, 0.18) for name in bench[: max(0, SQUAD_SIZE - len(ordered_players))])

    for idx, (player_name, start_probability) in enumerate(ordered_players[:SQUAD_SIZE]):
        prior = player_prior_vector(player_name, player_priors)
        minutes_prior = float(prior[0]) if prior.size > 0 else 0.0
        starts_prior = float(prior[1]) if prior.size > 1 else 0.0
        attacking_signal = float(prior[2] + prior[4]) if prior.size > 4 else 0.0
        creative_signal = float(prior[3] + prior[5]) if prior.size > 5 else 0.0
        confidence = float(prior[-1]) if prior.size else 0.0
        importance = max(
            0.08,
            min(
                1.0,
                (0.34 * min(minutes_prior, 1.0))
                + (0.24 * min(starts_prior, 1.0))
                + (0.22 * attacking_signal)
                + (0.12 * creative_signal)
                + (0.08 * confidence),
            ),
        )
        base = [
            float(start_probability),
            1.0,
            min(1.0, starts_prior if start_probability >= 0.5 else 0.25 * starts_prior),
            min(1.0, minutes_prior),
            0.45 if idx < 11 else 0.15,
            importance,
            0.0,
        ]
        rows.append(base + prior.tolist())
        ids.append(-20_000 - idx)
        names.append(player_name)

    while len(rows) < SQUAD_SIZE:
        rows.append([0.0] * len(DATASET_SQUAD_FEATURE_NAMES))
        ids.append(-1)
        names.append("Unknown")
    return torch.tensor(rows, dtype=torch.float32), ids, names


def squad_graph_from_squad_matrix(
    squad_x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    rows = squad_x[:SQUAD_SIZE].detach().cpu().numpy().astype(np.float32)
    node_rows: list[np.ndarray] = []
    node_ids: list[int] = []
    for idx, row in enumerate(rows):
        if row.size == 0 or float(row[0]) <= 0.0:
            continue
        prior_offset = len(SQUAD_FEATURE_NAMES)
        prior = row[prior_offset:prior_offset + len(SQUAD_PRIOR_FEATURE_NAMES)]
        vector = np.zeros(len(PLAYER_FEATURE_NAMES), dtype=np.float32)
        start_prob = float(row[0])
        recent_minutes = float(row[3]) if row.size > 3 else 0.0
        importance = float(row[5]) if row.size > 5 else 0.0
        vector[0] = 90.0 * max(start_prob, recent_minutes)
        vector[1] = 35.0 + 45.0 * importance
        vector[2] = 18.0 + 22.0 * float(prior[5] if prior.size > 5 else 0.0)
        vector[3] = vector[2] / max(0.65, min(0.92, 0.78 + 0.08 * importance))
        vector[4] = float(prior[4] if prior.size > 4 else 0.0) * max(start_prob, 0.15)
        vector[5] = max(float(prior[4] if prior.size > 4 else 0.0) * 2.8, float(prior[2] if prior.size > 2 else 0.0) * 2.4)
        vector[6] = float(prior[2] if prior.size > 2 else 0.0)
        vector[7] = float(prior[4] if prior.size > 4 else 0.0)
        vector[8] = 8.0 * max(0.1, importance)
        vector[9] = 10.0 * max(0.1, recent_minutes)
        vector[10] = 2.0 * float(prior[5] if prior.size > 5 else 0.0)
        vector[11] = 4.0 * max(0.1, importance)
        vector[len(BASE_PLAYER_FEATURE_NAMES):] = prior[:len(SQUAD_PRIOR_FEATURE_NAMES)]
        node_rows.append(vector)
        node_ids.append(-30_000 - idx)

    if not node_rows:
        return default_graph_sample()[:4]

    node_count = len(node_rows)
    edges: list[list[int]] = []
    weights: list[float] = []
    for idx in range(node_count):
        edges.append([idx, idx])
        weights.append(1.0)
        if idx + 1 < node_count:
            adjacency = 0.9 if idx < 11 and idx + 1 < 11 else 0.35
            edges.extend([[idx, idx + 1], [idx + 1, idx]])
            weights.extend([adjacency, adjacency])
    starter_indices = list(range(min(11, node_count)))
    for src in starter_indices:
        for dst in starter_indices:
            if src == dst or abs(src - dst) <= 1:
                continue
            edges.append([src, dst])
            weights.append(0.18)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    return torch.tensor(np.vstack(node_rows), dtype=torch.float32), edge_index, edge_weight, node_ids


def get_team_snapshot(
    *,
    team_name: str,
    team_key: str,
    team_profiles: dict[str, dict[str, Any]],
    team_state: TeamState,
    match_index: int,
    player_priors: dict[str, np.ndarray],
    team_prior_rosters: dict[str, list[tuple[str, np.ndarray]]] | None = None,
    compact_graph: bool = False,
) -> dict[str, Any]:
    profile = team_profiles.get(team_key)
    if profile:
        squad_x = torch.tensor(profile["squad_x"], dtype=torch.float32)
        squad_names = list(profile.get("squad_player_names", []))
        squad_ids = list(profile.get("squad_player_ids", []))
        squad_x = ensure_prior_augmented_squad(squad_x, squad_names, player_priors)
        if compact_graph:
            graph_x, graph_edge_index, graph_edge_weight, graph_player_ids = default_graph_sample()[:4]
            graph_player_names = ["Unknown"]
        else:
            graph_x, graph_edge_index, graph_edge_weight, graph_player_ids = squad_graph_from_squad_matrix(squad_x)
            graph_player_names = squad_names[: len(graph_player_ids)]
        return {
            "x": graph_x,
            "edge_index": graph_edge_index,
            "edge_weight": graph_edge_weight,
            "player_ids": graph_player_ids,
            "player_names": graph_player_names,
            "squad_x": squad_x,
            "squad_ids": squad_ids,
            "squad_names": squad_names,
        }

    roster = (team_prior_rosters or {}).get(team_key, [])
    if roster:
        squad_x, squad_ids, squad_names = squad_from_prior_roster(roster, player_priors)
        if compact_graph:
            x, edge_index, edge_weight, player_ids = default_graph_sample()[:4]
            player_names = ["Unknown"]
        else:
            x, edge_index, edge_weight, player_ids = squad_graph_from_squad_matrix(squad_x)
            player_names = squad_names[: len(player_ids)]
        return {
            "x": x,
            "edge_index": edge_index,
            "edge_weight": edge_weight,
            "player_ids": player_ids,
            "player_names": player_names,
            "squad_x": squad_x,
            "squad_ids": squad_ids,
            "squad_names": squad_names,
        }

    squad_x, squad_ids, squad_names = build_squad_matrix(team_state.player_states, match_index)
    squad_x = augment_squad_matrix_with_priors(squad_x, squad_names, player_priors)
    if compact_graph:
        x, edge_index, edge_weight, player_ids = default_graph_sample()[:4]
        player_names = ["Unknown"]
    else:
        x, edge_index, edge_weight, player_ids = squad_graph_from_squad_matrix(squad_x)
        player_names = squad_names[: len(player_ids)]
    return {
        "x": x,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "player_ids": player_ids,
        "player_names": player_names,
        "squad_x": squad_x,
        "squad_ids": squad_ids,
        "squad_names": squad_names,
    }


def get_lineup_snapshot(
    *,
    starters: list[str],
    bench: list[str],
    player_priors: dict[str, np.ndarray],
) -> dict[str, Any]:
    squad_x, squad_ids, squad_names = squad_from_named_lineup(starters, bench, player_priors)
    x, edge_index, edge_weight, player_ids = squad_graph_from_squad_matrix(squad_x)
    return {
        "x": x,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "player_ids": player_ids,
        "player_names": squad_names[: len(player_ids)],
        "squad_x": squad_x,
        "squad_ids": squad_ids,
        "squad_names": squad_names,
    }


def prepare_dataset(
    data_dir: str,
    competition_ids: list[int] | None = None,
    season_limit: int | None = None,
    use_fifa_rankings: bool = False,
) -> None:
    paths = ProjectPaths(Path(data_dir))
    paths.ensure()
    ensure_manual_context_template(paths)
    statsbomb = StatsBombClient()
    fifa = FifaRankingClient()
    player_priors = build_player_prior_index(paths)
    _, team_prior_rosters = load_updated_player_priors(Path.cwd())
    tactical_knowledge = load_team_tactical_knowledge(paths)
    coaching_knowledge = load_team_coaching_knowledge(paths)
    manual_context = load_manual_team_context(paths)
    knowledge_config = KnowledgeTextConfig.from_file(Path("config/knowledge_config.json"))
    knowledge_encoder = KnowledgeTextEncoder(knowledge_config)

    competitions = statsbomb.competitions()
    competitions_path = paths.raw_dir / "competitions.json"
    competitions_path.write_text(json.dumps(competitions), encoding="utf-8")

    competitions_df = pd.DataFrame(competitions)
    if competition_ids:
        competitions_df = competitions_df[competitions_df["competition_id"].isin(competition_ids)]
    competitions_df = competitions_df.sort_values(["competition_id", "season_id"])

    if use_fifa_rankings:
        try:
            ranking_df = fifa.fetch_rankings()
            if ranking_df.empty:
                raise ValueError("FIFA ranking table was empty.")
            ranking_df["team_key"] = ranking_df["team"].map(normalize_team_name)
            fifa_map = ranking_df.set_index("team_key")[["rank", "points"]].to_dict("index")
            ranking_df.to_csv(paths.interim_dir / "fifa_rankings.csv", index=False)
        except Exception as exc:
            print(f"Warning: FIFA rankings unavailable, using neutral defaults. reason={exc}")
            ranking_df = pd.DataFrame(columns=["rank", "team", "points", "team_key"])
            fifa_map = {}
    else:
        ranking_df = pd.DataFrame(columns=["rank", "team", "points", "team_key"])
        fifa_map = {}

    team_states: dict[str, TeamState] = defaultdict(lambda: TeamState(recent_points=[], recent_goal_diff=[]))
    team_profiles: dict[str, dict[str, Any]] = {}
    samples: list[dict[str, Any]] = []

    grouped = competitions_df.groupby("competition_id")
    for competition_id, comp_frame in grouped:
        rows = comp_frame.to_dict("records")
        if season_limit is not None:
            rows = rows[-season_limit:]
        for row in rows:
            season_id = int(row["season_id"])
            matches = statsbomb.matches(int(competition_id), season_id)
            for match in tqdm(matches, desc=f"competition={competition_id} season={season_id}"):
                match_id = int(match["match_id"])
                match_date = parse_match_date(match)
                events = statsbomb.events(match_id)
                lineups = statsbomb.lineups(match_id)
                home_team = str((match.get("home_team") or {}).get("home_team_name") or (match.get("home_team") or {}).get("name") or "")
                away_team = str((match.get("away_team") or {}).get("away_team_name") or (match.get("away_team") or {}).get("name") or "")
                home_score = int(match.get("home_score") or 0)
                away_score = int(match.get("away_score") or 0)

                home_features, home_player_names, home_edges = build_player_features(events, home_team)
                away_features, away_player_names, away_edges = build_player_features(events, away_team)
                home_features = augment_player_features_with_priors(home_features, home_player_names, player_priors)
                away_features = augment_player_features_with_priors(away_features, away_player_names, player_priors)
                home_x, home_edge_index, home_edge_weight, home_player_ids = graph_from_player_features(home_features, home_edges)
                away_x, away_edge_index, away_edge_weight, away_player_ids = graph_from_player_features(away_features, away_edges)

                home_key = normalize_team_name(home_team)
                away_key = normalize_team_name(away_team)
                home_knowledge = team_knowledge_vector(home_key, tactical_knowledge, coaching_knowledge)
                away_knowledge = team_knowledge_vector(away_key, tactical_knowledge, coaching_knowledge)
                home_rank = fifa_map.get(home_key, {}).get("rank", 999.0)
                away_rank = fifa_map.get(away_key, {}).get("rank", 999.0)
                home_points = fifa_map.get(home_key, {}).get("points", 0.0)
                away_points = fifa_map.get(away_key, {}).get("points", 0.0)

                home_state = team_states[home_key]
                away_state = team_states[away_key]
                _, home_form_sum = squash(home_state.recent_points)
                _, away_form_sum = squash(away_state.recent_points)
                home_gd_mean, _ = squash(home_state.recent_goal_diff)
                away_gd_mean, _ = squash(away_state.recent_goal_diff)
                home_rest_days = (match_date - home_state.last_match_date).days if home_state.last_match_date else 7
                away_rest_days = (match_date - away_state.last_match_date).days if away_state.last_match_date else 7
                match_index = len(samples)
                home_squad_x, home_squad_ids, home_squad_names = build_squad_matrix(home_state.player_states, match_index)
                away_squad_x, away_squad_ids, away_squad_names = build_squad_matrix(away_state.player_states, match_index)
                home_squad_x = augment_squad_matrix_with_priors(home_squad_x, home_squad_names, player_priors)
                away_squad_x = augment_squad_matrix_with_priors(away_squad_x, away_squad_names, player_priors)
                home_x, home_edge_index, home_edge_weight, home_player_ids = squad_graph_from_squad_matrix(home_squad_x)
                away_x, away_edge_index, away_edge_weight, away_player_ids = squad_graph_from_squad_matrix(away_squad_x)
                home_expert_features = summarize_squad_expert_features(home_squad_x)
                away_expert_features = summarize_squad_expert_features(away_squad_x)
                home_manual_context = manual_context.get(home_key, {})
                away_manual_context = manual_context.get(away_key, {})
                home_knowledge_text = summarize_team_knowledge_text(
                    team_name=home_team,
                    team_knowledge=home_knowledge,
                    squad_player_names=home_squad_names,
                    squad_x=home_squad_x.numpy(),
                    form_points=float(home_form_sum),
                    goal_diff_form=float(home_gd_mean),
                    manual_context=home_manual_context,
                )
                away_knowledge_text = summarize_team_knowledge_text(
                    team_name=away_team,
                    team_knowledge=away_knowledge,
                    squad_player_names=away_squad_names,
                    squad_x=away_squad_x.numpy(),
                    form_points=float(away_form_sum),
                    goal_diff_form=float(away_gd_mean),
                    manual_context=away_manual_context,
                )
                encoded_knowledge = knowledge_encoder.encode_texts([home_knowledge_text, away_knowledge_text])
                home_knowledge_vector = torch.tensor(encoded_knowledge[0], dtype=torch.float32)
                away_knowledge_vector = torch.tensor(encoded_knowledge[1], dtype=torch.float32)
                home_goal_scorers = extract_goal_scorers(events, home_team)
                away_goal_scorers = extract_goal_scorers(events, away_team)
                home_scorer_targets = build_player_binary_targets(home_squad_ids, home_goal_scorers)
                away_scorer_targets = build_player_binary_targets(away_squad_ids, away_goal_scorers)
                home_scorer_mask = build_player_valid_mask(home_squad_ids)
                away_scorer_mask = build_player_valid_mask(away_squad_ids)

                home_xg = float(sum((event.get("shot") or {}).get("statsbomb_xg") or 0.0 for event in events if normalize_team_name(event_team_name(event)) == home_key))
                away_xg = float(sum((event.get("shot") or {}).get("statsbomb_xg") or 0.0 for event in events if normalize_team_name(event_team_name(event)) == away_key))
                home_shots = float(sum(1 for event in events if normalize_team_name(event_team_name(event)) == home_key and ((event.get("type") or {}).get("name") or "").lower() == "shot"))
                away_shots = float(sum(1 for event in events if normalize_team_name(event_team_name(event)) == away_key and ((event.get("type") or {}).get("name") or "").lower() == "shot"))
                home_shots_on_target = float(sum(
                    1
                    for event in events
                    if normalize_team_name(event_team_name(event)) == home_key
                    and ((event.get("type") or {}).get("name") or "").lower() == "shot"
                    and ((event.get("shot") or {}).get("outcome") or {}).get("name") in {"Goal", "Saved", "Saved to Post"}
                ))
                away_shots_on_target = float(sum(
                    1
                    for event in events
                    if normalize_team_name(event_team_name(event)) == away_key
                    and ((event.get("type") or {}).get("name") or "").lower() == "shot"
                    and ((event.get("shot") or {}).get("outcome") or {}).get("name") in {"Goal", "Saved", "Saved to Post"}
                ))
                home_passes = float(sum(1 for event in events if normalize_team_name(event_team_name(event)) == home_key and ((event.get("type") or {}).get("name") or "").lower() == "pass"))
                away_passes = float(sum(1 for event in events if normalize_team_name(event_team_name(event)) == away_key and ((event.get("type") or {}).get("name") or "").lower() == "pass"))
                total_passes = max(home_passes + away_passes, 1.0)
                home_possession_proxy = home_passes / total_passes
                away_possession_proxy = away_passes / total_passes
                competition_name = safe_lower(str(row.get("competition_name") or row.get("competition_name_en") or ""))
                is_tournament = float(any(token in competition_name for token in ("world cup", "euro", "copa", "cup", "nations league")))

                context = build_context_vector(
                    home_rank=home_rank,
                    away_rank=away_rank,
                    home_points=home_points,
                    away_points=away_points,
                    home_form_sum=home_form_sum,
                    away_form_sum=away_form_sum,
                    home_gd_mean=home_gd_mean,
                    away_gd_mean=away_gd_mean,
                    home_rest_days=float(home_rest_days),
                    away_rest_days=float(away_rest_days),
                    neutral_venue=0.0,
                    is_tournament=is_tournament,
                    home_knowledge=home_knowledge,
                    away_knowledge=away_knowledge,
                    home_expert_features=home_expert_features,
                    away_expert_features=away_expert_features,
                )
                regression_targets = torch.tensor(
                    [
                        home_xg,
                        away_xg,
                        home_shots,
                        away_shots,
                        home_passes,
                        away_passes,
                        home_possession_proxy,
                        away_possession_proxy,
                        home_shots_on_target,
                        away_shots_on_target,
                    ],
                    dtype=torch.float32,
                )

                sample = {
                    "match_id": match_id,
                    "date": match_date.isoformat(),
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_x": home_x,
                    "home_edge_index": home_edge_index,
                    "home_edge_weight": home_edge_weight,
                    "away_x": away_x,
                    "away_edge_index": away_edge_index,
                    "away_edge_weight": away_edge_weight,
                    "home_squad_x": home_squad_x,
                    "away_squad_x": away_squad_x,
                    "home_news": torch.zeros(NEWS_VECTOR_DIM, dtype=torch.float32),
                    "away_news": torch.zeros(NEWS_VECTOR_DIM, dtype=torch.float32),
                    "home_knowledge": home_knowledge_vector,
                    "away_knowledge": away_knowledge_vector,
                    "context": context,
                    "target_class": torch.tensor(result_class(home_score, away_score), dtype=torch.long),
                    "targets": regression_targets,
                    "regression_mask": torch.ones(len(REGRESSION_TARGET_NAMES), dtype=torch.float32),
                    "score_targets": torch.tensor([float(home_score), float(away_score)], dtype=torch.float32),
                    "score_mask": torch.ones(len(SCORE_TARGET_NAMES), dtype=torch.float32),
                    "home_scorer_targets": home_scorer_targets,
                    "away_scorer_targets": away_scorer_targets,
                    "home_scorer_mask": home_scorer_mask,
                    "away_scorer_mask": away_scorer_mask,
                    "sample_source": "statsbomb",
                }
                samples.append(sample)

                team_profiles[home_key] = {
                    "team_name": home_team,
                    "player_ids": home_player_ids,
                    "player_names": home_squad_names[: len(home_player_ids)],
                    "x": home_x.tolist(),
                    "edge_index": home_edge_index.tolist(),
                    "edge_weight": home_edge_weight.tolist(),
                    "squad_x": home_squad_x.tolist(),
                    "squad_player_ids": home_squad_ids,
                    "squad_player_names": home_squad_names,
                    "team_knowledge": home_knowledge,
                    "knowledge_text": home_knowledge_text,
                    "knowledge_vector": home_knowledge_vector.tolist(),
                    "form_points": float(home_form_sum),
                    "goal_diff_form": float(home_gd_mean),
                    "last_context": context.tolist(),
                    "expert_features": home_expert_features,
                }
                team_profiles[away_key] = {
                    "team_name": away_team,
                    "player_ids": away_player_ids,
                    "player_names": away_squad_names[: len(away_player_ids)],
                    "x": away_x.tolist(),
                    "edge_index": away_edge_index.tolist(),
                    "edge_weight": away_edge_weight.tolist(),
                    "squad_x": away_squad_x.tolist(),
                    "squad_player_ids": away_squad_ids,
                    "squad_player_names": away_squad_names,
                    "team_knowledge": away_knowledge,
                    "knowledge_text": away_knowledge_text,
                    "knowledge_vector": away_knowledge_vector.tolist(),
                    "form_points": float(away_form_sum),
                    "goal_diff_form": float(away_gd_mean),
                    "last_context": context.tolist(),
                    "expert_features": away_expert_features,
                }

                home_starters, home_participants, home_lineup_names = extract_lineup_player_sets(lineups, home_team)
                away_starters, away_participants, away_lineup_names = extract_lineup_player_sets(lineups, away_team)
                home_minutes_by_player = {player_id: float(home_features.get(player_id, np.zeros(1))[0]) for player_id in home_participants}
                away_minutes_by_player = {player_id: float(away_features.get(player_id, np.zeros(1))[0]) for player_id in away_participants}
                update_player_availability(home_state.player_states, match_index, home_starters, home_participants, home_minutes_by_player, home_lineup_names)
                update_player_availability(away_state.player_states, match_index, away_starters, away_participants, away_minutes_by_player, away_lineup_names)

                home_points_gained = 3.0 if home_score > away_score else 1.0 if home_score == away_score else 0.0
                away_points_gained = 3.0 if away_score > home_score else 1.0 if home_score == away_score else 0.0
                home_state.recent_points.append(home_points_gained)
                away_state.recent_points.append(away_points_gained)
                home_state.recent_goal_diff.append(float(home_score - away_score))
                away_state.recent_goal_diff.append(float(away_score - home_score))
                home_state.last_match_date = match_date
                away_state.last_match_date = match_date

    imported_matches: list[dict[str, Any]] = []
    more_matches_path = Path("config/more_matches.txt")
    imported_matches.extend(parse_more_matches_file(more_matches_path))
    structured_match_sources = [
        (
            Path("new_data/football_prediction_dataset/data/processed/all_international_matches.csv"),
            "all_international_matches_csv",
        ),
    ]
    for source_path, source_name in structured_match_sources:
        imported_matches.extend(parse_structured_match_results(source_path, source_name))
    imported_matches.extend(load_data_final_match_rows(Path.cwd()))
    imported_matches.extend(load_local_match_stat_rows(Path.cwd()))
    if imported_matches:
        imported_matches = sorted(
            imported_matches,
            key=lambda row: (
                row["date"],
                row["home_team"],
                row["away_team"],
                -sum(float(value) for value in row.get("regression_mask", [])),
                row["source"],
            ),
        )
        deduped_matches: list[dict[str, Any]] = []
        seen_match_keys: set[tuple[str, str, str, int, int]] = set()
        for row in imported_matches:
            home_score = int(row["home_score"])
            away_score = int(row["away_score"])
            if home_score < 0 or away_score < 0 or home_score > 15 or away_score > 15:
                continue
            if normalize_team_name(str(row["home_team"])) == normalize_team_name(str(row["away_team"])):
                continue
            match_key = (
                str(row["date"]),
                normalize_team_name(str(row["home_team"])),
                normalize_team_name(str(row["away_team"])),
                home_score,
                away_score,
            )
            if match_key in seen_match_keys:
                continue
            seen_match_keys.add(match_key)
            deduped_matches.append(row)
        imported_matches = deduped_matches
        (paths.raw_dir / "imported_matches.json").write_text(
            json.dumps(imported_matches, indent=2),
            encoding="utf-8",
        )

    imported_knowledge_cache: dict[tuple[Any, ...], tuple[str, torch.Tensor]] = {}

    def cached_imported_knowledge(
        *,
        team_key: str,
        team_name: str,
        team_knowledge: dict[str, Any],
        squad_names: list[str],
        squad_x: torch.Tensor,
        form_points: float,
        goal_diff_form: float,
        manual_context_row: dict[str, Any],
        formation: str,
    ) -> tuple[str, torch.Tensor]:
        cache_key = (
            team_key,
            str(team_knowledge.get("manager_name") or ""),
            formation,
            tuple(squad_names[:SQUAD_SIZE]),
        )
        cached = imported_knowledge_cache.get(cache_key)
        if cached is not None:
            return cached
        text = summarize_team_knowledge_text(
            team_name=team_name,
            team_knowledge=team_knowledge,
            squad_player_names=squad_names,
            squad_x=squad_x.numpy(),
            form_points=form_points,
            goal_diff_form=goal_diff_form,
            manual_context=manual_context_row,
        )
        vector = torch.tensor(knowledge_encoder.encode_texts([text])[0], dtype=torch.float32)
        imported_knowledge_cache[cache_key] = (text, vector)
        return text, vector

    total_imported_matches = len(imported_matches)
    for imported_idx, row in enumerate(imported_matches, start=1):
        if imported_idx == 1 or imported_idx % 5000 == 0 or imported_idx == total_imported_matches:
            print(
                f"Building imported/data_final samples {imported_idx}/{total_imported_matches}",
                flush=True,
            )
        match_date = datetime.fromisoformat(str(row["date"]))
        home_team = str(row["home_team"])
        away_team = str(row["away_team"])
        home_score = int(row["home_score"])
        away_score = int(row["away_score"])
        home_key = normalize_team_name(home_team)
        away_key = normalize_team_name(away_team)

        home_state = team_states[home_key]
        away_state = team_states[away_key]
        match_index = len(samples)
        if row.get("home_lineup") or row.get("away_lineup"):
            home_snapshot = get_lineup_snapshot(
                starters=list(row.get("home_lineup") or []),
                bench=list(row.get("home_bench") or []),
                player_priors=player_priors,
            )
            away_snapshot = get_lineup_snapshot(
                starters=list(row.get("away_lineup") or []),
                bench=list(row.get("away_bench") or []),
                player_priors=player_priors,
            )
        else:
            home_snapshot = get_team_snapshot(
                team_name=home_team,
                team_key=home_key,
                team_profiles=team_profiles,
                team_state=home_state,
                match_index=match_index,
                player_priors=player_priors,
                team_prior_rosters=team_prior_rosters,
                compact_graph=True,
            )
            away_snapshot = get_team_snapshot(
                team_name=away_team,
                team_key=away_key,
                team_profiles=team_profiles,
                team_state=away_state,
                match_index=match_index,
                player_priors=player_priors,
                team_prior_rosters=team_prior_rosters,
                compact_graph=True,
            )

        home_knowledge = team_knowledge_vector(home_key, tactical_knowledge, coaching_knowledge)
        away_knowledge = team_knowledge_vector(away_key, tactical_knowledge, coaching_knowledge)
        if row.get("home_manager"):
            home_knowledge["manager_name"] = str(row.get("home_manager") or "")
            home_knowledge["coach_known"] = 1.0
            home_knowledge["coach_match_count"] = max(float(home_knowledge.get("coach_match_count", 0.0)), 1.0)
        if row.get("away_manager"):
            away_knowledge["manager_name"] = str(row.get("away_manager") or "")
            away_knowledge["coach_known"] = 1.0
            away_knowledge["coach_match_count"] = max(float(away_knowledge.get("coach_match_count", 0.0)), 1.0)
        home_rank = fifa_map.get(home_key, {}).get("rank", 999.0)
        away_rank = fifa_map.get(away_key, {}).get("rank", 999.0)
        home_points = fifa_map.get(home_key, {}).get("points", 0.0)
        away_points = fifa_map.get(away_key, {}).get("points", 0.0)
        _, home_form_sum = squash(home_state.recent_points)
        _, away_form_sum = squash(away_state.recent_points)
        home_gd_mean, _ = squash(home_state.recent_goal_diff)
        away_gd_mean, _ = squash(away_state.recent_goal_diff)
        home_rest_days = (match_date - home_state.last_match_date).days if home_state.last_match_date else 7
        away_rest_days = (match_date - away_state.last_match_date).days if away_state.last_match_date else 7
        home_manual_context = manual_context.get(home_key, {})
        away_manual_context = manual_context.get(away_key, {})
        home_expert_features = summarize_squad_expert_features(home_snapshot["squad_x"])
        away_expert_features = summarize_squad_expert_features(away_snapshot["squad_x"])
        home_formation = str(row.get("home_formation") or "")
        away_formation = str(row.get("away_formation") or "")
        home_knowledge_text, home_knowledge_vector = cached_imported_knowledge(
            team_key=home_key,
            team_name=home_team,
            team_knowledge=home_knowledge,
            squad_names=home_snapshot["squad_names"],
            form_points=float(home_form_sum),
            goal_diff_form=float(home_gd_mean),
            squad_x=home_snapshot["squad_x"],
            manual_context_row=home_manual_context,
            formation=home_formation,
        )
        away_knowledge_text, away_knowledge_vector = cached_imported_knowledge(
            team_key=away_key,
            team_name=away_team,
            team_knowledge=away_knowledge,
            squad_names=away_snapshot["squad_names"],
            form_points=float(away_form_sum),
            goal_diff_form=float(away_gd_mean),
            squad_x=away_snapshot["squad_x"],
            manual_context_row=away_manual_context,
            formation=away_formation,
        )
        competition_name = safe_lower(str(row.get("competition_name") or ""))
        is_tournament = float(any(token in competition_name for token in ("world cup", "euro", "copa", "cup", "nations league", "friendlies", "friendly")))
        context = build_context_vector(
            home_rank=home_rank,
            away_rank=away_rank,
            home_points=home_points,
            away_points=away_points,
            home_form_sum=home_form_sum,
            away_form_sum=away_form_sum,
            home_gd_mean=home_gd_mean,
            away_gd_mean=away_gd_mean,
            home_rest_days=float(home_rest_days),
            away_rest_days=float(away_rest_days),
            neutral_venue=0.0,
            is_tournament=is_tournament,
            home_knowledge=home_knowledge,
            away_knowledge=away_knowledge,
            home_expert_features=home_expert_features,
            away_expert_features=away_expert_features,
            home_formation=home_formation,
            away_formation=away_formation,
        )

        regression_targets = torch.tensor(row.get("targets", [0.0] * len(REGRESSION_TARGET_NAMES)), dtype=torch.float32)
        regression_mask = torch.tensor(row.get("regression_mask", [0.0] * len(REGRESSION_TARGET_NAMES)), dtype=torch.float32)
        score_mask = torch.tensor(row.get("score_mask", [1.0] * len(SCORE_TARGET_NAMES)), dtype=torch.float32)
        home_scorer_targets = torch.zeros(len(home_snapshot["squad_ids"]), dtype=torch.float32)
        away_scorer_targets = torch.zeros(len(away_snapshot["squad_ids"]), dtype=torch.float32)
        home_scorer_mask = torch.zeros(len(home_snapshot["squad_ids"]), dtype=torch.float32)
        away_scorer_mask = torch.zeros(len(away_snapshot["squad_ids"]), dtype=torch.float32)
        if row.get("home_scorers"):
            home_scorer_targets = build_player_name_targets(home_snapshot["squad_names"], row.get("home_scorers", []))
            home_scorer_mask = build_player_valid_mask(home_snapshot["squad_ids"])
        if row.get("away_scorers"):
            away_scorer_targets = build_player_name_targets(away_snapshot["squad_names"], row.get("away_scorers", []))
            away_scorer_mask = build_player_valid_mask(away_snapshot["squad_ids"])

        sample = {
            "match_id": -1_000_000 - len(samples),
            "date": match_date.isoformat(),
            "home_team": home_team,
            "away_team": away_team,
            "home_x": home_snapshot["x"],
            "home_edge_index": home_snapshot["edge_index"],
            "home_edge_weight": home_snapshot["edge_weight"],
            "away_x": away_snapshot["x"],
            "away_edge_index": away_snapshot["edge_index"],
            "away_edge_weight": away_snapshot["edge_weight"],
            "home_squad_x": home_snapshot["squad_x"],
            "away_squad_x": away_snapshot["squad_x"],
            "home_news": torch.zeros(NEWS_VECTOR_DIM, dtype=torch.float32),
            "away_news": torch.zeros(NEWS_VECTOR_DIM, dtype=torch.float32),
            "home_knowledge": home_knowledge_vector,
            "away_knowledge": away_knowledge_vector,
            "context": context,
            "target_class": torch.tensor(result_class(home_score, away_score), dtype=torch.long),
            "targets": regression_targets,
            "regression_mask": regression_mask,
            "score_targets": torch.tensor([float(home_score), float(away_score)], dtype=torch.float32),
            "score_mask": score_mask,
            "home_scorer_targets": home_scorer_targets,
            "away_scorer_targets": away_scorer_targets,
            "home_scorer_mask": home_scorer_mask,
            "away_scorer_mask": away_scorer_mask,
            "sample_source": str(row.get("source") or "imported_match"),
        }
        samples.append(sample)

        if home_key not in team_profiles:
            team_profiles[home_key] = {
                "team_name": home_team,
                "player_ids": home_snapshot["player_ids"],
                "player_names": home_snapshot["player_names"],
                "x": home_snapshot["x"].tolist(),
                "edge_index": home_snapshot["edge_index"].tolist(),
                "edge_weight": home_snapshot["edge_weight"].tolist(),
                "squad_x": home_snapshot["squad_x"].tolist(),
                "squad_player_ids": home_snapshot["squad_ids"],
                "squad_player_names": home_snapshot["squad_names"],
                "team_knowledge": home_knowledge,
                "knowledge_text": home_knowledge_text,
                "knowledge_vector": home_knowledge_vector.tolist(),
                "form_points": float(home_form_sum),
                "goal_diff_form": float(home_gd_mean),
                "last_context": context.tolist(),
                "expert_features": home_expert_features,
            }
        if away_key not in team_profiles:
            team_profiles[away_key] = {
                "team_name": away_team,
                "player_ids": away_snapshot["player_ids"],
                "player_names": away_snapshot["player_names"],
                "x": away_snapshot["x"].tolist(),
                "edge_index": away_snapshot["edge_index"].tolist(),
                "edge_weight": away_snapshot["edge_weight"].tolist(),
                "squad_x": away_snapshot["squad_x"].tolist(),
                "squad_player_ids": away_snapshot["squad_ids"],
                "squad_player_names": away_snapshot["squad_names"],
                "team_knowledge": away_knowledge,
                "knowledge_text": away_knowledge_text,
                "knowledge_vector": away_knowledge_vector.tolist(),
                "form_points": float(away_form_sum),
                "goal_diff_form": float(away_gd_mean),
                "last_context": context.tolist(),
                "expert_features": away_expert_features,
            }

        home_points_gained = 3.0 if home_score > away_score else 1.0 if home_score == away_score else 0.0
        away_points_gained = 3.0 if away_score > home_score else 1.0 if home_score == away_score else 0.0
        home_state.recent_points.append(home_points_gained)
        away_state.recent_points.append(away_points_gained)
        home_state.recent_goal_diff.append(float(home_score - away_score))
        away_state.recent_goal_diff.append(float(away_score - home_score))
        home_state.last_match_date = match_date
        away_state.last_match_date = match_date

    output_path = paths.processed_dir / "match_graph_dataset.pt"
    torch.save(
        {
            "samples": samples,
            "player_feature_names": PLAYER_FEATURE_NAMES,
            "squad_feature_names": DATASET_SQUAD_FEATURE_NAMES,
            "context_feature_names": CONTEXT_FEATURE_NAMES,
            "news_vector_dim": NEWS_VECTOR_DIM,
            "knowledge_vector_dim": int(knowledge_encoder.output_dim),
            "knowledge_encoder_meta": {
                "mode": knowledge_encoder.mode,
                "vector_dim": int(knowledge_encoder.output_dim),
                "model_path": knowledge_config.model_path,
                "fallback_dim": int(knowledge_config.fallback_dim),
            },
            "regression_target_names": REGRESSION_TARGET_NAMES,
            "score_target_names": SCORE_TARGET_NAMES,
        },
        output_path,
    )
    (paths.processed_dir / "team_profiles.json").write_text(json.dumps(team_profiles), encoding="utf-8")
    print(f"Saved dataset to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--competition-id", action="append", type=int, dest="competition_ids")
    parser.add_argument("--season-limit", type=int, default=4)
    args = parser.parse_args()
    prepare_dataset(args.data_dir, args.competition_ids, args.season_limit)
