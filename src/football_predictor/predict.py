from __future__ import annotations

import argparse
import copy
import json
import math
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import ProjectPaths
from .data.build_dataset import (
    build_context_vector,
    build_player_prior_index,
    player_prior_vector,
    squad_graph_from_squad_matrix,
    summarize_squad_expert_features,
)
from .data.knowledge_text import KnowledgeTextConfig, KnowledgeTextEncoder, summarize_team_knowledge_text
from .data.name_normalization import canonical_person_name, canonical_team_name
from .data.team_context import (
    SQUAD_FEATURE_NAMES,
    apply_manual_player_context,
    collect_team_news_vectors,
    load_manual_team_context,
    merge_team_context,
    normalize_name,
)
from .models.foundation import FootballFoundationModel


TEAM_NAME_ALIASES = {
    "bosnia and herzegovina": "bosnia-herzegovina",
    "bosnia & herzegovina": "bosnia-herzegovina",
    "bosnia herzegovina": "bosnia-herzegovina",
    "czech republic": "czechia",
    "korea republic": "south korea",
    "south korea": "south korea",
    "usa": "united states",
    "u.s.a.": "united states",
    "us": "united states",
    "u.s.": "united states",
    "united states": "united states",
    "united states of america": "united states",
}


def normalize_team_name(name: str) -> str:
    return canonical_team_name(name)


def resolve_team_profile_key(name: str, team_profiles: dict[str, dict]) -> str:
    key = normalize_team_name(name)
    candidates = [key, TEAM_NAME_ALIASES.get(key, "")]
    compact_key = key.replace("-", " ")
    for profile_key, profile in team_profiles.items():
        profile_name = normalize_team_name(str(profile.get("team_name") or profile_key))
        if profile_key in candidates or profile_name in candidates:
            return profile_key
        if profile_key.replace("-", " ") == compact_key or profile_name.replace("-", " ") == compact_key:
            return profile_key
    raise KeyError(f"Team profile not found for '{name}'.")


def ascii_normalize(value: str) -> str:
    return unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii")


def normalize_person_label(name: str) -> str:
    return canonical_person_name(name)


def tokenize_person_name(name: str) -> list[str]:
    return [token for token in normalize_person_label(name).split() if token]


def load_team_profiles(paths: ProjectPaths) -> dict[str, dict]:
    profile_path = paths.processed_dir / "team_profiles.json"
    if not profile_path.exists():
        raise FileNotFoundError("Missing team profiles. Run training or dataset build first.")
    return json.loads(profile_path.read_text(encoding="utf-8"))


def load_scenario_file(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    scenario_path = Path(path)
    if not scenario_path.exists():
        raise FileNotFoundError(f"Scenario file not found: {scenario_path}")
    return json.loads(scenario_path.read_text(encoding="utf-8"))


def align_scenario_blocks_to_teams(scenario: dict[str, Any], home: str, away: str) -> dict[str, Any]:
    aligned = copy.deepcopy(scenario or {})
    home_block = aligned.get("home") or {}
    away_block = aligned.get("away") or {}
    home_tag = normalize_team_name(str(home_block.get("team_name", "")))
    away_tag = normalize_team_name(str(away_block.get("team_name", "")))
    requested_home = normalize_team_name(home)
    requested_away = normalize_team_name(away)
    if home_tag == requested_away and away_tag == requested_home:
        aligned["home"], aligned["away"] = away_block, home_block
    return aligned


def load_compatible_model_state(model: FootballFoundationModel, checkpoint: dict[str, Any]) -> list[str]:
    checkpoint_state = checkpoint.get("ema_model_state", checkpoint["model_state"])
    model_state = model.state_dict()
    compatible_state = {}
    skipped = []
    for key, value in checkpoint_state.items():
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape):
            compatible_state[key] = value
        else:
            skipped.append(key)
    model_state.update(compatible_state)
    model.load_state_dict(model_state, strict=True)
    return skipped


def build_cli_team_context(players: list[str] | None, coach_name: str | None) -> dict[str, Any]:
    context: dict[str, Any] = {}
    if players:
        context["probable_xi"] = [{"player": player.strip(), "start_probability": 0.92} for player in players if player.strip()]
    if coach_name:
        context["coach"] = {"name": coach_name.strip(), "known_match_count": 0}
    return context


def name_match_score(raw_name: str, candidate_name: str) -> float:
    raw_tokens = tokenize_person_name(raw_name)
    candidate_tokens = tokenize_person_name(candidate_name)
    if not raw_tokens or not candidate_tokens:
        return -1.0
    if " ".join(raw_tokens) == " ".join(candidate_tokens):
        return 10.0
    raw_last = raw_tokens[-1]
    candidate_last = candidate_tokens[-1]
    if raw_last != candidate_last:
        return -1.0
    score = 4.0
    raw_first = raw_tokens[0]
    candidate_first = candidate_tokens[0]
    if raw_first == candidate_first:
        score += 3.0
    elif raw_first[0] == candidate_first[0]:
        score += 2.0
    raw_initials = "".join(token[0] for token in raw_tokens[:-1] if token)
    candidate_initials = "".join(token[0] for token in candidate_tokens[:-1] if token)
    common_prefix = 0
    for left, right in zip(raw_initials, candidate_initials):
        if left != right:
            break
        common_prefix += 1
    score += common_prefix * 0.5
    score += 0.2 * len(set(raw_tokens).intersection(candidate_tokens))
    return score


def resolve_player_name(raw_name: str, team_profile: dict[str, Any], player_priors: dict[str, Any]) -> str:
    squad_names = [str(name) for name in team_profile.get("squad_player_names", []) if str(name).strip()]
    # Keep team context membership team-local. Global priors are still used later for
    # feature vectors, but they must not rewrite South Africa players into Mexico
    # players, or vice versa, before scorer filtering.
    _ = player_priors
    exact_candidates = squad_names
    best_name = raw_name
    best_score = -1.0
    for candidate_name in exact_candidates:
        score = name_match_score(raw_name, candidate_name)
        if score > best_score:
            best_score = score
            best_name = candidate_name
    return best_name if best_score >= 4.0 else raw_name


def resolve_team_context_names(team_profile: dict[str, Any], team_context: dict[str, Any], player_priors: dict[str, Any]) -> dict[str, Any]:
    resolved = copy.deepcopy(team_context)
    for key in ("probable_xi", "bench", "injuries"):
        rows = []
        for row in resolved.get(key, []):
            item = dict(row)
            player_name = str(item.get("player", "")).strip()
            if player_name:
                item["player"] = resolve_player_name(player_name, team_profile, player_priors)
            rows.append(item)
        resolved[key] = rows
    return resolved


def inject_context_players(
    team_profile: dict[str, Any],
    squad_x: torch.Tensor,
    team_context: dict[str, Any],
    player_priors: dict[str, Any],
) -> tuple[dict[str, Any], torch.Tensor]:
    updated_profile = copy.deepcopy(team_profile)
    updated_squad_x = squad_x.clone()
    squad_names = list(updated_profile.get("squad_player_names", []))
    squad_ids = list(updated_profile.get("squad_player_ids", []))

    additions: list[tuple[str, float]] = []
    for row in team_context.get("probable_xi", []):
        player_name = str(row.get("player", "")).strip()
        if player_name:
            additions.append((player_name, float(row.get("start_probability", 0.9) or 0.9)))
    for row in team_context.get("bench", []):
        player_name = str(row.get("player", "")).strip()
        if player_name:
            additions.append((player_name, float(row.get("start_probability", 0.25) or 0.25)))

    for player_name, start_probability in additions:
        player_key = normalize_name(player_name)
        if player_key in {normalize_name(name) for name in squad_names}:
            continue
        replace_idx = next((idx for idx, name in enumerate(squad_names) if normalize_name(name) == "unknown"), None)
        if replace_idx is None:
            replace_idx = min(range(len(squad_names)), key=lambda idx: float(updated_squad_x[idx, 5].item()))
        prior = player_prior_vector(player_name, player_priors)
        base = torch.zeros((len(SQUAD_FEATURE_NAMES),), dtype=torch.float32)
        base[0] = min(1.0, max(0.0, start_probability))
        base[1] = 1.0
        base[2] = min(1.0, max(0.15, start_probability))
        base[3] = min(1.0, max(0.2, start_probability * 0.85))
        base[4] = 0.0
        base[5] = 0.6 if start_probability >= 0.5 else 0.35
        row = torch.tensor(list(base.tolist()) + list(prior.tolist()), dtype=torch.float32)
        updated_squad_x[replace_idx] = row
        squad_names[replace_idx] = player_name
        squad_ids[replace_idx] = -1

    updated_profile["squad_player_names"] = squad_names
    updated_profile["squad_player_ids"] = squad_ids
    return updated_profile, updated_squad_x


def reorder_squad_by_context(team_profile: dict[str, Any], squad_x: torch.Tensor, team_context: dict[str, Any]) -> tuple[dict[str, Any], torch.Tensor]:
    probable_order = {
        normalize_name(str(row.get("player", ""))): idx
        for idx, row in enumerate(team_context.get("probable_xi", []))
        if str(row.get("player", "")).strip()
    }
    bench_order = {
        normalize_name(str(row.get("player", ""))): idx
        for idx, row in enumerate(team_context.get("bench", []))
        if str(row.get("player", "")).strip()
    }
    squad_names = list(team_profile.get("squad_player_names", []))
    squad_ids = list(team_profile.get("squad_player_ids", []))
    indices = list(range(len(squad_names)))

    def sort_key(idx: int) -> tuple[int, float]:
        name_key = normalize_name(squad_names[idx])
        if name_key in probable_order:
            return (1000 - probable_order[name_key], float(squad_x[idx, 0].item()))
        if name_key in bench_order:
            return (100 - bench_order[name_key], float(squad_x[idx, 0].item()))
        return (0, float(squad_x[idx, 0].item()))

    indices.sort(key=sort_key, reverse=True)
    reordered_profile = copy.deepcopy(team_profile)
    reordered_profile["squad_player_names"] = [squad_names[idx] for idx in indices]
    reordered_profile["squad_player_ids"] = [squad_ids[idx] for idx in indices]
    return reordered_profile, squad_x[indices]


def apply_team_knowledge_context(team_profile: dict[str, Any], team_context: dict[str, Any]) -> dict[str, Any]:
    knowledge = copy.deepcopy(team_profile.get("team_knowledge", {}))
    tactical = dict(knowledge.get("tactical", {}))
    tactical.update(team_context.get("tactical_overrides", {}))
    knowledge["tactical"] = tactical
    coach = team_context.get("coach", {})
    coach_name = str(coach.get("name", "")).strip()
    if coach_name:
        knowledge["manager_name"] = coach_name
        knowledge["coach_known"] = 1.0
        knowledge["coach_match_count"] = float(coach.get("known_match_count", knowledge.get("coach_match_count", 0.0)) or 0.0)
    return knowledge


def align_context_to_checkpoint(context: torch.Tensor, expected_dim: int) -> torch.Tensor:
    current_dim = int(context.shape[-1])
    if current_dim == expected_dim:
        return context
    if current_dim > expected_dim:
        return context[..., :expected_dim]
    padding = torch.zeros((*context.shape[:-1], expected_dim - current_dim), dtype=context.dtype)
    return torch.cat([context, padding], dim=-1)


def align_feature_tensor(feature_tensor: torch.Tensor, expected_dim: int) -> torch.Tensor:
    current_dim = int(feature_tensor.shape[-1])
    if current_dim == expected_dim:
        return feature_tensor
    if current_dim > expected_dim:
        return feature_tensor[..., :expected_dim]
    padding = torch.zeros((*feature_tensor.shape[:-1], expected_dim - current_dim), dtype=feature_tensor.dtype)
    return torch.cat([feature_tensor, padding], dim=-1)


def poisson_probability(rate: float, goals: int) -> float:
    safe_rate = max(rate, 1e-6)
    return math.exp(-safe_rate) * (safe_rate ** goals) / math.factorial(goals)


def scoreline_distribution(home_rate: float, away_rate: float, max_goals: int = 7, top_k: int = 5) -> list[dict[str, float | int]]:
    outcomes: list[dict[str, float | int]] = []
    for home_goals in range(max_goals + 1):
        home_prob = poisson_probability(home_rate, home_goals)
        for away_goals in range(max_goals + 1):
            away_prob = poisson_probability(away_rate, away_goals)
            outcomes.append(
                {
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "probability": round(home_prob * away_prob, 4),
                }
            )
    outcomes.sort(key=lambda item: float(item["probability"]), reverse=True)
    return outcomes[:top_k]


def coherent_goal_rates(
    raw_home_rate: float,
    raw_away_rate: float,
    expected_metrics: dict[str, float],
) -> tuple[float, float]:
    home_xg = max(float(expected_metrics.get("home_xg", raw_home_rate) or raw_home_rate), 0.05)
    away_xg = max(float(expected_metrics.get("away_xg", raw_away_rate) or raw_away_rate), 0.05)
    raw_home_rate = max(float(raw_home_rate), 0.05)
    raw_away_rate = max(float(raw_away_rate), 0.05)
    home_rate = (0.62 * raw_home_rate) + (0.38 * home_xg)
    away_rate = (0.62 * raw_away_rate) + (0.38 * away_xg)
    ratio = home_xg / max(away_xg, 0.05)
    rate_ratio = home_rate / max(away_rate, 0.05)
    if ratio >= 1.25 and rate_ratio < 1.05:
        total = home_rate + away_rate
        home_share = min(0.72, max(0.52, ratio / (1.0 + ratio)))
        home_rate = total * home_share
        away_rate = total - home_rate
    elif ratio <= 0.8 and rate_ratio > 0.95:
        total = home_rate + away_rate
        home_share = max(0.28, min(0.48, ratio / (1.0 + ratio)))
        home_rate = total * home_share
        away_rate = total - home_rate
    return max(0.15, min(4.2, home_rate)), max(0.15, min(4.2, away_rate))


def normalize_match_profile_consistency(expected_metrics: dict[str, float]) -> dict[str, float]:
    metrics = dict(expected_metrics)
    for prefix in ("home", "away"):
        shots_key = f"{prefix}_shots"
        sot_key = f"{prefix}_shots_on_target"
        if shots_key in metrics and sot_key in metrics:
            shots = max(float(metrics.get(shots_key, 0.0) or 0.0), 0.0)
            sot = max(float(metrics.get(sot_key, 0.0) or 0.0), 0.0)
            metrics[shots_key] = round(shots, 3)
            metrics[sot_key] = round(min(sot, shots), 3)
    if "home_possession_proxy" in metrics and "away_possession_proxy" in metrics:
        home_possession = max(float(metrics["home_possession_proxy"]), 0.0)
        away_possession = max(float(metrics["away_possession_proxy"]), 0.0)
        total = home_possession + away_possession
        if total <= 0:
            home_possession = away_possession = 0.5
        else:
            home_possession /= total
            away_possession = 1.0 - home_possession
        metrics["home_possession_proxy"] = round(home_possession, 4)
        metrics["away_possession_proxy"] = round(away_possession, 4)
        metrics["home_possession_pct"] = round(home_possession * 100.0, 2)
        metrics["away_possession_pct"] = round(away_possession * 100.0, 2)

        home_passes = max(float(metrics.get("home_passes", 0.0) or 0.0), 0.0)
        away_passes = max(float(metrics.get("away_passes", 0.0) or 0.0), 0.0)
        total_passes = home_passes + away_passes
        if total_passes >= 100.0:
            metrics["home_passes"] = round(total_passes * home_possession, 3)
            metrics["away_passes"] = round(total_passes * away_possession, 3)
    return metrics


def build_prediction_explanation(
    home: str,
    away: str,
    probs: torch.Tensor,
    expected_metrics: dict[str, float],
    best_scoreline: dict[str, float | int],
    home_scorers: list[dict[str, float | str]],
    away_scorers: list[dict[str, float | str]],
    home_context: dict[str, Any],
    away_context: dict[str, Any],
    confidence: float,
) -> dict[str, Any]:
    prob_values = [float(probs[0].item()), float(probs[1].item()), float(probs[2].item())]
    labels = [f"{home} win", "draw", f"{away} win"]
    favorite_idx = max(range(3), key=lambda idx: prob_values[idx])
    home_xg = float(expected_metrics.get("home_xg", 0.0))
    away_xg = float(expected_metrics.get("away_xg", 0.0))
    home_possession = float(expected_metrics.get("home_possession_pct", 0.0))
    away_possession = float(expected_metrics.get("away_possession_pct", 0.0))
    top_home_scorer = home_scorers[0]["player"] if home_scorers else "no clear scorer"
    top_away_scorer = away_scorers[0]["player"] if away_scorers else "no clear scorer"
    drivers = [
        f"Most likely result: {labels[favorite_idx]} at {prob_values[favorite_idx] * 100:.1f}%.",
        f"Most likely scoreline: {int(best_scoreline['home_goals'])}-{int(best_scoreline['away_goals'])}.",
        f"Expected xG: {home} {home_xg:.2f}, {away} {away_xg:.2f}.",
        f"Expected possession: {home} {home_possession:.1f}%, {away} {away_possession:.1f}%.",
        f"Top scorer candidates: {home} - {top_home_scorer}; {away} - {top_away_scorer}.",
    ]
    if home_context.get("formation") or away_context.get("formation"):
        drivers.append(f"Formation input used: {home_context.get('formation', '-') or '-'} vs {away_context.get('formation', '-') or '-'}.")
    if confidence < 0.45:
        risk = "Model confidence is low; treat percentages as directional."
    elif confidence < 0.65:
        risk = "Model confidence is medium; lineup and tactical input are influencing the result."
    else:
        risk = "Model confidence is relatively strong for this matchup."
    return {
        "summary": " ".join(drivers[:3]),
        "drivers": drivers,
        "risk_note": risk,
    }


def probable_scorers(team_profile: dict, scorer_probs: torch.Tensor, top_k: int = 5) -> list[dict[str, float | str]]:
    names = team_profile.get("squad_player_names", [])
    ids = team_profile.get("squad_player_ids", [])
    picks: list[dict[str, float | str]] = []
    for idx, probability in enumerate(scorer_probs.tolist()):
        if idx >= len(names) or idx >= len(ids):
            continue
        if normalize_person_label(str(names[idx])) in {"", "unknown"}:
            continue
        picks.append(
            {
                "player": str(names[idx]),
                "anytime_goal_probability": round(float(probability), 4),
            }
        )
    picks.sort(key=lambda item: float(item["anytime_goal_probability"]), reverse=True)
    return picks[:top_k]


def scorer_candidate_context(team_context: dict[str, Any]) -> tuple[set[str], dict[str, str], dict[str, str]]:
    allowed_players: set[str] = set()
    roles: dict[str, str] = {}
    display_names: dict[str, str] = {}

    def add_candidate(row: dict[str, Any]) -> None:
        name = normalize_name(str(row.get("player", "")).strip())
        if not name:
            return
        allowed_players.add(name)
        roles[name] = str(row.get("role", "")).strip().upper()
        display_names[name] = str(row.get("player", "")).strip()

    for row in team_context.get("probable_xi", []):
        add_candidate(row)
    for row in team_context.get("bench", []):
        name = normalize_name(str(row.get("player", "")).strip())
        if not name:
            continue
        allowed_players.add(name)
        roles[name] = str(row.get("role", "")).strip().upper()
        display_names[name] = str(row.get("player", "")).strip()
    return allowed_players, roles, display_names


def match_scorer_context_name(player: str, allowed_players: set[str]) -> str | None:
    player_key = normalize_name(player)
    if not allowed_players:
        return player_key
    if player_key in allowed_players:
        return player_key
    best_key = None
    best_score = -1.0
    for allowed_key in allowed_players:
        score = name_match_score(allowed_key, player)
        if score > best_score:
            best_score = score
            best_key = allowed_key
    return best_key if best_score >= 4.0 else None


def scorer_role_weight(role: str) -> float:
    role_code = str(role or "").upper()
    if role_code == "FW":
        return 1.0
    if role_code == "MF":
        return 0.72
    if role_code == "DF":
        return 0.28
    if role_code == "GK":
        return 0.0
    return 0.6


def filter_and_rank_scorers(
    picks: list[dict[str, float | str]],
    team_context: dict[str, Any],
    top_k: int = 5,
) -> list[dict[str, float | str]]:
    allowed_players, roles, display_names = scorer_candidate_context(team_context)
    ranked: list[dict[str, float | str]] = []
    seen: set[str] = set()
    for item in picks:
        player = str(item.get("player", "")).strip()
        if not player:
            continue
        key = match_scorer_context_name(player, allowed_players)
        if allowed_players and key is None:
            continue
        dedupe_key = key or normalize_name(player)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        role = roles.get(key, "")
        weight = scorer_role_weight(role)
        if weight <= 0.0:
            continue
        adjusted = round(float(item.get("anytime_goal_probability", 0.0)) * weight, 4)
        ranked.append({"player": display_names.get(key, player), "role": role, "anytime_goal_probability": adjusted})
    ranked.sort(key=lambda item: float(item["anytime_goal_probability"]), reverse=True)
    return ranked[:top_k]


def context_only_probable_scorers(
    team_profile: dict[str, Any],
    scorer_probs: torch.Tensor,
    team_context: dict[str, Any],
    top_k: int = 5,
) -> list[dict[str, float | str]]:
    allowed_players, roles, display_names = scorer_candidate_context(team_context)
    if not allowed_players:
        return filter_and_rank_scorers(probable_scorers(team_profile, scorer_probs, top_k=50), team_context, top_k=top_k)

    names = [str(name) for name in team_profile.get("squad_player_names", [])]
    probabilities = scorer_probs.tolist()
    picks: list[dict[str, float | str]] = []
    seen: set[str] = set()
    for allowed_key in allowed_players:
        role = roles.get(allowed_key, "")
        weight = scorer_role_weight(role)
        if weight <= 0.0:
            continue
        best_idx = None
        best_score = -1.0
        display_name = display_names.get(allowed_key, allowed_key)
        for idx, candidate in enumerate(names):
            score = name_match_score(display_name, candidate)
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is not None and best_score >= 4.0 and best_idx < len(probabilities):
            probability = float(probabilities[best_idx]) * weight
        else:
            probability = 0.16 * weight
        dedupe_key = normalize_name(display_name)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        picks.append(
            {
                "player": display_name,
                "role": role,
                "anytime_goal_probability": round(max(0.01, min(0.82, probability)), 4),
            }
        )
    picks.sort(key=lambda item: float(item["anytime_goal_probability"]), reverse=True)
    return picks[:top_k]


def calibrate_scorer_probabilities(
    scorers: list[dict[str, float | str]],
    team_goal_rate: float,
    top_k: int = 5,
) -> list[dict[str, float | str]]:
    if not scorers:
        return scorers
    goal_rate = max(float(team_goal_rate), 0.05)
    individual_cap = min(0.58, max(0.18, 0.30 + 0.16 * goal_rate))
    target_mass = min(1.35, max(0.35, 0.78 * goal_rate))
    adjusted = []
    for item in scorers:
        probability = max(0.0, min(float(item.get("anytime_goal_probability", 0.0)), individual_cap))
        adjusted.append({**item, "anytime_goal_probability": probability})
    mass = sum(float(item["anytime_goal_probability"]) for item in adjusted)
    if mass > target_mass and mass > 0:
        scale = target_mass / mass
        adjusted = [
            {
                **item,
                "anytime_goal_probability": max(0.01, float(item["anytime_goal_probability"]) * scale),
            }
            for item in adjusted
        ]
    adjusted.sort(key=lambda item: float(item["anytime_goal_probability"]), reverse=True)
    return [
        {
            **item,
            "anytime_goal_probability": round(float(item["anytime_goal_probability"]), 4),
        }
        for item in adjusted[:top_k]
    ]


def formation_bias(formation: str | None) -> dict[str, float]:
    text = str(formation or "").strip()
    if text == "4-4-2":
        return {"attack": 0.04, "control": -0.01, "defense": 0.03}
    if text == "4-3-3":
        return {"attack": 0.06, "control": 0.04, "defense": 0.0}
    if text == "4-2-3-1":
        return {"attack": 0.05, "control": 0.05, "defense": 0.02}
    if text == "3-5-2":
        return {"attack": 0.05, "control": 0.06, "defense": -0.01}
    return {"attack": 0.0, "control": 0.0, "defense": 0.0}


def logistic(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def estimate_team_strength(team_profile: dict[str, Any], team_context: dict[str, Any]) -> dict[str, float]:
    knowledge = team_profile.get("team_knowledge", {})
    tactical = knowledge.get("tactical", {}) if isinstance(knowledge, dict) else {}
    expert = team_profile.get("expert_features", {}) or {}
    form_points = float(team_profile.get("form_points", 0.0))
    form_goal_diff = float(team_profile.get("goal_diff_form", 0.0))
    bias = formation_bias(team_context.get("formation"))
    pass_completion = float(tactical.get("pass_completion_rate", 0.82) or 0.82)
    shots = float(tactical.get("shots", 11.0) or 11.0)
    xg = float(tactical.get("xg", 1.25) or 1.25)
    progressive_passes = float(tactical.get("progressive_passes", 75.0) or 75.0)
    direct_speed = float(tactical.get("direct_speed_proxy", 1.0) or 1.0)
    pressing = float(tactical.get("pressing_intensity_proxy", 0.08) or 0.08)
    lineup_strength = float(expert.get("projected_lineup_strength", 0.45) or 0.45)
    bench_strength = float(expert.get("projected_bench_strength", 0.05) or 0.05)
    availability = float(expert.get("availability_index", 0.7) or 0.7)
    attacking = float(expert.get("attacking_potential", 0.18) or 0.18)
    creative = float(expert.get("creative_control", 0.18) or 0.18)
    defensive = float(expert.get("defensive_discipline", 0.22) or 0.22)
    coach_signal = min(float(knowledge.get("coach_match_count", 0.0) or 0.0) / 40.0, 1.0)
    strength = (
        1.4 * lineup_strength
        + 0.6 * availability
        + 0.45 * attacking
        + 0.35 * creative
        + 0.25 * defensive
        + 0.2 * coach_signal
        + 0.03 * form_points
        + 0.06 * form_goal_diff
    )
    control = (
        0.9 * (pass_completion - 0.78)
        + 0.003 * (progressive_passes - 70.0)
        + 0.3 * creative
        + 0.12 * lineup_strength
        + bias["control"]
    )
    attack = 0.12 * shots + 0.85 * xg + 0.6 * attacking + 0.15 * bench_strength + bias["attack"]
    defense = 0.55 * defensive + 0.18 * pressing + 0.12 * availability + bias["defense"]
    tempo = max(0.8, min(1.2, 1.03 - 0.08 * (direct_speed - 1.0) + 0.03 * creative))
    return {
        "strength": strength,
        "control": control,
        "attack": attack,
        "defense": defense,
        "pass_completion": pass_completion,
        "shots_base": shots,
        "xg_base": xg,
        "tempo": tempo,
    }


def estimate_match_profile_from_strength(
    home_profile: dict[str, Any],
    away_profile: dict[str, Any],
    home_context: dict[str, Any],
    away_context: dict[str, Any],
) -> dict[str, float]:
    home = estimate_team_strength(home_profile, home_context)
    away = estimate_team_strength(away_profile, away_context)
    strength_gap = home["strength"] - away["strength"]
    control_gap = home["control"] - away["control"]
    home_possession = max(0.28, min(0.72, logistic(0.22 + 1.8 * control_gap + 0.55 * strength_gap)))
    away_possession = 1.0 - home_possession
    average_completion = (home["pass_completion"] + away["pass_completion"]) / 2.0
    total_passes = max(520.0, min(980.0, 730.0 + 240.0 * (average_completion - 0.8)))
    home_passes = total_passes * home_possession
    away_passes = total_passes * away_possession
    home_shots = max(4.0, min(24.0, home["shots_base"] + 3.6 * strength_gap + 10.0 * (home_possession - 0.5)))
    away_shots = max(3.0, min(20.0, away["shots_base"] - 3.0 * strength_gap + 10.0 * (away_possession - 0.5)))
    home_xg = max(0.35, min(3.6, home["xg_base"] + 0.18 * (home_shots - home["shots_base"]) + 0.45 * strength_gap))
    away_xg = max(0.25, min(3.2, away["xg_base"] + 0.16 * (away_shots - away["shots_base"]) - 0.35 * strength_gap))
    home_sot = min(home_shots, max(1.0, 0.28 * home_shots + 0.55 * home_xg))
    away_sot = min(away_shots, max(0.8, 0.27 * away_shots + 0.52 * away_xg))
    return {
        "home_possession_proxy": home_possession,
        "away_possession_proxy": away_possession,
        "home_passes": home_passes,
        "away_passes": away_passes,
        "home_shots": home_shots,
        "away_shots": away_shots,
        "home_shots_on_target": home_sot,
        "away_shots_on_target": away_sot,
        "home_xg": home_xg,
        "away_xg": away_xg,
        "home_goal_rate": max(0.2, min(4.0, 0.15 + 0.95 * home_xg)),
        "away_goal_rate": max(0.15, min(3.6, 0.12 + 0.95 * away_xg)),
    }


def outcome_probs_from_rates(home_rate: float, away_rate: float, max_goals: int = 8) -> tuple[float, float, float]:
    home_win = draw = away_win = 0.0
    for home_goals in range(max_goals + 1):
        home_prob = poisson_probability(home_rate, home_goals)
        for away_goals in range(max_goals + 1):
            away_prob = poisson_probability(away_rate, away_goals)
            joint = home_prob * away_prob
            if home_goals > away_goals:
                home_win += joint
            elif home_goals == away_goals:
                draw += joint
            else:
                away_win += joint
    total = max(home_win + draw + away_win, 1e-6)
    return home_win / total, draw / total, away_win / total


def scorer_priors_from_squad(team_profile: dict[str, Any], team_context: dict[str, Any], top_k: int = 5) -> list[dict[str, float | str]]:
    rows = np.asarray(team_profile.get("squad_x", []), dtype=np.float32)
    names = list(team_profile.get("squad_player_names", []))
    picks: list[dict[str, float | str]] = []
    allowed_players, roles, display_names = scorer_candidate_context(team_context)
    seen: set[str] = set()
    if rows.ndim != 2:
        return picks
    for idx, row in enumerate(rows):
        if idx >= len(names):
            continue
        name = str(names[idx]).strip()
        if normalize_person_label(name) in {"", "unknown"}:
            continue
        name_key = match_scorer_context_name(name, allowed_players)
        if allowed_players and name_key is None:
            continue
        dedupe_key = name_key or normalize_name(name)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        role = roles.get(name_key, "")
        role_weight = scorer_role_weight(role)
        if role_weight <= 0.0:
            continue
        start_prob = float(row[0]) if row.size > 0 else 0.0
        goals_per90 = float(row[len(SQUAD_FEATURE_NAMES) + 2]) if row.size > len(SQUAD_FEATURE_NAMES) + 2 else 0.0
        xg90 = float(row[len(SQUAD_FEATURE_NAMES) + 4]) if row.size > len(SQUAD_FEATURE_NAMES) + 4 else 0.0
        xa90 = float(row[len(SQUAD_FEATURE_NAMES) + 5]) if row.size > len(SQUAD_FEATURE_NAMES) + 5 else 0.0
        importance = float(row[5]) if row.size > 5 else 0.0
        raw_score = start_prob * ((0.62 * xg90) + (0.46 * goals_per90) + (0.10 * xa90) + (0.08 * importance))
        probability = max(0.01, min(0.82, logistic(-1.65 + 4.4 * raw_score) * role_weight))
        picks.append({"player": display_names.get(name_key, name), "role": role, "anytime_goal_probability": round(probability, 4)})
    picks.sort(key=lambda item: float(item["anytime_goal_probability"]), reverse=True)
    return picks[:top_k]


def build_match_batch(
    home_profile: dict,
    away_profile: dict,
    match_date: str | None,
    news_vector_dim: int,
    knowledge_vector_dim: int,
    neutral_venue: bool = True,
    tournament_flag: bool | None = None,
) -> dict[str, torch.Tensor]:
    home_squad_x = torch.tensor(home_profile["squad_x"], dtype=torch.float32)
    away_squad_x = torch.tensor(away_profile["squad_x"], dtype=torch.float32)
    home_x, home_edge_index, home_edge_weight, _ = squad_graph_from_squad_matrix(home_squad_x)
    away_x, away_edge_index, away_edge_weight, _ = squad_graph_from_squad_matrix(away_squad_x)

    home_knowledge = home_profile.get("team_knowledge", {})
    away_knowledge = away_profile.get("team_knowledge", {})
    home_expert_features = home_profile.get("expert_features", {})
    away_expert_features = away_profile.get("expert_features", {})
    context = build_context_vector(
        home_rank=999.0,
        away_rank=999.0,
        home_points=0.0,
        away_points=0.0,
        home_form_sum=float(home_profile.get("form_points", 0.0)),
        away_form_sum=float(away_profile.get("form_points", 0.0)),
        home_gd_mean=float(home_profile.get("goal_diff_form", 0.0)),
        away_gd_mean=float(away_profile.get("goal_diff_form", 0.0)),
        home_rest_days=7.0,
        away_rest_days=7.0,
        neutral_venue=1.0 if neutral_venue else 0.0,
        is_tournament=(
            1.0
            if tournament_flag is True
            else 0.0
            if tournament_flag is False
            else 1.0 if match_date and datetime.fromisoformat(match_date).year >= 2026 else 0.0
        ),
        home_knowledge=home_knowledge,
        away_knowledge=away_knowledge,
        home_expert_features=home_expert_features,
        away_expert_features=away_expert_features,
        home_formation=str(home_profile.get("input_formation") or ""),
        away_formation=str(away_profile.get("input_formation") or ""),
    )
    return {
        "home_x": home_x,
        "home_edge_index": home_edge_index,
        "home_edge_weight": home_edge_weight,
        "home_batch_index": torch.zeros(home_x.size(0), dtype=torch.long),
        "away_x": away_x,
        "away_edge_index": away_edge_index,
        "away_edge_weight": away_edge_weight,
        "away_batch_index": torch.zeros(away_x.size(0), dtype=torch.long),
        "home_squad_x": home_squad_x,
        "home_squad_batch_index": torch.zeros(home_squad_x.size(0), dtype=torch.long),
        "away_squad_x": away_squad_x,
        "away_squad_batch_index": torch.zeros(away_squad_x.size(0), dtype=torch.long),
        "home_news": torch.zeros((1, news_vector_dim), dtype=torch.float32),
        "away_news": torch.zeros((1, news_vector_dim), dtype=torch.float32),
        "home_knowledge": torch.tensor([home_profile.get("knowledge_vector", [0.0] * knowledge_vector_dim)], dtype=torch.float32),
        "away_knowledge": torch.tensor([away_profile.get("knowledge_vector", [0.0] * knowledge_vector_dim)], dtype=torch.float32),
        "context": context.unsqueeze(0),
    }


def generate_prediction_payload(
    data_dir: str,
    home: str,
    away: str,
    match_date: str | None,
    use_live_news: bool = True,
    scenario_file: str | None = None,
    scenario_override: dict[str, Any] | None = None,
    home_players: list[str] | None = None,
    away_players: list[str] | None = None,
    home_coach: str | None = None,
    away_coach: str | None = None,
    neutral_venue: bool = True,
    tournament_flag: bool | None = None,
) -> dict[str, Any]:
    paths = ProjectPaths(Path(data_dir))
    checkpoint = torch.load(paths.models_dir / "football_foundation.pt", map_location="cpu", weights_only=False)
    model = FootballFoundationModel(
        node_dim=len(checkpoint["player_feature_names"]),
        squad_dim=len(checkpoint["squad_feature_names"]),
        context_dim=len(checkpoint["context_feature_names"]),
        news_dim=int(checkpoint["news_vector_dim"]),
        knowledge_dim=int(checkpoint.get("knowledge_vector_dim", 1) or 1),
        hidden_dim=int((checkpoint.get("train_config") or {}).get("hidden_dim", 64)),
        regression_dim=len(checkpoint["regression_target_names"]),
        dropout=float((checkpoint.get("train_config") or {}).get("dropout", 0.1)),
        num_experts=int((checkpoint.get("train_config") or {}).get("num_experts", 4)),
        transformer_layers=int((checkpoint.get("train_config") or {}).get("transformer_layers", 2)),
        attention_heads=int((checkpoint.get("train_config") or {}).get("attention_heads", 4)),
        knowledge_dropout=float((checkpoint.get("train_config") or {}).get("knowledge_dropout", 0.1)),
    )
    skipped_checkpoint_keys = load_compatible_model_state(model, checkpoint)
    model.eval()

    team_profiles = load_team_profiles(paths)
    home_profile = copy.deepcopy(team_profiles[resolve_team_profile_key(home, team_profiles)])
    away_profile = copy.deepcopy(team_profiles[resolve_team_profile_key(away, team_profiles)])
    player_priors = build_player_prior_index(paths)
    knowledge_vector_dim = int(checkpoint.get("knowledge_vector_dim", 1) or 1)
    manual_context = load_manual_team_context(paths)
    scenario = load_scenario_file(scenario_file)
    if scenario_override:
        scenario = dict(scenario_override)
    scenario = align_scenario_blocks_to_teams(scenario, home, away)
    cli_home_context = build_cli_team_context(home_players, home_coach)
    cli_away_context = build_cli_team_context(away_players, away_coach)
    home_context = merge_team_context(manual_context.get(normalize_team_name(home), {}), scenario.get("home"))
    away_context = merge_team_context(manual_context.get(normalize_team_name(away), {}), scenario.get("away"))
    home_context = merge_team_context(home_context, cli_home_context)
    away_context = merge_team_context(away_context, cli_away_context)
    home_context = resolve_team_context_names(home_profile, home_context, player_priors)
    away_context = resolve_team_context_names(away_profile, away_context, player_priors)

    if "neutral_venue" in scenario:
        neutral_venue = bool(scenario.get("neutral_venue"))
    if "is_tournament" in scenario:
        tournament_flag = bool(scenario.get("is_tournament"))

    home_profile["team_knowledge"] = apply_team_knowledge_context(home_profile, home_context)
    away_profile["team_knowledge"] = apply_team_knowledge_context(away_profile, away_context)
    home_squad_x = torch.tensor(home_profile["squad_x"], dtype=torch.float32)
    away_squad_x = torch.tensor(away_profile["squad_x"], dtype=torch.float32)
    home_profile, home_squad_x = inject_context_players(home_profile, home_squad_x, home_context, player_priors)
    away_profile, away_squad_x = inject_context_players(away_profile, away_squad_x, away_context, player_priors)
    home_squad_x = apply_manual_player_context(home_profile, home_squad_x, home_context)
    away_squad_x = apply_manual_player_context(away_profile, away_squad_x, away_context)
    home_profile, home_squad_x = reorder_squad_by_context(home_profile, home_squad_x, home_context)
    away_profile, away_squad_x = reorder_squad_by_context(away_profile, away_squad_x, away_context)
    home_profile["squad_x"] = home_squad_x.tolist()
    away_profile["squad_x"] = away_squad_x.tolist()
    home_profile["input_formation"] = str(home_context.get("formation") or "")
    away_profile["input_formation"] = str(away_context.get("formation") or "")
    home_profile["expert_features"] = summarize_squad_expert_features(home_squad_x.numpy())
    away_profile["expert_features"] = summarize_squad_expert_features(away_squad_x.numpy())
    batch = build_match_batch(
        home_profile,
        away_profile,
        match_date,
        int(checkpoint["news_vector_dim"]),
        knowledge_vector_dim,
        neutral_venue=neutral_venue,
        tournament_flag=tournament_flag,
    )
    batch["context"] = align_context_to_checkpoint(batch["context"], len(checkpoint["context_feature_names"]))
    batch["home_squad_x"] = home_squad_x
    batch["away_squad_x"] = away_squad_x
    knowledge_meta = checkpoint.get("knowledge_encoder_meta", {})
    knowledge_config = KnowledgeTextConfig.from_file(Path("config/knowledge_config.json"))
    if not knowledge_config.model_path and knowledge_meta.get("model_path"):
        knowledge_config.model_path = str(knowledge_meta.get("model_path"))
    if int(knowledge_meta.get("fallback_dim", 0) or 0) > 0:
        knowledge_config.fallback_dim = int(knowledge_meta["fallback_dim"])
    knowledge_encoder = KnowledgeTextEncoder(knowledge_config)
    home_knowledge_text = summarize_team_knowledge_text(
        team_name=home_profile.get("team_name", home),
        team_knowledge=home_profile.get("team_knowledge", {}),
        squad_player_names=home_profile.get("squad_player_names", []),
        squad_x=batch["home_squad_x"].numpy(),
        form_points=float(home_profile.get("form_points", 0.0)),
        goal_diff_form=float(home_profile.get("goal_diff_form", 0.0)),
        manual_context=home_context,
    )
    away_knowledge_text = summarize_team_knowledge_text(
        team_name=away_profile.get("team_name", away),
        team_knowledge=away_profile.get("team_knowledge", {}),
        squad_player_names=away_profile.get("squad_player_names", []),
        squad_x=batch["away_squad_x"].numpy(),
        form_points=float(away_profile.get("form_points", 0.0)),
        goal_diff_form=float(away_profile.get("goal_diff_form", 0.0)),
        manual_context=away_context,
    )
    knowledge_vectors = knowledge_encoder.encode_texts([home_knowledge_text, away_knowledge_text])
    batch["home_knowledge"] = align_feature_tensor(torch.tensor([knowledge_vectors[0]], dtype=torch.float32), knowledge_vector_dim)
    batch["away_knowledge"] = align_feature_tensor(torch.tensor([knowledge_vectors[1]], dtype=torch.float32), knowledge_vector_dim)
    home_news_titles: list[str] = []
    away_news_titles: list[str] = []
    if use_live_news:
        home_news, away_news, home_news_titles, away_news_titles = collect_team_news_vectors(home, away, manual_context)
        batch["home_news"] = home_news.unsqueeze(0)
        batch["away_news"] = away_news.unsqueeze(0)

    with torch.no_grad():
        outputs = model(batch)
        outcome_temperature = max(float(checkpoint.get("outcome_temperature", 1.0) or 1.0), 0.2)
        probs = torch.softmax(outputs["logits"] / outcome_temperature, dim=-1)[0]
        regression_mean = torch.tensor(checkpoint.get("regression_mean", [0.0] * len(checkpoint["regression_target_names"])), dtype=torch.float32)
        regression_std = torch.tensor(checkpoint.get("regression_std", [1.0] * len(checkpoint["regression_target_names"])), dtype=torch.float32)
        reg = outputs["regression"][0] * regression_std + regression_mean
        score_rates = outputs["score_rates"][0]
        home_scorer_probs = torch.sigmoid(outputs["home_scorer_logits"])
        away_scorer_probs = torch.sigmoid(outputs["away_scorer_logits"])
        confidence = float(outputs["confidence"][0].item())

    expected_metrics = {
        name: round(float(reg[idx].item()), 3)
        for idx, name in enumerate(checkpoint["regression_target_names"])
    }
    home_goal_rate = float(score_rates[0].item())
    away_goal_rate = float(score_rates[1].item())
    heuristic_profile = estimate_match_profile_from_strength(home_profile, away_profile, home_context, away_context)
    implausible_profile = (
        expected_metrics.get("home_passes", 0.0) < 50.0
        or expected_metrics.get("away_passes", 0.0) < 50.0
        or expected_metrics.get("home_shots", 0.0) < 1.0
        or expected_metrics.get("away_shots", 0.0) < 1.0
        or not 0.9 <= (
            expected_metrics.get("home_possession_proxy", 0.0)
            + expected_metrics.get("away_possession_proxy", 0.0)
        ) <= 1.1
    )
    if implausible_profile:
        for key in (
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
        ):
            expected_metrics[key] = round(float(heuristic_profile[key]), 3)
        home_goal_rate = float(heuristic_profile["home_goal_rate"])
        away_goal_rate = float(heuristic_profile["away_goal_rate"])
    expected_metrics = normalize_match_profile_consistency(expected_metrics)
    home_goal_rate, away_goal_rate = coherent_goal_rates(home_goal_rate, away_goal_rate, expected_metrics)
    rate_outcomes = outcome_probs_from_rates(home_goal_rate, away_goal_rate)
    model_outcomes = probs
    blend_weight = 0.35 if not implausible_profile else 1.0
    probs = ((1.0 - blend_weight) * model_outcomes) + (blend_weight * torch.tensor(rate_outcomes, dtype=torch.float32))
    probs = probs / probs.sum().clamp_min(1e-6)
    scorelines = scoreline_distribution(home_goal_rate, away_goal_rate)
    best_scoreline = scorelines[0] if scorelines else {"home_goals": 0, "away_goals": 0, "probability": 0.0}
    home_scorers = context_only_probable_scorers(home_profile, home_scorer_probs, home_context)
    away_scorers = context_only_probable_scorers(away_profile, away_scorer_probs, away_context)
    home_scorer_values = [float(item["anytime_goal_probability"]) for item in home_scorers]
    away_scorer_values = [float(item["anytime_goal_probability"]) for item in away_scorers]
    if not home_scorers or max(home_scorer_values) < 0.05 or (max(home_scorer_values) - min(home_scorer_values) < 0.02):
        home_scorers = scorer_priors_from_squad(home_profile, home_context)
    if not away_scorers or max(away_scorer_values) < 0.05 or (max(away_scorer_values) - min(away_scorer_values) < 0.02):
        away_scorers = scorer_priors_from_squad(away_profile, away_context)
    home_scorers = calibrate_scorer_probabilities(home_scorers, home_goal_rate)
    away_scorers = calibrate_scorer_probabilities(away_scorers, away_goal_rate)
    explanation = build_prediction_explanation(
        home=home,
        away=away,
        probs=probs,
        expected_metrics=expected_metrics,
        best_scoreline=best_scoreline,
        home_scorers=home_scorers,
        away_scorers=away_scorers,
        home_context=home_context,
        away_context=away_context,
        confidence=confidence,
    )

    return {
        "home_team": home,
        "away_team": away,
        "date": match_date,
        "win_draw_loss": {
            "home_win": round(float(probs[0].item()), 4),
            "draw": round(float(probs[1].item()), 4),
            "away_win": round(float(probs[2].item()), 4),
        },
        "expected_match_profile": {
            **expected_metrics,
            "expected_home_goals": round(home_goal_rate, 3),
            "expected_away_goals": round(away_goal_rate, 3),
        },
        "probable_scoreline": {
            "home_goals": int(best_scoreline["home_goals"]),
            "away_goals": int(best_scoreline["away_goals"]),
            "probability": float(best_scoreline["probability"]),
        },
        "top_scorelines": scorelines,
        "probable_scorers": {
            "home": home_scorers,
            "away": away_scorers,
        },
        "confidence": round(confidence, 4),
        "prediction_explanation": explanation,
        "squad_context": {
            "home_coach": home_profile.get("team_knowledge", {}).get("manager_name", ""),
            "away_coach": away_profile.get("team_knowledge", {}).get("manager_name", ""),
            "home_formation": home_context.get("formation", ""),
            "away_formation": away_context.get("formation", ""),
            "home_manual_injuries": home_context.get("injuries", []),
            "away_manual_injuries": away_context.get("injuries", []),
            "home_probable_xi": home_context.get("probable_xi", []),
            "away_probable_xi": away_context.get("probable_xi", []),
            "home_bench": home_context.get("bench", []),
            "away_bench": away_context.get("bench", []),
        },
        "expert_team_features": {
            "home": home_profile.get("expert_features", {}),
            "away": away_profile.get("expert_features", {}),
        },
        "profile_generation": {
            "used_heuristic_match_profile": implausible_profile,
            "outcome_temperature": round(float(checkpoint.get("outcome_temperature", 1.0) or 1.0), 4),
            "model_architecture_version": int(checkpoint.get("model_architecture_version", 1) or 1),
            "skipped_checkpoint_keys": skipped_checkpoint_keys[:8],
            "consistency_layer": "goal_rates_blended_with_xg_and_scorer_caps",
        },
        "news_context": {
            "home_headlines": home_news_titles[:8],
            "away_headlines": away_news_titles[:8],
        },
        "knowledge_context": {
            "home_summary": home_knowledge_text,
            "away_summary": away_knowledge_text,
            "encoder_mode": knowledge_encoder.mode,
        },
    }


def predict(
    data_dir: str,
    home: str,
    away: str,
    match_date: str | None,
    use_live_news: bool = True,
    scenario_file: str | None = None,
    scenario_override: dict[str, Any] | None = None,
    home_players: list[str] | None = None,
    away_players: list[str] | None = None,
    home_coach: str | None = None,
    away_coach: str | None = None,
    neutral_venue: bool = True,
    tournament_flag: bool | None = None,
) -> dict[str, Any]:
    payload = generate_prediction_payload(
        data_dir=data_dir,
        home=home,
        away=away,
        match_date=match_date,
        use_live_news=use_live_news,
        scenario_file=scenario_file,
        scenario_override=scenario_override,
        home_players=home_players,
        away_players=away_players,
        home_coach=home_coach,
        away_coach=away_coach,
        neutral_venue=neutral_venue,
        tournament_flag=tournament_flag,
    )
    print(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--home", required=True)
    parser.add_argument("--away", required=True)
    parser.add_argument("--date")
    parser.add_argument("--no-live-news", action="store_true")
    parser.add_argument("--scenario-file")
    parser.add_argument("--home-player", action="append", dest="home_players")
    parser.add_argument("--away-player", action="append", dest="away_players")
    parser.add_argument("--home-coach")
    parser.add_argument("--away-coach")
    parser.add_argument("--neutral-venue", action="store_true")
    parser.add_argument("--not-neutral-venue", action="store_true")
    parser.add_argument("--tournament", action="store_true")
    parser.add_argument("--not-tournament", action="store_true")
    args = parser.parse_args()
    tournament_flag = None
    if args.tournament:
        tournament_flag = True
    elif args.not_tournament:
        tournament_flag = False
    neutral_venue = True
    if args.not_neutral_venue:
        neutral_venue = False
    elif args.neutral_venue:
        neutral_venue = True
    predict(
        args.data_dir,
        args.home,
        args.away,
        args.date,
        use_live_news=not args.no_live_news,
        scenario_file=args.scenario_file,
        home_players=args.home_players,
        away_players=args.away_players,
        home_coach=args.home_coach,
        away_coach=args.away_coach,
        neutral_venue=neutral_venue,
        tournament_flag=tournament_flag,
    )
