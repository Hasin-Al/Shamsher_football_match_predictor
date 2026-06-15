from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .predict import generate_prediction_payload


TIME_BINS = [
    ("0-15", 7.5, 0.145),
    ("16-30", 22.5, 0.155),
    ("31-45+", 39.0, 0.175),
    ("46-60", 52.5, 0.175),
    ("61-75", 67.5, 0.18),
    ("76-90+", 83.0, 0.17),
]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _team_bin_weights(goal_rate: float, possession_pct: float, is_favorite: bool) -> list[float]:
    weights = [weight for _, _, weight in TIME_BINS]
    possession_delta = (possession_pct - 50.0) / 100.0
    if is_favorite:
        weights = [
            weights[0] + 0.020 + possession_delta * 0.025,
            weights[1] + 0.012 + possession_delta * 0.020,
            weights[2] + 0.006,
            weights[3],
            weights[4] - 0.010,
            weights[5] - 0.028,
        ]
    else:
        weights = [
            weights[0] - 0.012,
            weights[1] - 0.006,
            weights[2],
            weights[3] + 0.006,
            weights[4] + 0.008,
            weights[5] + 0.004,
        ]
    if goal_rate >= 2.2:
        weights[0] += 0.012
        weights[1] += 0.008
        weights[5] -= 0.020
    weights = [max(0.06, weight) for weight in weights]
    total = sum(weights)
    return [weight / total for weight in weights]


def _goal_probability(goal_rate: float, bin_weight: float) -> float:
    return 1.0 - math.exp(-max(goal_rate, 0.0) * bin_weight)


def _scorer_shares(scorers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    usable = [
        {
            "player": str(row.get("player") or "").strip(),
            "probability": max(_safe_float(row.get("anytime_goal_probability")), 0.0),
        }
        for row in scorers
        if str(row.get("player") or "").strip()
    ]
    total = sum(row["probability"] for row in usable)
    if total <= 0:
        return []
    return [
        {
            "player": row["player"],
            "share": row["probability"] / total,
        }
        for row in usable
    ]


def _predict_shots_on_target(shots: float, xg: float, goal_rate: float) -> float:
    shots = max(float(shots), 0.0)
    xg = max(float(xg), 0.0)
    goal_rate = max(float(goal_rate), 0.0)
    base = 0.28 * shots
    chance_quality = 1.15 * xg
    finishing_pressure = 0.25 * goal_rate
    estimate = base + chance_quality + finishing_pressure
    lower = min(shots, max(1.0 if goal_rate > 0.35 else 0.0, 0.18 * shots))
    upper = min(shots, max(lower, 0.62 * shots))
    return max(lower, min(upper, estimate))


def add_shots_on_target_predictions(payload: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(payload)
    profile = dict(enriched.get("expected_match_profile") or {})
    home_rate = _safe_float(profile.get("expected_home_goals"), _safe_float(profile.get("home_xg"), 1.0))
    away_rate = _safe_float(profile.get("expected_away_goals"), _safe_float(profile.get("away_xg"), 1.0))
    home_sot = _predict_shots_on_target(
        _safe_float(profile.get("home_shots")),
        _safe_float(profile.get("home_xg")),
        home_rate,
    )
    away_sot = _predict_shots_on_target(
        _safe_float(profile.get("away_shots")),
        _safe_float(profile.get("away_xg")),
        away_rate,
    )
    profile["home_shots_on_target"] = int(round(home_sot))
    profile["away_shots_on_target"] = int(round(away_sot))
    enriched["expected_match_profile"] = profile
    return enriched


def _ordered_goal_minutes(goal_count: int, weights: list[float], trailing: bool) -> list[tuple[str, int, float]]:
    if goal_count <= 0:
        return []
    ranked = sorted(
        [
            {
                "idx": idx,
                "label": label,
                "midpoint": midpoint,
                "weight": weights[idx],
            }
            for idx, (label, midpoint, _) in enumerate(TIME_BINS)
        ],
        key=lambda row: row["weight"],
        reverse=True,
    )
    chosen = sorted(ranked[:goal_count], key=lambda row: row["midpoint"])
    if trailing and goal_count > 0:
        chosen[-1]["midpoint"] = min(88.0, float(chosen[-1]["midpoint"]) + 4.0)
    if not trailing and goal_count >= 2:
        chosen[0]["midpoint"] = max(5.0, float(chosen[0]["midpoint"]) - 3.0)
    return [
        (str(row["label"]), int(round(float(row["midpoint"]))), float(row["weight"]))
        for row in chosen
    ]


def _pick_scorer_for_goal(scorer_shares: list[dict[str, Any]], goal_index: int) -> str:
    if not scorer_shares:
        return "Unknown"
    if goal_index < len(scorer_shares):
        return str(scorer_shares[goal_index]["player"])
    return str(scorer_shares[goal_index % len(scorer_shares)]["player"])


def _scoreline_aligned_timeline(
    payload: dict[str, Any],
    home_weights: list[float],
    away_weights: list[float],
    home_scorer_shares: list[dict[str, Any]],
    away_scorer_shares: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    scoreline = dict(payload.get("probable_scoreline") or {})
    home_team = str(payload.get("home_team") or "Home")
    away_team = str(payload.get("away_team") or "Away")
    home_goals = int(scoreline.get("home_goals") or 0)
    away_goals = int(scoreline.get("away_goals") or 0)
    home_rate = _safe_float((payload.get("expected_match_profile") or {}).get("expected_home_goals"), 1.0)
    away_rate = _safe_float((payload.get("expected_match_profile") or {}).get("expected_away_goals"), 1.0)
    home_minutes = _ordered_goal_minutes(home_goals, home_weights, trailing=home_goals < away_goals)
    away_minutes = _ordered_goal_minutes(away_goals, away_weights, trailing=away_goals < home_goals)
    events: list[dict[str, Any]] = []
    for idx, (window, minute, weight) in enumerate(home_minutes):
        events.append(
            {
                "team": home_team,
                "player": _pick_scorer_for_goal(home_scorer_shares, idx),
                "goal_number_for_team": idx + 1,
                "window": window,
                "minute_estimate": minute,
                "probability_hint": round(_goal_probability(home_rate, weight), 4),
            }
        )
    for idx, (window, minute, weight) in enumerate(away_minutes):
        events.append(
            {
                "team": away_team,
                "player": _pick_scorer_for_goal(away_scorer_shares, idx),
                "goal_number_for_team": idx + 1,
                "window": window,
                "minute_estimate": minute,
                "probability_hint": round(_goal_probability(away_rate, weight), 4),
            }
        )
    events.sort(key=lambda row: int(row["minute_estimate"]))
    home_running = 0
    away_running = 0
    for event in events:
        if event["team"] == home_team:
            home_running += 1
        else:
            away_running += 1
        event["score_after_goal"] = f"{home_running}-{away_running}"
    return events


def add_goal_time_predictions(payload: dict[str, Any], top_events: int = 8) -> dict[str, Any]:
    enriched = add_shots_on_target_predictions(payload)
    profile = dict(payload.get("expected_match_profile") or {})
    home_team = str(payload.get("home_team") or "Home")
    away_team = str(payload.get("away_team") or "Away")
    home_rate = _safe_float(profile.get("expected_home_goals"), _safe_float(profile.get("home_xg"), 1.0))
    away_rate = _safe_float(profile.get("expected_away_goals"), _safe_float(profile.get("away_xg"), 1.0))
    home_possession = _safe_float(profile.get("home_possession_pct"), 50.0)
    away_possession = _safe_float(profile.get("away_possession_pct"), 50.0)
    home_favorite = home_rate >= away_rate
    home_weights = _team_bin_weights(home_rate, home_possession, home_favorite)
    away_weights = _team_bin_weights(away_rate, away_possession, not home_favorite)
    home_scorer_shares = _scorer_shares(list((payload.get("probable_scorers") or {}).get("home") or []))
    away_scorer_shares = _scorer_shares(list((payload.get("probable_scorers") or {}).get("away") or []))

    bins: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    no_goal_probability = math.exp(-(home_rate + away_rate))
    for idx, (label, minute_midpoint, _) in enumerate(TIME_BINS):
        home_prob = _goal_probability(home_rate, home_weights[idx])
        away_prob = _goal_probability(away_rate, away_weights[idx])
        any_prob = 1.0 - ((1.0 - home_prob) * (1.0 - away_prob))
        row = {
            "window": label,
            "minute_midpoint": minute_midpoint,
            "home_goal_probability": round(home_prob, 4),
            "away_goal_probability": round(away_prob, 4),
            "any_goal_probability": round(any_prob, 4),
        }
        bins.append(row)
        for share in home_scorer_shares[:5]:
            events.append(
                {
                    "team": home_team,
                    "player": share["player"],
                    "window": label,
                    "minute_estimate": int(round(minute_midpoint)),
                    "probability": home_prob * float(share["share"]),
                }
            )
        for share in away_scorer_shares[:5]:
            events.append(
                {
                    "team": away_team,
                    "player": share["player"],
                    "window": label,
                    "minute_estimate": int(round(minute_midpoint)),
                    "probability": away_prob * float(share["share"]),
                }
            )

    events.sort(key=lambda item: float(item["probability"]), reverse=True)
    aligned_timeline = _scoreline_aligned_timeline(
        enriched,
        home_weights,
        away_weights,
        home_scorer_shares,
        away_scorer_shares,
    )
    enriched["goal_time_predictions"] = {
        "method": "heuristic_from_expected_goals_possession_and_scorer_probs",
        "note": "Not trained on goal-minute labels yet; replace with a trained time-bin head when linked goal-minute data is available.",
        "no_goal_probability": round(no_goal_probability, 4),
        "scoreline_aligned_timeline": aligned_timeline,
        "time_bins": bins,
        "top_goal_events": [
            {
                **event,
                "probability": round(float(event["probability"]), 4),
            }
            for event in events[:top_events]
        ],
    }
    return enriched


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--home", required=True)
    parser.add_argument("--away", required=True)
    parser.add_argument("--date")
    parser.add_argument("--scenario-file")
    parser.add_argument("--no-live-news", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    payload = generate_prediction_payload(
        data_dir=args.data_dir,
        home=args.home,
        away=args.away,
        match_date=args.date,
        use_live_news=not args.no_live_news,
        scenario_file=args.scenario_file,
    )
    enriched = add_goal_time_predictions(payload)
    text = json.dumps(enriched, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
