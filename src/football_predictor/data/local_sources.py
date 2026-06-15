from __future__ import annotations

import ast
import json
import re
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .name_normalization import TEAM_CODE_TOKENS, canonical_person_name, canonical_team_name, clean_display_name


SQUAD_PRIOR_DIM = 8
STAT_TARGET_DIM = 10
SCORE_TARGET_DIM = 2


def normalize_team_name(name: str) -> str:
    return canonical_team_name(name)


TEAM_CODE_SUFFIX_PATTERN = re.compile(r"\s+[a-z]{2,4}$", re.IGNORECASE)
TEAM_CODE_PREFIX_PATTERN = re.compile(r"^[a-z]{2,4}\s+", re.IGNORECASE)


def clean_team_label(value: Any) -> str:
    text = clean_display_name(str(value or "").replace("‏", ""))
    text = text.replace("Rep. of Ireland", "Republic of Ireland")
    text = text.replace("N. Macedonia", "North Macedonia")
    tokens = text.split()
    if len(tokens) > 1 and tokens[0].lower() in TEAM_CODE_TOKENS:
        tokens = tokens[1:]
    if len(tokens) > 1 and tokens[-1].lower() in TEAM_CODE_TOKENS:
        tokens = tokens[:-1]
    return " ".join(tokens).strip()


def normalize_person_name(name: str | None) -> str:
    return canonical_person_name(name)


def parse_number(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, float) and np.isnan(value):
        return 0.0
    text = str(value).strip().replace(",", "")
    if not text or text.lower() == "nan":
        return 0.0
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else 0.0


def parse_percent(value: Any) -> float:
    numeric = parse_number(value)
    if numeric > 1.5:
        return numeric / 100.0
    return numeric


def safe_divide(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-6:
        return 0.0
    return numerator / denominator


def _weighted_collapse(rows: dict[str, list[tuple[np.ndarray, float]]]) -> dict[str, np.ndarray]:
    collapsed: dict[str, np.ndarray] = {}
    for key, values in rows.items():
        total_weight = float(sum(weight for _, weight in values))
        if total_weight <= 0.0:
            continue
        total = np.sum([vector * weight for vector, weight in values], axis=0)
        merged = (total / total_weight).astype(np.float32)
        merged[0] = max(vector[0] for vector, _ in values)
        merged[1] = max(vector[1] for vector, _ in values)
        merged[-1] = max(vector[-1] for vector, _ in values)
        collapsed[key] = merged
    return collapsed


def _add_roster_row(
    rosters: dict[str, dict[str, tuple[str, np.ndarray]]],
    team_name: str,
    player_name: str,
    vector: np.ndarray,
) -> None:
    team_key = normalize_team_name(team_name)
    player_key = normalize_person_name(player_name)
    if not team_key or not player_key:
        return
    existing = rosters[team_key].get(player_key)
    if existing is None or vector[-1] >= existing[1][-1]:
        rosters[team_key][player_key] = (str(player_name).strip(), vector.astype(np.float32))


def _read_csv_from_zip(zip_path: Path, member_name: str, **kwargs: Any) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(member_name) as handle:
            return pd.read_csv(handle, **kwargs)


def load_updated_player_priors(project_root: Path) -> tuple[dict[str, np.ndarray], dict[str, list[tuple[str, np.ndarray]]]]:
    """Read every local player-stat source that can improve squad/scorer priors."""
    weighted: dict[str, list[tuple[np.ndarray, float]]] = defaultdict(list)
    rosters: dict[str, dict[str, tuple[str, np.ndarray]]] = defaultdict(dict)

    candidate_frames: list[pd.DataFrame] = []
    seen_hashes: set[int] = set()
    for path in sorted((project_root / "data_updated_").glob("data*/players_data-*.csv")):
        frame = pd.read_csv(path)
        frame_hash = hash(tuple(frame.columns)) ^ int(frame.shape[0])
        if frame_hash in seen_hashes:
            continue
        seen_hashes.add(frame_hash)
        candidate_frames.append(frame)
    archive1 = project_root / "data_updated_" / "archive (1).zip"
    if archive1.exists():
        try:
            candidate_frames.append(_read_csv_from_zip(archive1, "players_data-2025_2026.csv"))
        except Exception:
            pass

    for frame in candidate_frames:
        for row in frame.fillna(0).to_dict("records"):
            player_name = str(row.get("Player") or "").strip()
            player_key = normalize_person_name(player_name)
            if not player_key:
                continue
            minutes = parse_number(row.get("Min"))
            starts = parse_number(row.get("Starts"))
            nineties = max(parse_number(row.get("90s")), safe_divide(minutes, 90.0), 1.0)
            goals = parse_number(row.get("Gls"))
            assists = parse_number(row.get("Ast"))
            shots = parse_number(row.get("Sh"))
            cards = parse_number(row.get("CrdY")) + parse_number(row.get("CrdR")) + parse_number(row.get("CrdY_stats_misc")) + parse_number(row.get("CrdR_stats_misc"))
            vector = np.array(
                [
                    min(minutes / 3000.0, 1.5),
                    min(starts / 38.0, 1.5),
                    goals / nineties,
                    assists / nineties,
                    max(0.0, shots / nineties * 0.10),
                    0.0,
                    cards / nineties,
                    0.72,
                ],
                dtype=np.float32,
            )
            weight = max(minutes, 1.0)
            weighted[player_key].append((vector, weight))
            _add_roster_row(rosters, str(row.get("Squad") or ""), player_name, vector)

    understat_frames: list[pd.DataFrame] = []
    archive4 = project_root / "data_updated_" / "archive (4).zip"
    if archive4.exists():
        try:
            understat_frames.append(_read_csv_from_zip(archive4, "player_stats.csv"))
        except Exception:
            pass
    for path in sorted((project_root / "new_data").glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        if isinstance(payload, list):
            understat_frames.append(pd.DataFrame(payload))
    for path in sorted((project_root / "data" / "external" / "understat").glob("*_players.csv")):
        try:
            understat_frames.append(pd.read_csv(path))
        except Exception:
            pass
    for path in sorted((project_root / "data" / "external" / "imports" / "understat").glob("*_players.json")):
        try:
            understat_frames.append(pd.read_json(path, encoding="utf-8-sig"))
        except Exception:
            pass

    for frame in understat_frames:
        for row in frame.fillna(0).to_dict("records"):
            player_name = str(row.get("player") or row.get("Player") or row.get("name") or "").strip()
            player_key = normalize_person_name(player_name)
            if not player_key:
                continue
            minutes = parse_number(row.get("min") or row.get("Min") or row.get("time"))
            apps = parse_number(row.get("apps") or row.get("Apps") or row.get("games"))
            goals = parse_number(row.get("goals") or row.get("G"))
            assists = parse_number(row.get("a") or row.get("A") or row.get("assists"))
            xg_total = parse_number(row.get("xG") or row.get("xg"))
            xa_total = parse_number(row.get("xA") or row.get("xa"))
            nineties = max(minutes / 90.0, 1.0)
            xg90 = parse_number(row.get("xG90") or row.get("xg90")) or (xg_total / nineties)
            xa90 = parse_number(row.get("xA90") or row.get("xa90")) or (xa_total / nineties)
            cards = parse_number(row.get("yellow")) + parse_number(row.get("red"))
            vector = np.array(
                [
                    min(minutes / 3000.0, 1.5),
                    min(apps / 38.0, 1.5),
                    goals / nineties,
                    assists / nineties,
                    xg90,
                    xa90,
                    cards / nineties,
                    1.0,
                ],
                dtype=np.float32,
            )
            weight = max(minutes, apps * 90.0, 1.0)
            weighted[player_key].append((vector, weight))
            _add_roster_row(rosters, str(row.get("team") or row.get("team_title") or row.get("team_name") or ""), player_name, vector)

    priors = _weighted_collapse(weighted)
    roster_lists = {
        team_key: sorted(players.values(), key=lambda item: float(item[1][0] + item[1][1] + item[1][4]), reverse=True)
        for team_key, players in rosters.items()
    }
    return priors, roster_lists


def _empty_targets() -> tuple[list[float], list[float]]:
    return [0.0] * STAT_TARGET_DIM, [0.0] * STAT_TARGET_DIM


def _parse_data_final_date(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return text
    for fmt in ("%A %B %d, %Y", "%a %B %d, %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            pass
    parts = text.split(" ", 1)
    if len(parts) == 2:
        try:
            return datetime.strptime(parts[1], "%B %d, %Y").date().isoformat()
        except ValueError:
            pass
    return text


def _parse_score_pair(value: Any) -> tuple[int, int] | None:
    match = re.search(r"(\d+)\s*[-–—]\s*(\d+)", str(value or ""))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _lineup_names(lineup: Any, key: str) -> list[str]:
    if not isinstance(lineup, dict):
        return []
    rows = lineup.get(key) or []
    names: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        player_name = clean_display_name(row.get("player"))
        if player_name:
            names.append(player_name)
    return names


def _team_summary_totals(rows: Any) -> dict[str, Any]:
    totals = {
        "shots": 0.0,
        "shots_on_target": 0.0,
        "scorers": [],
        "player_rows": [],
    }
    if not isinstance(rows, list):
        return totals
    scorers: list[str] = []
    player_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        player_name = str(row.get("player") or "").strip()
        goals = int(parse_number(row.get("goals")))
        shots = parse_number(row.get("shots"))
        shots_on_target = parse_number(row.get("shots_on_target"))
        totals["shots"] += shots
        totals["shots_on_target"] += shots_on_target
        if player_name and goals > 0:
            scorers.extend([player_name] * goals)
        if player_name:
            player_rows.append(
                {
                    "player": player_name,
                    "position": str(row.get("position") or ""),
                    "minutes": parse_number(row.get("minutes")),
                    "goals": goals,
                    "assists": parse_number(row.get("assists")),
                    "shots": shots,
                    "shots_on_target": shots_on_target,
                    "cards": parse_number(row.get("cards_yellow")) + parse_number(row.get("cards_red")),
                }
            )
    totals["scorers"] = scorers
    totals["player_rows"] = player_rows
    return totals


def load_data_final_match_rows(project_root: Path) -> list[dict[str, Any]]:
    """Load high-quality FBref match reports from data_mafuz_bhai/data_final.

    These rows are especially useful because they contain pre-match-ish lineup,
    formation, manager, player goal, shot and shots-on-target supervision.
    """
    root = project_root / "data_mafuz_bhai" / "data_final"
    if not root.exists():
        return []

    rows: list[dict[str, Any]] = []
    for match_path in sorted(root.rglob("matches/*.json")):
        try:
            payload = json.loads(match_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, dict):
            continue
        teams = data.get("teams") or {}
        home = teams.get("home") or {}
        away = teams.get("away") or {}
        if not isinstance(home, dict) or not isinstance(away, dict):
            continue
        score_pair = _parse_score_pair(data.get("score"))
        if score_pair is None:
            home_score = parse_number(home.get("score"))
            away_score = parse_number(away.get("score"))
        else:
            home_score, away_score = score_pair

        home_team = clean_team_label(home.get("name"))
        away_team = clean_team_label(away.get("name"))
        match_date = _parse_data_final_date(data.get("date"))
        if not home_team or not away_team or not match_date:
            continue

        home_totals = _team_summary_totals(home.get("summary"))
        away_totals = _team_summary_totals(away.get("summary"))
        targets, mask = _empty_targets()
        targets[2] = float(home_totals["shots"])
        targets[3] = float(away_totals["shots"])
        if STAT_TARGET_DIM >= 10:
            targets[8] = float(home_totals["shots_on_target"])
            targets[9] = float(away_totals["shots_on_target"])
            mask[8] = mask[9] = 1.0
        mask[2] = mask[3] = 1.0

        rows.append(
            {
                "competition_name": match_path.parents[2].name,
                "season": match_path.parents[1].name,
                "date": match_date,
                "home_team": home_team,
                "away_team": away_team,
                "home_score": int(home_score),
                "away_score": int(away_score),
                "targets": targets,
                "regression_mask": mask,
                "score_mask": [1.0, 1.0],
                "home_scorers": home_totals["scorers"],
                "away_scorers": away_totals["scorers"],
                "home_lineup": _lineup_names(home.get("lineup"), "starters"),
                "away_lineup": _lineup_names(away.get("lineup"), "starters"),
                "home_bench": _lineup_names(home.get("lineup"), "subs"),
                "away_bench": _lineup_names(away.get("lineup"), "subs"),
                "home_manager": str(home.get("manager") or "").strip(),
                "away_manager": str(away.get("manager") or "").strip(),
                "home_formation": str(home.get("formation") or "").strip(),
                "away_formation": str(away.get("formation") or "").strip(),
                "home_player_rows": home_totals["player_rows"],
                "away_player_rows": away_totals["player_rows"],
                "source": "data_final_fbref_match_report",
            }
        )
    return rows


def _parse_scorer_list(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except Exception:
        pass
    return [part.strip() for part in re.split(r",|;", text) if part.strip()]


def _date_from_archive3(row: dict[str, Any]) -> str:
    season = str(row.get("season_year") or "").split("/")[0] or "2025"
    day = str(row.get("Date_day") or "").strip()
    hour = str(row.get("Date_hour") or "00:00").strip()
    parsed = pd.to_datetime(f"{day}.{season} {hour}", dayfirst=True, errors="coerce")
    if pd.isna(parsed):
        return f"{season}-01-01"
    return parsed.isoformat()


def load_local_match_stat_rows(project_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    friendlies_path = project_root / "updated_data" / "International_Friendlies " / "results_2024.txt"
    if friendlies_path.exists():
        for raw_line in friendlies_path.read_text(encoding="utf-8").splitlines():
            parts = [part.strip() for part in raw_line.split("\t")]
            if len(parts) < 6 or parts[0] == "Day" or parts[1] == "Date":
                continue
            score_match = re.fullmatch(r"(\d+)\s*[–—−-]\s*(\d+)", parts[4])
            if not score_match:
                continue
            rows.append(
                {
                    "competition_name": "International Friendlies 2024",
                    "date": parts[1],
                    "home_team": clean_team_label(parts[3]),
                    "away_team": clean_team_label(parts[5]),
                    "home_score": int(score_match.group(1)),
                    "away_score": int(score_match.group(2)),
                    "targets": [0.0] * STAT_TARGET_DIM,
                    "regression_mask": [0.0] * STAT_TARGET_DIM,
                    "score_mask": [1.0, 1.0],
                    "home_scorers": [],
                    "away_scorers": [],
                    "source": "updated_data_international_friendlies_2024",
                }
            )

    archive2 = project_root / "data_updated_" / "archive (2).zip"
    if archive2.exists():
        frame = _read_csv_from_zip(archive2, "football_matches.csv")
        for row in frame.to_dict("records"):
            targets, mask = _empty_targets()
            targets[2] = parse_number(row.get("Home_Team_Shots"))
            targets[3] = parse_number(row.get("Away_Team_Shots"))
            targets[4] = parse_number(row.get("Home_Team_Passes"))
            targets[5] = parse_number(row.get("Away_Team_Passes"))
            targets[6] = parse_percent(row.get("Home_Team_Possession"))
            targets[7] = parse_percent(row.get("Away_Team_Possession"))
            for idx in [2, 3, 4, 5, 6, 7]:
                mask[idx] = 1.0
            date = pd.to_datetime(row.get("Date"), dayfirst=True, errors="coerce")
            rows.append(
                {
                    "competition_name": str(row.get("League") or "data_updated_matches"),
                    "date": date.date().isoformat() if not pd.isna(date) else "2025-01-01",
                    "home_team": str(row.get("Home_Team") or ""),
                    "away_team": str(row.get("Away_Team") or ""),
                    "home_score": int(parse_number(row.get("Home_Team_Score"))),
                    "away_score": int(parse_number(row.get("Away_Team_Score"))),
                    "targets": targets,
                    "regression_mask": mask,
                    "score_mask": [1.0, 1.0],
                    "home_scorers": [],
                    "away_scorers": [],
                    "source": "data_updated_archive2_match_stats",
                }
            )

    archive3 = project_root / "data_updated_" / "archive (3).zip"
    if archive3.exists():
        frame = _read_csv_from_zip(
            archive3,
            "Football.csv",
            usecols=[
                "Country",
                "League",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
                "season_year",
                "Date_day",
                "Date_hour",
                "home_team_goals",
                "away_team_goals",
                "expected_goals_xg_home",
                "expected_goals_xg_host",
                "Ball_Possession_Home",
                "Ball_Possession_Host",
                "Goal_Attempts_Home",
                "Goal_Attempts_Host",
                "Total_Passes_Home",
                "Total_Passes_Host",
            ],
            low_memory=False,
        )
        for row in frame.to_dict("records"):
            targets, mask = _empty_targets()
            mappings = [
                (0, "expected_goals_xg_home"),
                (1, "expected_goals_xg_host"),
                (2, "Goal_Attempts_Home"),
                (3, "Goal_Attempts_Host"),
                (4, "Total_Passes_Home"),
                (5, "Total_Passes_Host"),
                (6, "Ball_Possession_Home"),
                (7, "Ball_Possession_Host"),
            ]
            for idx, key in mappings:
                raw = row.get(key)
                if raw is None or (isinstance(raw, float) and np.isnan(raw)):
                    continue
                targets[idx] = parse_percent(raw) if "Possession" in key else parse_number(raw)
                mask[idx] = 1.0
            rows.append(
                {
                    "competition_name": str(row.get("League") or row.get("Country") or "football_csv"),
                    "date": _date_from_archive3(row),
                    "home_team": str(row.get("home_team") or ""),
                    "away_team": str(row.get("away_team") or ""),
                    "home_score": int(parse_number(row.get("home_score"))),
                    "away_score": int(parse_number(row.get("away_score"))),
                    "targets": targets,
                    "regression_mask": mask,
                    "score_mask": [1.0, 1.0],
                    "home_scorers": _parse_scorer_list(row.get("home_team_goals")),
                    "away_scorers": _parse_scorer_list(row.get("away_team_goals")),
                    "source": "data_updated_archive3_football_csv",
                }
            )

    archive4 = project_root / "data_updated_" / "archive (4).zip"
    if archive4.exists():
        frame = _read_csv_from_zip(
            archive4,
            "game_stats.csv",
            usecols=["id", "league", "club_name", "home_away", "xG", "scored", "date"],
        )
        for game_id, group in frame.groupby("id"):
            home = group[group["home_away"] == "h"]
            away = group[group["home_away"] == "a"]
            if home.empty or away.empty:
                continue
            h = home.iloc[0].to_dict()
            a = away.iloc[0].to_dict()
            targets, mask = _empty_targets()
            targets[0] = parse_number(h.get("xG"))
            targets[1] = parse_number(a.get("xG"))
            mask[0] = mask[1] = 1.0
            rows.append(
                {
                    "competition_name": str(h.get("league") or "understat_game_stats"),
                    "date": str(h.get("date") or "2020-01-01"),
                    "home_team": str(h.get("club_name") or ""),
                    "away_team": str(a.get("club_name") or ""),
                    "home_score": int(parse_number(h.get("scored"))),
                    "away_score": int(parse_number(a.get("scored"))),
                    "targets": targets,
                    "regression_mask": mask,
                    "score_mask": [1.0, 1.0],
                    "home_scorers": [],
                    "away_scorers": [],
                    "source": "data_updated_archive4_understat_games",
                }
            )

    return [
        row for row in rows
        if row["home_team"].strip() and row["away_team"].strip()
    ]
