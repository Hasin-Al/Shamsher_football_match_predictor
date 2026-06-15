from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .config import ProjectPaths
from .predict import generate_prediction_payload, load_team_profiles, resolve_team_profile_key


COMMON_TEAM_CODE_ALIASES = {
    "ARG": "Argentina",
    "AUS": "Australia",
    "AUT": "Austria",
    "BEL": "Belgium",
    "BRA": "Brazil",
    "CAN": "Canada",
    "COL": "Colombia",
    "CRO": "Croatia",
    "DEN": "Denmark",
    "ECU": "Ecuador",
    "ENG": "England",
    "ESP": "Spain",
    "FRA": "France",
    "GER": "Germany",
    "GHA": "Ghana",
    "IRN": "IR Iran",
    "ISL": "Iceland",
    "ITA": "Italy",
    "JPN": "Japan",
    "KOR": "Korea Republic",
    "MAR": "Morocco",
    "MEX": "Mexico",
    "NED": "Netherlands",
    "NOR": "Norway",
    "PAR": "Paraguay",
    "POR": "Portugal",
    "QAT": "Qatar",
    "SCO": "Scotland",
    "SEN": "Senegal",
    "SUI": "Switzerland",
    "SWE": "Sweden",
    "TUN": "Tunisia",
    "TUR": "Turkiye",
    "URU": "Uruguay",
    "USA": "USA",
}

MINUTE_PATTERN = re.compile(r"^\d{1,3}[’']?$")
FORMATION_PATTERN = re.compile(r"\b\d-\d-\d(?:-\d)?\b")
TEAM_CODE_PATTERN = re.compile(r"^[A-Z]{2,4}$")
COACH_PATTERN = re.compile(r"^\s*coach\s*:\s*(.+)$", re.IGNORECASE)
FORMATION_LINE_PATTERN = re.compile(r"^\s*formation\s*:\s*([0-9\-]+)\s*$", re.IGNORECASE)
POSITION_HEADER_PATTERN = re.compile(r"^(goalkeepers?|defenders?|midfielders?|forwards?|attackers?)\s*:\s*(.*)$", re.IGNORECASE)
POSITION_PLAYER_PATTERN = re.compile(r"^(GK|RB|RWB|CB|LB|LWB|WB|DF|DM|CDM|CM|AM|CAM|LM|RM|MF|LW|RW|ST|CF|FW)\s*:\s*(.+)$", re.IGNORECASE)
SUBSTITUTES_HEADER_PATTERN = re.compile(r"^substitutes?\s*:\s*(.*)$", re.IGNORECASE)
TEAM_HEADING_CLEAN_PATTERN = re.compile(r"^[^\w]*|[^\w\s].*$")
HEAD_COACH_SENTENCE_PATTERN = re.compile(r"\b([A-Z][A-Za-zÀ-ÿ .'\-]+?)\s+is\s+the\s+head\s+coach\s+of\b", re.IGNORECASE)
HEAD_COACH_TEAM_SENTENCE_PATTERN = re.compile(
    r"\b([A-Z][A-Za-zÀ-ÿ .'\-]+?)\s+is\s+the\s+head\s+coach\s+of\s+([A-Z][A-Za-zÀ-ÿ .'\-]+?)(?:\s+for|\s*\(|[,.]|$)",
    re.IGNORECASE,
)


def _clean_lines(raw_text: str) -> list[str]:
    return [" ".join(line.replace("\ufeff", "").split()).strip() for line in raw_text.splitlines()]


def _next_nonempty_index(lines: list[str], start: int) -> int | None:
    for idx in range(start, len(lines)):
        if lines[idx]:
            return idx
    return None


def _find_lineup_sections(lines: list[str]) -> list[tuple[str, int]]:
    sections: list[tuple[str, int]] = []
    for idx, line in enumerate(lines):
        if not TEAM_CODE_PATTERN.fullmatch(line or ""):
            continue
        next_idx = _next_nonempty_index(lines, idx + 1)
        if next_idx is None or lines[next_idx] != "-":
            continue
        next_next_idx = _next_nonempty_index(lines, next_idx + 1)
        if next_next_idx is None or lines[next_next_idx].lower() != "line up":
            continue
        sections.append((line, next_next_idx + 1))
    return sections


def _is_player_name(line: str) -> bool:
    if not line:
        return False
    if line.isdigit():
        return False
    if MINUTE_PATTERN.fullmatch(line):
        return False
    lowered = line.lower()
    if lowered in {"substitutes", "injuries and suspensions", "no sidelined players"}:
        return False
    return any(char.isalpha() for char in line)


def _parse_team_block(lines: list[str], start_idx: int) -> dict[str, Any]:
    probable_xi: list[dict[str, Any]] = []
    bench: list[dict[str, Any]] = []
    injuries: list[dict[str, Any]] = []
    idx = start_idx
    state = "lineup"
    while idx < len(lines):
        line = lines[idx]
        lowered = line.lower()
        if TEAM_CODE_PATTERN.fullmatch(line) and idx > start_idx:
            break
        if lowered == "substitutes":
            state = "bench"
            idx += 1
            continue
        if lowered == "injuries and suspensions":
            state = "injuries"
            idx += 1
            continue
        if lowered == "no sidelined players":
            idx += 1
            continue
        if not line or line == "-":
            idx += 1
            continue
        if state == "injuries":
            if _is_player_name(line):
                injuries.append({"player": line, "status": "out"})
            idx += 1
            continue
        if line.isdigit() or MINUTE_PATTERN.fullmatch(line):
            idx += 1
            continue
        if _is_player_name(line):
            if state == "lineup" and len(probable_xi) < 11:
                probable_xi.append({"player": line, "start_probability": round(max(0.91, 0.99 - (0.008 * len(probable_xi))), 2)})
            elif state == "bench":
                bench.append({"player": line, "start_probability": round(max(0.05, 0.18 - (0.008 * len(bench))), 2)})
        idx += 1
    return {
        "probable_xi": probable_xi,
        "bench": bench,
        "injuries": injuries,
    }


def _strip_club_context(value: str) -> str:
    cleaned = re.sub(r"\([^)]*\)", "", value or "")
    cleaned = re.sub(r"\[[^\]]*\]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;")
    return cleaned


def _clean_coach_name(value: str) -> str:
    cleaned = _strip_club_context(value)
    for prefix in ("Belgian tactician ", "tactician ", "head coach "):
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix):].strip()
    return cleaned


def _apply_global_coach_sentences(
    raw_text: str,
    scenario: dict[str, Any],
    home_team: str,
    away_team: str,
) -> None:
    home_key = home_team.lower()
    away_key = away_team.lower()
    for match in HEAD_COACH_TEAM_SENTENCE_PATTERN.finditer(raw_text):
        coach_name = _clean_coach_name(match.group(1))
        team_text = _strip_club_context(match.group(2)).lower()
        if not coach_name:
            continue
        if home_key in team_text or team_text in home_key:
            scenario.setdefault("home", {}).setdefault("coach", {"name": coach_name, "known_match_count": 0})
        elif away_key in team_text or team_text in away_key:
            scenario.setdefault("away", {}).setdefault("coach", {"name": coach_name, "known_match_count": 0})


def _extract_players_from_segment(text: str) -> list[str]:
    players: list[str] = []
    for part in re.split(r",|;|/|\u2022", text):
        candidate = _strip_club_context(part)
        if not candidate:
            continue
        if candidate.lower() in {"lineup", "squad"}:
            continue
        if any(char.isalpha() for char in candidate):
            players.append(candidate)
    return players


def _extract_position_line_players(text: str) -> list[str]:
    players = _extract_players_from_segment(text)
    if "/" in str(text) and len(players) > 1:
        return [players[-1]]
    return players


def _clean_lineup_player_name(text: str) -> str:
    cleaned = re.sub(r"^\s*\d{1,2}\s*[.)-]?\s*", "", text or "")
    cleaned = re.sub(r"\b(?:GK|RB|RWB|CB|LB|LWB|DF|DM|CDM|CM|AM|CAM|LM|RM|MF|LW|RW|ST|CF|FW)\b", "", cleaned, flags=re.IGNORECASE)
    return _strip_club_context(cleaned).strip(" .-")


def _role_from_position(position: str) -> str:
    position = position.upper()
    if position == "GK":
        return "goalkeepers"
    if position in {"RB", "RWB", "CB", "LB", "LWB", "WB", "DF"}:
        return "defenders"
    if position in {"DM", "CDM", "CM", "AM", "CAM", "LM", "RM", "MF"}:
        return "midfielders"
    return "forwards"


def _append_position_player(
    position: str,
    player: str,
    goalkeepers: list[str],
    defenders: list[str],
    midfielders: list[str],
    forwards: list[str],
) -> None:
    player_name = _clean_lineup_player_name(player)
    if not player_name:
        return
    bucket = _role_from_position(position)
    if bucket == "goalkeepers":
        goalkeepers.append(player_name)
    elif bucket == "defenders":
        defenders.append(player_name)
    elif bucket == "midfielders":
        midfielders.append(player_name)
    else:
        forwards.append(player_name)


def _extract_inline_starting_players(line: str) -> list[str]:
    if ":" not in line or not FORMATION_PATTERN.search(line):
        return []
    after_colon = line.split(":", 1)[1]
    parts = re.split(r"\s+[—–-]\s+|,", after_colon)
    players = []
    for part in parts:
        player = _clean_lineup_player_name(part)
        if player and any(char.isalpha() for char in player):
            players.append(player)
    return players


def _dedupe_players(players: list[str]) -> list[str]:
    unique_players: list[str] = []
    seen: set[str] = set()
    for player in players:
        key = player.lower()
        if key in seen:
            continue
        seen.add(key)
        unique_players.append(player)
    return unique_players


def _formation_role_counts(formation: str) -> tuple[int, int, int]:
    formation_text = str(formation or "").strip()
    parts = [int(part) for part in formation_text.split("-") if part.isdigit()]
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    if len(parts) == 4:
        return parts[0], parts[1] + parts[2], parts[3]
    return 4, 3, 3


def _build_probable_xi_from_positions(
    goalkeepers: list[str],
    defenders: list[str],
    midfielders: list[str],
    forwards: list[str],
    formation: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    gks = _dedupe_players(goalkeepers)
    defs = _dedupe_players(defenders)
    mids = _dedupe_players(midfielders)
    fwds = _dedupe_players(forwards)
    def_count, mid_count, fwd_count = _formation_role_counts(formation)

    xi_names: list[str] = []
    role_lookup: dict[str, str] = {}

    def take(players: list[str], count: int, role: str) -> list[str]:
        chosen = players[:count]
        for player in chosen:
            role_lookup[player.lower()] = role
        return chosen

    xi_names.extend(take(gks, 1, "GK"))
    xi_names.extend(take(defs, def_count, "DF"))
    xi_names.extend(take(mids, mid_count, "MF"))
    xi_names.extend(take(fwds, fwd_count, "FW"))

    all_players = _dedupe_players(gks + defs + mids + fwds)
    if len(xi_names) < 11:
        for player in all_players:
            if player.lower() in {name.lower() for name in xi_names}:
                continue
            inferred_role = "FW" if player in fwds else "MF" if player in mids else "DF" if player in defs else "GK"
            role_lookup[player.lower()] = inferred_role
            xi_names.append(player)
            if len(xi_names) >= 11:
                break

    xi_keys = {player.lower() for player in xi_names}
    bench_names = [player for player in all_players if player.lower() not in xi_keys]

    probable_xi = [
        {
            "player": player,
            "role": role_lookup.get(player.lower(), ""),
            "start_probability": round(max(0.91, 0.99 - (0.008 * idx)), 2),
        }
        for idx, player in enumerate(xi_names[:11])
    ]
    bench = [
        {
            "player": player,
            "role": "FW" if player in fwds else "MF" if player in mids else "DF" if player in defs else "GK",
            "start_probability": round(max(0.05, 0.18 - (0.008 * idx)), 2),
        }
        for idx, player in enumerate(bench_names)
    ]
    return probable_xi, bench


def _find_named_team_section(lines: list[str], team_name: str) -> int | None:
    team_name = str(team_name or "").strip().lower()
    if not team_name:
        return None
    tokens = [token for token in re.split(r"\s+", team_name) if token]
    for idx, line in enumerate(lines):
        lowered = line.lower()
        if re.search(r"\b(?:vs|v)\.?\b", lowered):
            continue
        candidate = _normalize_team_heading_candidate(line).lower()
        if candidate == team_name:
            return idx
        next_idx = _next_nonempty_index(lines, idx + 1)
        if candidate and team_name in candidate and next_idx is not None:
            next_line = lines[next_idx].lower()
            if next_line.startswith("coach:") or next_line.startswith("formation:"):
                return idx
        if team_name in lowered:
            return idx
        if tokens and all(token in lowered for token in tokens[:2]):
            return idx
    return None


def _find_text_split(lines: list[str], away_team: str | None) -> int | None:
    if away_team:
        named_idx = _find_named_team_section(lines, away_team)
        if named_idx is not None:
            return named_idx
    for idx, line in enumerate(lines):
        lowered = line.lower()
        if lowered.endswith("lineup") or lowered.endswith("line up"):
            return idx
    return None


def _normalize_team_heading_candidate(line: str) -> str:
    cleaned = re.sub(r"\([^)]*\)", "", line or "")
    cleaned = TEAM_HEADING_CLEAN_PATTERN.sub("", cleaned).strip()
    return " ".join(cleaned.split())


def _is_team_heading_line(line: str) -> bool:
    candidate = _normalize_team_heading_candidate(line)
    lowered = candidate.lower()
    if re.search(r"\b(?:vs|v)\.?\b", line.lower()):
        return False
    if not candidate:
        return False
    if ":" in line:
        return False
    if lowered in {
        "lineup",
        "line up",
        "injuries and suspensions",
        "no sidelined players",
        "substitutes",
    }:
        return False
    if FORMATION_PATTERN.search(candidate):
        return False
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z .'\-]{1,40}", candidate))


def _find_team_heading_sections(lines: list[str]) -> list[tuple[str, int]]:
    sections: list[tuple[str, int]] = []
    for idx, line in enumerate(lines):
        if not _is_team_heading_line(line):
            continue
        next_idx = _next_nonempty_index(lines, idx + 1)
        if next_idx is None:
            continue
        next_line = lines[next_idx].lower()
        if next_line.startswith("coach:") or next_line.startswith("formation:") or next_line.startswith("goalkeepers:"):
            sections.append((_normalize_team_heading_candidate(line), idx))
    return sections


def _parse_squad_style_block(lines: list[str]) -> dict[str, Any]:
    goalkeepers: list[str] = []
    defenders: list[str] = []
    midfielders: list[str] = []
    forwards: list[str] = []
    bench_players: list[str] = []
    coach_name = ""
    formation = ""
    injuries: list[dict[str, Any]] = []
    current_bucket: str | None = None
    for line in lines:
        if not line:
            continue
        coach_match = COACH_PATTERN.match(line)
        if coach_match:
            coach_name = _strip_club_context(coach_match.group(1))
            continue
        formation_match = FORMATION_LINE_PATTERN.match(line)
        if formation_match:
            formation = formation_match.group(1).strip()
            continue
        if not formation:
            inline_formation = FORMATION_PATTERN.search(line)
            if inline_formation:
                formation = inline_formation.group(0)
        coach_sentence = HEAD_COACH_SENTENCE_PATTERN.search(line)
        if coach_sentence and not coach_name:
            coach_name = _clean_coach_name(coach_sentence.group(1))
            continue
        position_player_match = POSITION_PLAYER_PATTERN.match(line)
        if position_player_match:
            for player in _extract_position_line_players(position_player_match.group(2)):
                _append_position_player(
                    position_player_match.group(1),
                    player,
                    goalkeepers,
                    defenders,
                    midfielders,
                    forwards,
                )
            current_bucket = _role_from_position(position_player_match.group(1))
            continue
        position_match = POSITION_HEADER_PATTERN.match(line)
        if position_match:
            header = position_match.group(1).lower()
            players = _extract_players_from_segment(position_match.group(2))
            if header.startswith("goal"):
                goalkeepers.extend(players)
                current_bucket = "goalkeepers"
            elif header.startswith("def"):
                defenders.extend(players)
                current_bucket = "defenders"
            elif header.startswith("mid"):
                midfielders.extend(players)
                current_bucket = "midfielders"
            else:
                forwards.extend(players)
                current_bucket = "forwards"
            continue
        substitutes_match = SUBSTITUTES_HEADER_PATTERN.match(line)
        if substitutes_match:
            bench_players.extend(_extract_players_from_segment(substitutes_match.group(1)))
            current_bucket = "bench"
            continue
        lowered = line.lower()
        if lowered in {"bench", "likely substitutes", "substitutes", "starting xi", "starting 11"}:
            current_bucket = "bench" if lowered in {"bench", "likely substitutes", "substitutes"} else None
            continue
        if lowered.startswith("injur") or lowered.startswith("susp"):
            if ":" in line and not lowered.startswith("susp"):
                for player in _extract_players_from_segment(line.split(":", 1)[1]):
                    if player.lower() not in {"none", "no"}:
                        injuries.append({"player": player, "status": "out"})
            current_bucket = None
            continue
        if lowered == "no sidelined players":
            continue
        if line.endswith("Lineup") or line.endswith("Line up"):
            continue
        inline_players = _extract_inline_starting_players(line)
        if inline_players:
            def_count, mid_count, fwd_count = _formation_role_counts(formation)
            goalkeepers.extend(inline_players[:1])
            defenders.extend(inline_players[1:1 + def_count])
            midfielders.extend(inline_players[1 + def_count:1 + def_count + mid_count])
            forwards.extend(inline_players[1 + def_count + mid_count:1 + def_count + mid_count + fwd_count])
            continue
        if "," in line and any(char.isalpha() for char in line):
            players = _extract_players_from_segment(line)
            if current_bucket == "goalkeepers":
                goalkeepers.extend(players)
            elif current_bucket == "defenders":
                defenders.extend(players)
            elif current_bucket == "midfielders":
                midfielders.extend(players)
            elif current_bucket == "forwards":
                forwards.extend(players)
            elif current_bucket == "bench":
                bench_players.extend(players)
            continue
        if current_bucket == "bench" and _is_player_name(line):
            bench_players.append(_clean_lineup_player_name(line))
            continue
    probable_xi, bench = _build_probable_xi_from_positions(
        goalkeepers=goalkeepers,
        defenders=defenders,
        midfielders=midfielders,
        forwards=forwards,
        formation=formation,
    )
    if bench_players:
        existing_xi = {row["player"].lower() for row in probable_xi}
        explicit_bench = []
        seen_bench: set[str] = set()
        for idx, player in enumerate(_dedupe_players(bench_players)):
            key = player.lower()
            if key in existing_xi or key in seen_bench:
                continue
            seen_bench.add(key)
            explicit_bench.append(
                {
                    "player": player,
                    "start_probability": round(max(0.05, 0.18 - (0.008 * idx)), 2),
                }
            )
        if explicit_bench:
            bench = explicit_bench
    block: dict[str, Any] = {
        "formation": formation,
        "probable_xi": probable_xi,
        "bench": bench,
        "injuries": injuries,
    }
    if coach_name:
        block["coach"] = {"name": coach_name, "known_match_count": 0}
    return block


def _parse_squad_style_text(
    lines: list[str],
    home_team: str | None,
    away_team: str | None,
    formations: list[str],
) -> dict[str, Any]:
    home_start = _find_named_team_section(lines, home_team or "")
    away_start = _find_named_team_section(lines[home_start + 1:] if home_start is not None else lines, away_team or "")
    if away_start is not None and home_start is not None:
        away_start += home_start + 1
    split_idx = away_start if away_start is not None else _find_text_split(lines, away_team)
    if split_idx is None:
        heading_sections = _find_team_heading_sections(lines)
        if len(heading_sections) >= 2:
            split_idx = heading_sections[1][1]
    if split_idx is None:
        raise ValueError("Could not find two lineup sections in the provided text.")
    home_lines = lines[home_start:split_idx] if home_start is not None and home_start < split_idx else lines[:split_idx]
    away_lines = lines[split_idx:]
    home_block = _parse_squad_style_block(home_lines)
    away_block = _parse_squad_style_block(away_lines)
    if not home_block.get("formation") and formations:
        home_block["formation"] = formations[0]
    if not away_block.get("formation") and len(formations) > 1:
        away_block["formation"] = formations[1]
    if home_team:
        home_block["team_name"] = home_team
    if away_team:
        away_block["team_name"] = away_team
    return {
        "neutral_venue": False,
        "is_tournament": False,
        "home": home_block,
        "away": away_block,
    }


def resolve_team_code(code: str, team_profiles: dict[str, dict], explicit_name: str | None = None) -> str:
    if explicit_name:
        profile = team_profiles[resolve_team_profile_key(explicit_name, team_profiles)]
        return str(profile.get("team_name") or explicit_name)
    code = str(code or "").strip().upper()
    if code in COMMON_TEAM_CODE_ALIASES:
        return COMMON_TEAM_CODE_ALIASES[code]
    for profile in team_profiles.values():
        team_name = str(profile.get("team_name") or "").strip()
        if not team_name:
            continue
        compact = re.sub(r"[^A-Z]", "", team_name.upper())
        if compact.startswith(code):
            return team_name
    raise KeyError(f"Unable to resolve team code '{code}'. Pass --home/--away explicitly.")


def parse_lineup_text_to_scenario(
    raw_text: str,
    data_dir: str,
    home_team: str | None = None,
    away_team: str | None = None,
) -> dict[str, Any]:
    lines = _clean_lines(raw_text)
    formations = FORMATION_PATTERN.findall(raw_text)
    sections = _find_lineup_sections(lines)
    team_heading_sections = _find_team_heading_sections(lines)
    team_profiles = load_team_profiles(ProjectPaths(Path(data_dir)))
    if len(sections) >= 2:
        home_code, home_start = sections[0]
        away_code, away_start = sections[1]
        resolved_home = resolve_team_code(home_code, team_profiles, explicit_name=home_team)
        resolved_away = resolve_team_code(away_code, team_profiles, explicit_name=away_team)
        scenario = {
            "neutral_venue": False,
            "is_tournament": False,
            "home": {
                "team_name": resolved_home,
                "formation": formations[0] if len(formations) > 0 else "",
                **_parse_team_block(lines, home_start),
            },
            "away": {
                "team_name": resolved_away,
                "formation": formations[1] if len(formations) > 1 else "",
                **_parse_team_block(lines, away_start),
            },
        }
        parser_context = {
            "home_code": home_code,
            "away_code": away_code,
            "formations": formations[:2],
            "mode": "lineup_blocks",
        }
    else:
        inferred_home = home_team
        inferred_away = away_team
        if len(team_heading_sections) >= 2:
            inferred_home = inferred_home or team_heading_sections[0][0]
            inferred_away = inferred_away or team_heading_sections[1][0]
        resolved_home = resolve_team_code(str(home_team or ""), team_profiles, explicit_name=home_team) if home_team else None
        resolved_away = resolve_team_code(str(away_team or ""), team_profiles, explicit_name=away_team) if away_team else None
        if not resolved_home and inferred_home:
            resolved_home = resolve_team_code(str(inferred_home or ""), team_profiles, explicit_name=inferred_home)
        if not resolved_away and inferred_away:
            resolved_away = resolve_team_code(str(inferred_away or ""), team_profiles, explicit_name=inferred_away)
        if not resolved_home or not resolved_away:
            raise ValueError("Could not infer teams from squad-style text. Fill Home Team and Away Team.")
        scenario = _parse_squad_style_text(lines, resolved_home, resolved_away, formations)
        parser_context = {
            "home_code": resolved_home,
            "away_code": resolved_away,
            "formations": formations[:2],
            "mode": "squad_text",
        }
    _apply_global_coach_sentences(raw_text, scenario, resolved_home, resolved_away)
    return {
        "home_team": resolved_home,
        "away_team": resolved_away,
        "scenario": scenario,
        "parser_context": parser_context,
    }


def predict_from_lineup_text(
    raw_text: str,
    data_dir: str = "data",
    home_team: str | None = None,
    away_team: str | None = None,
    scenario_output_path: str | None = None,
    match_date: str | None = None,
    use_live_news: bool = False,
) -> dict[str, Any]:
    parsed = parse_lineup_text_to_scenario(raw_text, data_dir=data_dir, home_team=home_team, away_team=away_team)
    if scenario_output_path:
        Path(scenario_output_path).write_text(json.dumps(parsed["scenario"], indent=2), encoding="utf-8")
    payload = generate_prediction_payload(
        data_dir=data_dir,
        home=parsed["home_team"],
        away=parsed["away_team"],
        match_date=match_date,
        use_live_news=use_live_news,
        scenario_override=parsed["scenario"],
        neutral_venue=bool(parsed["scenario"].get("neutral_venue", False)),
        tournament_flag=bool(parsed["scenario"].get("is_tournament", False)),
    )
    payload["agent_context"] = {
        "parsed_home_team": parsed["home_team"],
        "parsed_away_team": parsed["away_team"],
        "parser_context": parsed["parser_context"],
        "scenario": parsed["scenario"],
        "scenario_output_path": scenario_output_path or "",
    }
    return payload
