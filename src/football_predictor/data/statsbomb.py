from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests

from ..config import STATSBOMB_BASE


class StatsBombClient:
    def __init__(self, base_url: str = STATSBOMB_BASE, timeout: int = 60) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.local_root = Path("new_data/football_prediction_dataset/data/raw/statsbomb")
        self._matches_index: dict[tuple[int, int], Path] | None = None
        self._events_index: dict[int, Path] | None = None
        self._lineups_index: dict[int, Path] | None = None

    def _has_local_archive(self) -> bool:
        return self.local_root.exists()

    def _load_local_json(self, path: Path) -> Any:
        return json.loads(path.read_text(encoding="utf-8"))

    def _build_matches_index(self) -> dict[tuple[int, int], Path]:
        if self._matches_index is not None:
            return self._matches_index
        index: dict[tuple[int, int], Path] = {}
        matches_dir = self.local_root / "matches"
        for path in sorted(matches_dir.glob("*.json")):
            try:
                payload = self._load_local_json(path)
            except Exception:
                continue
            if not payload:
                continue
            first = payload[0]
            competition = first.get("competition") or {}
            season = first.get("season") or {}
            competition_id = competition.get("competition_id")
            season_id = season.get("season_id")
            if competition_id is None or season_id is None:
                continue
            index[(int(competition_id), int(season_id))] = path
        self._matches_index = index
        return index

    def _build_match_file_index(self, folder_name: str) -> dict[int, Path]:
        cache = self._events_index if folder_name == "events" else self._lineups_index
        if cache is not None:
            return cache
        index: dict[int, Path] = {}
        base = self.local_root / folder_name
        for path in sorted(base.rglob("*.json")):
            try:
                match_id = int(path.stem)
            except ValueError:
                continue
            index[match_id] = path
        if folder_name == "events":
            self._events_index = index
        else:
            self._lineups_index = index
        return index

    def _get_json(self, relative_path: str) -> Any:
        url = f"{self.base_url}/{relative_path.lstrip('/')}"
        response = requests.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def download_json(self, relative_path: str, target: Path) -> Any:
        payload = self._get_json(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def competitions(self) -> list[dict[str, Any]]:
        local_path = self.local_root / "competitions.json"
        if self._has_local_archive() and local_path.exists():
            competitions = self._load_local_json(local_path)
            available_pairs = set(self._build_matches_index().keys())
            return [
                row
                for row in competitions
                if (int(row.get("competition_id", -1)), int(row.get("season_id", -1))) in available_pairs
            ]
        return self._get_json("competitions.json")

    def matches(self, competition_id: int, season_id: int) -> list[dict[str, Any]]:
        if self._has_local_archive():
            local_path = self._build_matches_index().get((int(competition_id), int(season_id)))
            if local_path is not None:
                return self._load_local_json(local_path)
            return []
        return self._get_json(f"matches/{competition_id}/{season_id}.json")

    def events(self, match_id: int) -> list[dict[str, Any]]:
        if self._has_local_archive():
            local_path = self._build_match_file_index("events").get(int(match_id))
            if local_path is not None:
                return self._load_local_json(local_path)
            return []
        return self._get_json(f"events/{match_id}.json")

    def lineups(self, match_id: int) -> list[dict[str, Any]]:
        if self._has_local_archive():
            local_path = self._build_match_file_index("lineups").get(int(match_id))
            if local_path is not None:
                return self._load_local_json(local_path)
            return []
        return self._get_json(f"lineups/{match_id}.json")
