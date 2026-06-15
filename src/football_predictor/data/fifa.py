from __future__ import annotations

import json
from io import StringIO
import re
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup

from ..config import FIFA_RANKING_URL


class FifaRankingClient:
    def __init__(self, ranking_url: str = FIFA_RANKING_URL, timeout: int = 60) -> None:
        self.ranking_url = ranking_url
        self.timeout = timeout
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }

    def fetch_rankings(self) -> pd.DataFrame:
        metadata = self.fetch_metadata()
        date_id = metadata.get("legacy_date_id") or metadata.get("current_date_id")
        response = requests.get(
            "https://inside.fifa.com/api/ranking-overview",
            headers=self.headers,
            params={"locale": "en", "dateId": date_id, "rankingType": "football"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        rankings = payload.get("rankings", []) if isinstance(payload, dict) else []
        if rankings:
            rows = []
            for row in rankings:
                ranking_item = row.get("rankingItem", {})
                rows.append(
                    {
                        "rank": ranking_item.get("rank"),
                        "team": ranking_item.get("name"),
                        "points": ranking_item.get("totalPoints"),
                        "previous_points": row.get("previousPoints"),
                        "previous_rank": ranking_item.get("previousRank"),
                        "country_code": ranking_item.get("countryCode"),
                        "confederation": (row.get("tag") or {}).get("text"),
                        "last_update_date": row.get("lastUpdateDate"),
                        "next_update_date": row.get("nextUpdateDate"),
                    }
                )
            frame = pd.DataFrame(rows)
            frame["team"] = frame["team"].astype(str).str.strip()
            frame["rank"] = pd.to_numeric(frame["rank"], errors="coerce")
            frame["points"] = pd.to_numeric(frame["points"], errors="coerce")
            return frame.dropna(subset=["rank", "team", "points"]).reset_index(drop=True)

        response = requests.get(self.ranking_url, headers=self.headers, timeout=self.timeout)
        response.raise_for_status()
        html = response.text
        tables = pd.read_html(StringIO(html))
        ranking_table = None
        for table in tables:
            lower = [str(column).lower() for column in table.columns]
            if "team" in lower and "points" in lower:
                ranking_table = table.copy()
                break
        if ranking_table is None:
            ranking_table = self._parse_embedded_table(html)
        ranking_table.columns = [str(column).strip().lower() for column in ranking_table.columns]
        if "rank" not in ranking_table.columns:
            ranking_table.rename(columns={ranking_table.columns[0]: "rank"}, inplace=True)
        ranking_table["team"] = ranking_table["team"].astype(str).str.strip()
        ranking_table["rank"] = pd.to_numeric(ranking_table["rank"], errors="coerce")
        ranking_table["points"] = pd.to_numeric(ranking_table["points"], errors="coerce")
        return ranking_table[["rank", "team", "points"]].dropna().reset_index(drop=True)

    def fetch_metadata(self) -> dict[str, Any]:
        response = requests.get(self.ranking_url, headers=self.headers, timeout=self.timeout)
        response.raise_for_status()
        html = response.text
        match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', html)
        if not match:
            raise ValueError("Unable to find FIFA __NEXT_DATA__ payload.")
        payload = json.loads(match.group(1))
        ranking = payload["props"]["pageProps"]["pageData"]["ranking"]
        all_dates = ranking.get("allAvailableDates") or []
        current_date_id = all_dates[0]["id"] if all_dates else None
        legacy_date_id = self._infer_legacy_date_id(html)
        return {
            "last_update_date": ranking.get("lastUpdateDate"),
            "next_update_date": ranking.get("nextUpdateDate"),
            "current_date_id": current_date_id,
            "legacy_date_id": legacy_date_id,
        }

    def fetch_recent_results(self) -> list[dict[str, Any]]:
        metadata = self.fetch_metadata()
        current_date_id = metadata.get("current_date_id")
        if not current_date_id:
            return []
        match = re.search(r"(\d{4})(\d{2})(\d{2})$", current_date_id or "")
        if not match:
            return []
        response = requests.get(
            "https://inside.fifa.com/api/get-match-window-matches",
            headers=self.headers,
            params={
                "from": f"{match.group(1)}-{match.group(2)}-{match.group(3)}",
                "to": str(metadata.get("last_update_date", ""))[:10],
                "locale": "en",
                "gender": "men",
                "rankingType": "football",
            },
            timeout=self.timeout,
        )
        if response.status_code != 200:
            return []
        try:
            payload = response.json()
        except Exception:
            return []
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("matches", "payload", "data"):
                if isinstance(payload.get(key), list):
                    return payload[key]
        return []

    def _parse_embedded_table(self, html: str) -> pd.DataFrame:
        soup = BeautifulSoup(html, "lxml")
        text = soup.get_text("\n", strip=True)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        rows: list[dict[str, Any]] = []
        pattern = re.compile(r"^(\d+)\s+(.+?)\s+([0-9]{3,4}\.?[0-9]*)$")
        for line in lines:
            match = pattern.match(line)
            if match:
                rows.append(
                    {
                        "rank": int(match.group(1)),
                        "team": match.group(2).strip(),
                        "points": float(match.group(3)),
                    }
                )
        if not rows:
            raise ValueError("Unable to parse FIFA ranking table from page.")
        return pd.DataFrame(rows)

    def _infer_legacy_date_id(self, html: str) -> str | None:
        match = re.search(r"/fifa-world-ranking/men\?dateId=(id\d+)", html)
        return match.group(1) if match else None
