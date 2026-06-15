from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


STATSBOMB_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
FIFA_RANKING_URL = "https://inside.fifa.com/fifa-world-ranking/men"
THESPORTSDB_BASE = "https://www.thesportsdb.com/api/v1/json/123"


@dataclass(slots=True)
class ProjectPaths:
    root: Path

    @property
    def raw_dir(self) -> Path:
        return self.root / "raw"

    @property
    def interim_dir(self) -> Path:
        return self.root / "interim"

    @property
    def processed_dir(self) -> Path:
        return self.root / "processed"

    @property
    def external_dir(self) -> Path:
        return self.root / "external"

    @property
    def manual_dir(self) -> Path:
        return self.root / "manual"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    def ensure(self) -> None:
        for path in [self.raw_dir, self.interim_dir, self.processed_dir, self.external_dir, self.manual_dir, self.models_dir]:
            path.mkdir(parents=True, exist_ok=True)
