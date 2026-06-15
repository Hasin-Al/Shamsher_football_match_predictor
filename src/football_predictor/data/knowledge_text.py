from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9'._-]+")


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def hash_texts(texts: list[str], dim: int) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float32)
    for text in texts:
        for token in TOKEN_PATTERN.findall(text.lower()):
            bucket = hash(token) % dim
            sign = 1.0 if (hash(f"{token}:sign") % 2 == 0) else -1.0
            vector[bucket] += sign
    norm = float(np.linalg.norm(vector))
    if norm > 0.0:
        vector /= norm
    return vector


def _project_vector(vector: np.ndarray, output_dim: int) -> np.ndarray:
    if vector.shape[0] == output_dim:
        return vector.astype(np.float32)
    generator = np.random.default_rng(2026 + vector.shape[0] + output_dim)
    projection = generator.standard_normal((vector.shape[0], output_dim), dtype=np.float32) / np.sqrt(max(output_dim, 1))
    projected = vector.astype(np.float32) @ projection
    norm = float(np.linalg.norm(projected))
    if norm > 0.0:
        projected /= norm
    return projected.astype(np.float32)


@dataclass(slots=True)
class KnowledgeTextConfig:
    model_path: str = ""
    fallback_dim: int = 128
    max_length: int = 256
    batch_size: int = 8

    @classmethod
    def from_file(cls, path: Path) -> "KnowledgeTextConfig":
        if not path.exists():
            return cls()
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            model_path=str(payload.get("model_path", "") or ""),
            fallback_dim=int(payload.get("fallback_dim", 128) or 128),
            max_length=int(payload.get("max_length", 256) or 256),
            batch_size=int(payload.get("batch_size", 8) or 8),
        )


class KnowledgeTextEncoder:
    def __init__(self, config: KnowledgeTextConfig) -> None:
        self.config = config
        self._mode = "hash"
        self._tokenizer = None
        self._model = None
        self.output_dim = config.fallback_dim

        model_path = config.model_path.strip()
        if not model_path:
            return

        try:
            from transformers import AutoModel, AutoTokenizer  # type: ignore
        except Exception:
            return

        model_dir = Path(model_path)
        if not model_dir.exists():
            return

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
            model = AutoModel.from_pretrained(model_dir, local_files_only=True)
        except Exception:
            return

        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False

        self._mode = "transformer"
        self._tokenizer = tokenizer
        self._model = model
        self.output_dim = int(getattr(model.config, "hidden_size", config.fallback_dim) or config.fallback_dim)

    @property
    def mode(self) -> str:
        return self._mode

    def encode_texts(self, texts: list[str]) -> list[np.ndarray]:
        cleaned = [normalize_text(text) for text in texts]
        if self._mode != "transformer" or self._tokenizer is None or self._model is None:
            return [hash_texts([text], self.output_dim) for text in cleaned]

        vectors: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(cleaned), self.config.batch_size):
                chunk = cleaned[start:start + self.config.batch_size]
                tokens = self._tokenizer(
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors="pt",
                )
                outputs = self._model(**tokens)
                hidden = outputs.last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
                pooled = torch.nn.functional.normalize(pooled, dim=-1)
                vectors.extend(pooled.cpu().numpy().astype(np.float32))
        return vectors


def player_priority_score(row: np.ndarray) -> float:
    start_prob = float(row[0]) if row.size > 0 else 0.0
    fitness = float(row[1]) if row.size > 1 else 0.0
    importance = float(row[5]) if row.size > 5 else 0.0
    confidence = float(row[-1]) if row.size > 0 else 0.0
    goals_per90 = float(row[-6]) if row.size >= 6 else 0.0
    xg90 = float(row[-4]) if row.size >= 4 else 0.0
    return (start_prob * 0.35) + (fitness * 0.15) + (importance * 0.2) + (goals_per90 * 0.15) + (xg90 * 0.1) + (confidence * 0.05)


def summarize_team_knowledge_text(
    *,
    team_name: str,
    team_knowledge: dict[str, Any],
    squad_player_names: list[str],
    squad_x: np.ndarray,
    form_points: float,
    goal_diff_form: float,
    manual_context: dict[str, Any] | None = None,
    top_k_players: int = 5,
) -> str:
    tactical = team_knowledge.get("tactical", {}) if isinstance(team_knowledge, dict) else {}
    manager_name = normalize_text((team_knowledge or {}).get("manager_name", ""))
    coach_match_count = float((team_knowledge or {}).get("coach_match_count", 0.0) or 0.0)

    lines = [
        f"Team {normalize_text(team_name)}.",
        f"Recent form points {form_points:.2f} and goal difference trend {goal_diff_form:.2f}.",
        (
            f"Coach {manager_name or 'unknown'} with {coach_match_count:.0f} tracked matches."
        ),
        (
            "Tactical profile "
            f"pass completion {float(tactical.get('pass_completion_rate', 0.0)):.3f}, "
            f"shots {float(tactical.get('shots', 0.0)):.2f}, "
            f"xg {float(tactical.get('xg', 0.0)):.2f}, "
            f"progressive passes {float(tactical.get('progressive_passes', 0.0)):.2f}, "
            f"final third entries {float(tactical.get('final_third_entries', 0.0)):.2f}, "
            f"direct speed {float(tactical.get('direct_speed_proxy', 0.0)):.2f}, "
            f"pressing intensity {float(tactical.get('pressing_intensity_proxy', 0.0)):.2f}."
        ),
    ]

    player_rows: list[tuple[float, str, np.ndarray]] = []
    for name, row in zip(squad_player_names, squad_x, strict=False):
        if normalize_text(name).lower() == "unknown":
            continue
        numeric_row = np.asarray(row, dtype=np.float32)
        player_rows.append((player_priority_score(numeric_row), normalize_text(name), numeric_row))
    player_rows.sort(key=lambda item: item[0], reverse=True)

    for _, name, row in player_rows[:top_k_players]:
        lines.append(
            (
                f"Player {name}: start_prob {float(row[0]):.2f}, fitness {float(row[1]):.2f}, "
                f"importance {float(row[5]):.2f}, club goals_per90 {float(row[-6]):.2f}, "
                f"assists_per90 {float(row[-5]):.2f}, xg90 {float(row[-4]):.2f}, "
                f"xa90 {float(row[-3]):.2f}, discipline {float(row[-2]):.2f}, prior_confidence {float(row[-1]):.2f}."
            )
        )

    manual_context = manual_context or {}
    coach_override = manual_context.get("coach", {})
    if coach_override:
        coach_name = normalize_text(coach_override.get("name", ""))
        known_match_count = float(coach_override.get("known_match_count", 0.0) or 0.0)
        if coach_name:
            lines.append(f"Manual coach input {coach_name} with {known_match_count:.0f} supplied matches.")
    tactical_override = manual_context.get("tactical_overrides", {})
    if tactical_override:
        lines.append(
            "Manual tactical override "
            f"pass completion {float(tactical_override.get('pass_completion_rate', 0.0)):.3f}, "
            f"shots {float(tactical_override.get('shots', 0.0)):.2f}, "
            f"xg {float(tactical_override.get('xg', 0.0)):.2f}, "
            f"progressive passes {float(tactical_override.get('progressive_passes', 0.0)):.2f}, "
            f"final third entries {float(tactical_override.get('final_third_entries', 0.0)):.2f}, "
            f"direct speed {float(tactical_override.get('direct_speed_proxy', 0.0)):.2f}, "
            f"pressing intensity {float(tactical_override.get('pressing_intensity_proxy', 0.0)):.2f}."
        )
    probable_xi = manual_context.get("probable_xi", [])
    injuries = manual_context.get("injuries", [])
    bench = manual_context.get("bench", [])
    if probable_xi:
        starters = [normalize_text(row.get("player", "")) for row in probable_xi[:11] if normalize_text(row.get("player", ""))]
        if starters:
            lines.append(f"Probable lineup core: {', '.join(starters)}.")
    if injuries:
        injury_bits = []
        for row in injuries[:8]:
            player = normalize_text(row.get("player", ""))
            status = normalize_text(row.get("status", "unknown"))
            if player:
                injury_bits.append(f"{player} {status}")
        if injury_bits:
            lines.append(f"Availability concerns: {', '.join(injury_bits)}.")
    if bench:
        bench_bits = [normalize_text(row.get("player", "")) for row in bench[:7] if normalize_text(row.get("player", ""))]
        if bench_bits:
            lines.append(f"Bench options: {', '.join(bench_bits)}.")

    return " ".join(lines)


def encode_summary_texts(texts: list[str], config: KnowledgeTextConfig) -> tuple[list[np.ndarray], dict[str, Any]]:
    encoder = KnowledgeTextEncoder(config)
    vectors = encoder.encode_texts(texts)
    return vectors, {
        "mode": encoder.mode,
        "vector_dim": int(encoder.output_dim),
        "model_path": config.model_path,
        "fallback_dim": int(config.fallback_dim),
    }
