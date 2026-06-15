from __future__ import annotations

import html
import json
import os
import socket
import sys
import warnings
from pathlib import Path
from typing import Any

import gradio as gr

warnings.filterwarnings("ignore", message=".*HTTP_422_UNPROCESSABLE_ENTITY.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*enable_nested_tensor.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Creating a tensor from a list of numpy.ndarrays.*", category=UserWarning)


APP_DIR = Path(__file__).resolve().parent
os.chdir(APP_DIR)
sys.path.insert(0, str(APP_DIR / "src"))

from football_predictor.agent import parse_lineup_text_to_scenario, predict_from_lineup_text
from football_predictor.goal_time import add_goal_time_predictions
from football_predictor.predict import generate_prediction_payload


DATA_DIR = APP_DIR / "data"

DEFAULT_LINEUP = """ARG
-
Line up
12
G. Rulli
6
L. Martinez
19
N. Otamendi
28
A. Giay
25
F. Medina
14
E. Palacios
8
V. Barco
17
G. Simeone
11
G. Lo Celso
18
N. Paz
21
J. Lopez

Substitutes
26
N. Molina
24
E. Fernandez
1
J. Musso
20
A. Mac Allister
4
G. Montiel
15
N. Gonzalez
7
R. De Paul
13
C. Romero
10
L. Messi
16
T. Almada
9
J. Alvarez

Injuries and Suspensions
No sidelined players

ISL
-
Line up
1
E. Olafsson
23
H. Magnusson
4
V. Palsson
3
D. Gretarsson
2
L. Tomasson
19
M. Ellertsson
8
I. Bergmann Johannesson
11
A. Gudmundsson
14
A. Baldursson
9
O. Oskarsson
7
H. Haraldsson

Substitutes
20
K. Hlynsson
16
K. Ingason
12
H. Valdimarsson
5
G. Thordarson
10
G. Sigurdsson
15
D. Thorhallsson
17
A. Gunnarsson
18
J. Thorsteinsson
21
A. Sigurdsson
13
A. Einarsson
6
H. Hermannsson
22
D. Gudjohnsen

Injuries and Suspensions
No sidelined players"""


PRO_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600;700&family=Fira+Sans:wght@400;500;600;700;800&display=swap');

:root {
  --page: #f8fafc;
  --panel: #ffffff;
  --text: #0f172a;
  --muted: #475569;
  --line: #dbe3ef;
  --line-strong: #cbd5e1;
  --blue: #1e40af;
  --blue-2: #2563eb;
  --blue-soft: #eff6ff;
  --amber: #f59e0b;
  --amber-soft: #fffbeb;
  --slate: #334155;
  --green: #047857;
  --red: #b91c1c;
  --mono: "Fira Code", "SFMono-Regular", Consolas, monospace;
  --sans: "Fira Sans", "Segoe UI", Arial, sans-serif;
}

html,
body,
.gradio-container {
  min-height: 100%;
  overflow-x: hidden !important;
  color: var(--text) !important;
  background:
    linear-gradient(180deg, rgba(30, 64, 175, 0.08), rgba(30, 64, 175, 0) 300px),
    var(--page) !important;
  font-family: var(--sans) !important;
  font-size: 16px !important;
}

.gradio-container {
  max-width: none !important;
}

.gradio-container * {
  box-sizing: border-box !important;
}

.pro-shell {
  width: min(1440px, calc(100vw - 40px));
  margin: 0 auto;
  padding: 22px 0 48px;
}

.topbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 16px;
  padding: 12px 0 18px;
}

.brand {
  display: flex;
  align-items: center;
  gap: 12px;
}

.brand-mark {
  display: grid;
  place-items: center;
  width: 44px;
  height: 44px;
  border-radius: 12px;
  background: var(--blue);
  color: #ffffff;
  box-shadow: 0 12px 26px rgba(30, 64, 175, 0.24);
}

.brand-mark svg {
  width: 24px;
  height: 24px;
}

.brand h1 {
  margin: 0;
  color: var(--text);
  font: 800 1.55rem/1 var(--sans);
  letter-spacing: 0;
}

.brand p {
  margin: 4px 0 0;
  color: var(--muted);
  font: 500 0.92rem/1.2 var(--sans);
}

.status-pill {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  padding: 10px 13px;
  border: 1px solid #bfdbfe;
  border-radius: 999px;
  background: var(--blue-soft);
  color: var(--blue);
  font: 700 0.82rem/1 var(--mono);
  text-transform: uppercase;
}

.status-pill i {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--amber);
}

.hero-strip {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 430px;
  gap: 22px;
  margin-bottom: 22px;
}

.hero-panel,
.bundle-panel,
.work-card {
  border: 1px solid var(--line);
  border-radius: 16px;
  background: var(--panel);
  box-shadow: 0 16px 42px rgba(15, 23, 42, 0.08);
}

.hero-panel {
  padding: 28px;
  display: grid;
  align-content: center;
  min-height: 180px;
}

.hero-panel h2 {
  max-width: 780px;
  margin: 0;
  color: var(--text);
  font: 800 clamp(2.1rem, 4vw, 4.2rem)/1 var(--sans);
  letter-spacing: -0.02em;
}

.hero-panel p {
  max-width: 720px;
  margin: 14px 0 0;
  color: var(--muted);
  font: 500 1.05rem/1.65 var(--sans);
}

.bundle-panel {
  padding: 20px;
}

.bundle-panel h3,
.section-title,
.micro-title {
  margin: 0;
  color: var(--slate);
  font: 700 0.78rem/1.2 var(--mono);
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.bundle-grid {
  display: grid;
  gap: 10px;
  margin-top: 14px;
}

.bundle-row {
  display: flex;
  justify-content: space-between;
  gap: 14px;
  padding: 11px 0;
  border-top: 1px solid var(--line);
}

.bundle-row span {
  color: var(--muted);
  font: 600 0.84rem/1.2 var(--sans);
}

.bundle-row strong {
  color: var(--text);
  font: 700 0.92rem/1.2 var(--sans);
  text-align: right;
}

.workspace {
  display: grid !important;
  grid-template-columns: minmax(560px, 0.9fr) minmax(620px, 1.1fr);
  gap: 24px;
  align-items: start !important;
}

.work-card {
  padding: 24px !important;
}

.card-head {
  display: flex;
  justify-content: space-between;
  gap: 18px;
  align-items: flex-start;
  padding-bottom: 16px;
  margin-bottom: 18px;
  border-bottom: 1px solid var(--line);
}

.card-head h2 {
  margin: 0;
  color: var(--text);
  font: 800 1.35rem/1.15 var(--sans);
}

.card-head p {
  margin: 6px 0 0;
  color: var(--muted);
  font: 500 0.95rem/1.45 var(--sans);
}

.step-block {
  margin-top: 22px;
}

.step-block:first-of-type {
  margin-top: 0;
}

.step-head {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: end;
  margin-bottom: 12px;
}

.step-head h3 {
  margin: 4px 0 0;
  color: var(--text);
  font: 800 1.02rem/1.2 var(--sans);
}

.step-head p {
  max-width: 32ch;
  margin: 0;
  color: var(--muted);
  font: 500 0.88rem/1.45 var(--sans);
  text-align: right;
}

.context-grid {
  display: grid !important;
  grid-template-columns: repeat(2, minmax(240px, 1fr));
  gap: 14px;
}

.option-grid {
  display: grid !important;
  grid-template-columns: minmax(260px, 1fr) minmax(190px, 230px);
  gap: 14px;
}

.context-grid > *,
.option-grid > *,
.gradio-container .block {
  min-width: 0 !important;
}

.gradio-container label,
.gradio-container .block-label {
  color: #1e293b !important;
  font: 700 0.86rem/1.35 var(--sans) !important;
  opacity: 1 !important;
}

.gradio-container input,
.gradio-container textarea,
.gradio-container select {
  color: var(--text) !important;
  background: #ffffff !important;
  border: 1px solid var(--line-strong) !important;
  border-radius: 10px !important;
  font: 500 1rem/1.45 var(--sans) !important;
  box-shadow: none !important;
}

.field-control input,
.field-control select {
  min-height: 50px !important;
  padding: 0 14px !important;
}

.gradio-container input::placeholder,
.gradio-container textarea::placeholder {
  color: #64748b !important;
  opacity: 1 !important;
}

.gradio-container input:focus,
.gradio-container textarea:focus,
.gradio-container select:focus {
  border-color: var(--blue-2) !important;
  box-shadow: 0 0 0 4px rgba(37, 99, 235, 0.14) !important;
  outline: none !important;
}

.lineup-field textarea {
  min-height: 440px !important;
  max-height: 620px !important;
  overflow: auto !important;
  padding: 16px !important;
  font: 500 0.9rem/1.65 var(--mono) !important;
}

.helper-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
  margin-top: 10px;
}

.helper-grid span {
  padding: 10px 12px;
  border: 1px solid var(--line);
  border-radius: 10px;
  background: #f8fafc;
  color: var(--muted);
  font: 500 0.88rem/1.4 var(--sans);
}

.actions {
  display: flex !important;
  flex-wrap: wrap;
  gap: 12px;
  padding: 14px;
  margin-top: 18px;
  border: 1px solid var(--line);
  border-radius: 12px;
  background: #f8fafc;
}

.gradio-container button {
  min-height: 46px !important;
  border-radius: 10px !important;
  cursor: pointer !important;
  font: 800 0.95rem/1 var(--sans) !important;
  letter-spacing: 0 !important;
  text-transform: none !important;
  transition: background-color 180ms ease, border-color 180ms ease, color 180ms ease, box-shadow 180ms ease !important;
}

.primary-button {
  min-width: 220px !important;
  color: #ffffff !important;
  background: var(--blue) !important;
  border: 1px solid var(--blue) !important;
  box-shadow: 0 12px 24px rgba(30, 64, 175, 0.22) !important;
}

.primary-button:hover {
  background: #1d4ed8 !important;
}

.secondary-button {
  color: #1e293b !important;
  background: #ffffff !important;
  border: 1px solid var(--line-strong) !important;
}

.secondary-button:hover {
  color: var(--blue) !important;
  border-color: var(--blue-2) !important;
}

.status-readout {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  min-height: 40px;
  padding: 9px 12px;
  border: 1px solid #bbf7d0;
  border-radius: 10px;
  background: #f0fdf4;
  color: var(--green);
  font: 800 0.82rem/1 var(--mono);
  text-transform: uppercase;
}

.status-readout i {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: currentColor;
}

.preview-box textarea {
  min-height: 170px !important;
  max-height: 280px !important;
  overflow: auto !important;
  color: #dbeafe !important;
  background: #0f172a !important;
  border: 0 !important;
  font: 500 0.82rem/1.55 var(--mono) !important;
}

.result-wrap {
  display: grid;
  gap: 16px;
}

.score-card {
  padding: 18px;
  border: 1px solid #bfdbfe;
  border-radius: 14px;
  background: var(--blue-soft);
}

.fixture-row {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: start;
  margin-bottom: 16px;
}

.fixture-row h3 {
  margin: 0;
  color: var(--text);
  font: 800 1.5rem/1.16 var(--sans);
}

.score-badge {
  flex: 0 0 auto;
  padding: 8px 10px;
  border-radius: 10px;
  background: var(--blue);
  color: #ffffff;
  font: 800 0.86rem/1 var(--mono);
  text-transform: uppercase;
}

.prob-list {
  display: grid;
  gap: 10px;
}

.prob-head {
  display: flex;
  justify-content: space-between;
  gap: 14px;
  color: #1e293b;
  font: 700 0.95rem/1.2 var(--sans);
}

.prob-head strong {
  font-family: var(--mono);
}

.track {
  height: 10px;
  border-radius: 999px;
  background: #dbeafe;
  overflow: hidden;
}

.track i {
  display: block;
  height: 100%;
  border-radius: inherit;
  background: var(--blue-2);
}

.prob-item.amber .track {
  background: #fde68a;
}

.prob-item.amber .track i {
  background: var(--amber);
}

.prob-item.slate .track {
  background: #e2e8f0;
}

.prob-item.slate .track i {
  background: var(--slate);
}

.notice {
  padding: 13px 14px;
  border: 1px solid #fde68a;
  border-radius: 12px;
  background: var(--amber-soft);
  color: #78350f;
  font: 600 0.94rem/1.5 var(--sans);
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(150px, 1fr));
  gap: 12px;
}

.metric {
  min-height: 86px;
  padding: 14px;
  border: 1px solid var(--line);
  border-left: 4px solid var(--blue-2);
  border-radius: 12px;
  background: #ffffff;
}

.metric span {
  display: block;
  margin-bottom: 8px;
  color: var(--muted);
  font: 700 0.72rem/1.25 var(--mono);
  letter-spacing: 0.05em;
  text-transform: uppercase;
}

.metric strong {
  display: block;
  color: var(--text);
  font: 800 1.45rem/1 var(--sans);
}

.data-section {
  padding-top: 14px;
  border-top: 1px solid var(--line);
}

.data-section h3 {
  margin: 0 0 10px;
  color: var(--slate);
  font: 800 0.8rem/1.2 var(--mono);
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.data-list {
  display: grid;
  gap: 2px;
}

.data-row {
  display: flex;
  justify-content: space-between;
  gap: 14px;
  padding: 8px 0;
  border-bottom: 1px solid #f1f5f9;
  color: #1e293b;
  font: 500 0.96rem/1.35 var(--sans);
}

.data-row span:first-child,
.data-row strong {
  color: #0f172a !important;
}

.data-row span:last-child {
  color: var(--muted);
  font-family: var(--mono);
  white-space: nowrap;
}

.json-box {
  max-height: 250px;
  overflow: auto;
  padding: 14px;
  border-radius: 12px;
  background: #0f172a;
  color: #dbeafe;
  font: 500 0.8rem/1.55 var(--mono);
  white-space: pre-wrap;
  word-break: break-word;
}

.empty-state,
.error-state {
  padding: 18px;
  border-radius: 14px;
  font: 600 0.98rem/1.5 var(--sans);
}

.empty-state {
  border: 1px dashed var(--line-strong);
  background: #f8fafc;
  color: var(--muted);
}

.error-state {
  border: 1px solid #fecaca;
  background: #fef2f2;
  color: #7f1d1d;
}

@media (prefers-reduced-motion: reduce) {
  *,
  *::before,
  *::after {
    transition-duration: 0.01ms !important;
    animation-duration: 0.01ms !important;
  }
}

@media (max-width: 1320px) {
  .hero-strip,
  .workspace {
    grid-template-columns: 1fr !important;
  }
}

@media (max-width: 760px) {
  .pro-shell {
    width: min(100vw - 24px, 1440px);
    padding-top: 16px;
  }
  .topbar,
  .card-head,
  .step-head,
  .fixture-row {
    display: block;
  }
  .status-pill {
    margin-top: 12px;
  }
  .hero-panel,
  .work-card {
    padding: 18px !important;
  }
  .context-grid,
  .option-grid,
  .helper-grid,
  .metrics-grid {
    grid-template-columns: 1fr !important;
  }
  .step-head p {
    max-width: none;
    margin-top: 6px;
    text-align: left;
  }
  .score-badge {
    display: inline-block;
    margin-top: 10px;
  }
  .primary-button,
  .secondary-button {
    width: 100% !important;
  }
}
"""


def _find_free_port(start_port: int = 8060, attempts: int = 200) -> int:
    for port in range(start_port, start_port + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _compact_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _normalize_match_date(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "date"):
        return value.date().isoformat()
    text = str(value).strip()
    if not text:
        return None
    return text.split("T", 1)[0].split(" ", 1)[0]


def _status(value: str, tone: str = "green") -> str:
    color = {
        "green": "#047857",
        "blue": "#1e40af",
        "red": "#b91c1c",
        "amber": "#92400e",
    }.get(tone, "#047857")
    return f'<div class="status-readout" style="color:{color}"><i></i><span>{html.escape(value)}</span></div>'


def _pct(value: Any) -> float:
    return max(0.0, min(100.0, _safe_float(value) * 100.0))


def _metric(label: str, value: Any) -> str:
    return f'<div class="metric"><span>{html.escape(label)}</span><strong>{html.escape(str(value))}</strong></div>'


def _row(label: Any, value: Any = "") -> str:
    return f'<div class="data-row"><span>{html.escape(str(label))}</span><span>{html.escape(str(value))}</span></div>'


def _prob(label: str, value: Any, tone: str) -> str:
    pct = _pct(value)
    return f"""
    <div class="prob-item {tone}">
      <div class="prob-head"><span>{html.escape(label)}</span><strong>{pct:.1f}%</strong></div>
      <div class="track"><i style="width:{pct:.1f}%"></i></div>
    </div>
    """


def _possession(profile: dict[str, Any]) -> tuple[float, float]:
    first = _safe_float(profile.get("home_possession_pct"), -1.0)
    second = _safe_float(profile.get("away_possession_pct"), -1.0)
    total = first + second
    if first < 0 or second < 0 or total <= 0:
        return 50.0, 50.0
    first_norm = first / total * 100.0
    return first_norm, 100.0 - first_norm


def _side_labels(venue_mode: str) -> tuple[str, str]:
    if venue_mode == "Team 1 Home":
        return "Home", "Away"
    if venue_mode == "Team 2 Home":
        return "Away", "Home"
    return "Team 1", "Team 2"


def _render_scorers(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return _row("No scorer candidates", "-")
    return "".join(_row(row.get("player", ""), f"{_pct(row.get('anytime_goal_probability')):.1f}%") for row in rows)


def _render_timeline(payload: dict[str, Any]) -> str:
    timing = payload.get("goal_time_predictions") or {}
    timeline = timing.get("scoreline_aligned_timeline") or []
    rows = "".join(
        _row(
            f"{item.get('minute_estimate')}' · {item.get('team')} · {item.get('player')}",
            item.get("score_after_goal", ""),
        )
        for item in timeline
    )
    return rows or _row("No goals in most likely scoreline", "0-0")


def _parse(lineup_text: str, team_1: str, team_2: str) -> dict[str, Any]:
    if not lineup_text.strip():
        raise gr.Error("Lineup source is required.")
    return parse_lineup_text_to_scenario(
        raw_text=lineup_text,
        data_dir=str(DATA_DIR),
        home_team=team_1.strip() or None,
        away_team=team_2.strip() or None,
    )


def _scenario_for_venue(parsed: dict[str, Any], venue_mode: str) -> tuple[dict[str, Any], str, str]:
    scenario = json.loads(json.dumps(parsed.get("scenario") or {}))
    parsed_home = str(parsed.get("home_team") or "")
    parsed_away = str(parsed.get("away_team") or "")
    if venue_mode == "Neutral":
        scenario["neutral_venue"] = True
        return scenario, parsed_home, parsed_away
    scenario["neutral_venue"] = False
    if venue_mode == "Team 1 Home":
        return scenario, parsed_home, parsed_away
    scenario = {
        **scenario,
        "home": json.loads(json.dumps(scenario.get("away") or {})),
        "away": json.loads(json.dumps(scenario.get("home") or {})),
    }
    return scenario, parsed_away, parsed_home


def parse_preview(lineup_text: str, team_1: str, team_2: str, venue_mode: str) -> tuple[str, str]:
    try:
        parsed = _parse(lineup_text, team_1, team_2)
        scenario, parsed_home, parsed_away = _scenario_for_venue(parsed, venue_mode)
        parsed = {**parsed, "home_team": parsed_home, "away_team": parsed_away, "scenario": scenario}
        return _compact_json(parsed), _status("Parsed", "blue")
    except Exception as exc:
        return f"Parse failed: {exc}", _status("Parse error", "red")


def _render_results(payload: dict[str, Any], venue_mode: str) -> str:
    probs = payload.get("win_draw_loss") or {}
    profile = payload.get("expected_match_profile") or {}
    scoreline = payload.get("probable_scoreline") or {}
    context = payload.get("agent_context") or {}
    first_label, second_label = _side_labels(venue_mode)
    first_poss, second_poss = _possession(profile)
    explanation = payload.get("prediction_explanation") or {}
    insight = str(explanation.get("summary") or "").strip()

    top_scorelines = "".join(
        _row(f"{row.get('home_goals')}-{row.get('away_goals')}", f"{_pct(row.get('probability')):.1f}%")
        for row in payload.get("top_scorelines", [])
    )
    venue_label = {
        "Neutral": "Neutral / unordered",
        "Team 1 Home": "Team 1 home",
        "Team 2 Home": "Team 2 home",
    }.get(venue_mode, venue_mode)

    return f"""
    <div class="result-wrap">
      <section class="score-card">
        <div class="fixture-row">
          <h3>{html.escape(str(payload.get('home_team', 'Team 1')))} vs {html.escape(str(payload.get('away_team', 'Team 2')))}</h3>
          <div class="score-badge">{int(scoreline.get('home_goals') or 0)}-{int(scoreline.get('away_goals') or 0)} most likely</div>
        </div>
        <div class="prob-list">
          {_prob(f"{first_label} Win", probs.get("home_win"), "blue")}
          {_prob("Draw", probs.get("draw"), "amber")}
          {_prob(f"{second_label} Win", probs.get("away_win"), "slate")}
        </div>
      </section>

      {f'<div class="notice">{html.escape(insight)}</div>' if insight else ''}

      <section class="data-section">
        <h3>Expected Match Profile</h3>
        <div class="metrics-grid">
          {_metric(f"{first_label} Shots on Target", round(_safe_float(profile.get("home_shots_on_target"))))}
          {_metric(f"{second_label} Shots on Target", round(_safe_float(profile.get("away_shots_on_target"))))}
          {_metric(f"{first_label} Passes", round(_safe_float(profile.get("home_passes"))))}
          {_metric(f"{second_label} Passes", round(_safe_float(profile.get("away_passes"))))}
          {_metric(f"{first_label} Possession", f"{first_poss:.1f}%")}
          {_metric(f"{second_label} Possession", f"{second_poss:.1f}%")}
          {_metric(f"{first_label} xG", f"{_safe_float(profile.get("home_xg")):.2f}")}
          {_metric(f"{second_label} xG", f"{_safe_float(profile.get("away_xg")):.2f}")}
          
        </div>
      </section>

      <section class="data-section">
        <h3>Probable Scorers</h3>
        <div class="data-list">
          {_row(payload.get("home_team", "Team 1"), "")}
          {_render_scorers((payload.get("probable_scorers") or {}).get("home") or [])}
          {_row(payload.get("away_team", "Team 2"), "")}
          {_render_scorers((payload.get("probable_scorers") or {}).get("away") or [])}
        </div>
      </section>

      <section class="data-section">
        <h3>Scoreline Timeline</h3>
        <div class="data-list">{_render_timeline(payload)}</div>
      </section>

      <section class="data-section">
        <h3>Top Scorelines</h3>
        <div class="data-list">{top_scorelines or _row("No scoreline distribution", "-")}</div>
      </section>

      <section class="data-section">
        <h3>Agent Parse</h3>
        <div class="data-list">
          {_row(f"Detected {first_label}", context.get("parsed_home_team") or payload.get("home_team", ""))}
          {_row(f"Detected {second_label}", context.get("parsed_away_team") or payload.get("away_team", ""))}
          {_row("Venue Model", venue_label)}
          {_row("Backend Path", context.get("prediction_backend") or "lineup-text")}
          {_row("Heuristic Fallback", "Yes" if (payload.get("profile_generation") or {}).get("used_heuristic_match_profile") else "No")}
        </div>
      </section>

      <pre class="json-box">{html.escape(_compact_json(payload))}</pre>
    </div>
    """


def _empty_results() -> str:
    return '<div class="empty-state">No prediction yet. Fill the match context, paste lineup text, then run the model.</div>'


def _error_results(message: str) -> str:
    return f'<div class="error-state"><strong>Prediction did not complete.</strong><br>{html.escape(message)}</div>'


def run_prediction(
    team_1: str,
    team_2: str,
    match_date: Any,
    scenario_path: str,
    lineup_text: str,
    venue_mode: str,
    use_live_news: bool,
) -> tuple[str, str, str]:
    try:
        normalized_date = _normalize_match_date(match_date)
        if venue_mode == "Neutral":
            payload = predict_from_lineup_text(
                raw_text=lineup_text,
                data_dir=str(DATA_DIR),
                home_team=team_1.strip() or None,
                away_team=team_2.strip() or None,
                scenario_output_path=scenario_path.strip() or None,
                match_date=normalized_date,
                use_live_news=bool(use_live_news),
            )
            payload = add_goal_time_predictions(payload)
            payload.setdefault("agent_context", {})["prediction_backend"] = "predict_from_lineup_text"
            parsed_payload = payload.get("agent_context") or {}
        else:
            parsed = _parse(lineup_text, team_1, team_2)
            scenario, parsed_home, parsed_away = _scenario_for_venue(parsed, venue_mode)
            scenario.setdefault("home", {})["team_name"] = parsed_home
            scenario.setdefault("away", {})["team_name"] = parsed_away
            if scenario_path.strip():
                Path(scenario_path.strip()).write_text(_compact_json(scenario), encoding="utf-8")
            payload = generate_prediction_payload(
                data_dir=str(DATA_DIR),
                home=parsed_home,
                away=parsed_away,
                match_date=normalized_date,
                use_live_news=bool(use_live_news),
                scenario_override=scenario,
                neutral_venue=bool(scenario.get("neutral_venue", False)),
                tournament_flag=bool(scenario.get("is_tournament", False)),
            )
            payload = add_goal_time_predictions(payload)
            payload["agent_context"] = {
                "parsed_home_team": parsed_home,
                "parsed_away_team": parsed_away,
                "parser_context": parsed.get("parser_context") or {},
                "scenario": scenario,
                "scenario_output_path": scenario_path.strip(),
                "prediction_backend": "scenario_override",
            }
            parsed_payload = {**parsed, "home_team": parsed_home, "away_team": parsed_away, "scenario": scenario}
        return _render_results(payload, venue_mode), _compact_json(parsed_payload), _status("Complete", "green")
    except Exception as exc:
        return _error_results(str(exc)), f"Prediction failed: {exc}", _status("Error", "red")


def load_example() -> tuple[str, str, Any, str, str]:
    return "", "", None, "", DEFAULT_LINEUP


def clear_form() -> tuple[str, str, Any, str, str, str, str]:
    return "", "", None, "", "", _empty_results(), _status("Idle", "amber")


with gr.Blocks(title="Match Oracle Pro Dashboard") as demo:
    with gr.Column(elem_classes=["pro-shell"]):
        gr.HTML(
            """
            <header class="topbar">
              <div class="brand">
                <div class="brand-mark" aria-hidden="true">
                  <svg viewBox="0 0 24 24" fill="none">
                    <path d="M4 18V8.5L12 4l8 4.5V18l-8 4-8-4Z" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/>
                    <path d="M8 12h8M12 8v8" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
                  </svg>
                </div>
                <div>
                  <h1>Match Oracle</h1>
                  <p>Standalone football prediction dashboard</p>
                </div>
              </div>
              <div class="status-pill"><i></i> Local inference bundle</div>
            </header>

            <section class="hero-strip">
              <div class="hero-panel">
                <h2>Convert lineup text into a complete match forecast.</h2>
                <p>Paste raw team news, confirm parser output, and run the copied foundation checkpoint with the same prediction pipeline.</p>
              </div>
              <aside class="bundle-panel">
                <h3>Workspace Summary</h3>
                <div class="bundle-grid">
                  <div class="bundle-row"><span>Input</span><strong>Team hints, date, lineup source</strong></div>
                  <div class="bundle-row"><span>Model</span><strong>Foundation checkpoint</strong></div>
                  <div class="bundle-row"><span>Output</span><strong>Result, profile, scorers</strong></div>
                </div>
              </aside>
            </section>
            """
        )

        with gr.Row(elem_classes=["workspace"]):
            with gr.Column(elem_classes=["work-card"]):
                gr.HTML(
                    """
                    <div class="card-head">
                      <div>
                        <h2>Prediction Setup</h2>
                        <p>Use text team hints, a date picker, and the raw lineup source.</p>
                      </div>
                    </div>
                    """
                )
                with gr.Column(elem_classes=["step-block"]):
                    gr.HTML(
                        """
                        <div class="step-head">
                          <div>
                            <div class="micro-title">Step 1</div>
                            <h3>Match Context</h3>
                          </div>
                          <p>Hints are optional, but help the parser resolve teams cleanly.</p>
                        </div>
                        """
                    )
                    with gr.Row(elem_classes=["context-grid"]):
                        team_1 = gr.Textbox(label="Team 1", placeholder="Argentina", elem_classes=["field-control"])
                        team_2 = gr.Textbox(label="Team 2", placeholder="Iceland", elem_classes=["field-control"])
                    with gr.Row(elem_classes=["context-grid"]):
                        match_date = gr.DateTime(
                            label="Match date",
                            include_time=False,
                            type="string",
                            elem_classes=["field-control"],
                        )
                        scenario_path = gr.Textbox(
                            label="Scenario save path",
                            placeholder="Optional JSON output path",
                            elem_classes=["field-control"],
                        )

                with gr.Column(elem_classes=["step-block"]):
                    gr.HTML(
                        """
                        <div class="step-head">
                          <div>
                            <div class="micro-title">Step 2</div>
                            <h3>Lineup Source</h3>
                          </div>
                          <p>Paste copied match text with starters, bench, injuries, formation, and coach when available.</p>
                        </div>
                        """
                    )
                    lineup_text = gr.Textbox(
                        label="Lineup block",
                        value=DEFAULT_LINEUP,
                        lines=22,
                        max_lines=34,
                        placeholder="Paste the full lineup block here.",
                        elem_classes=["lineup-field"],
                    )
                    gr.HTML(
                        """
                        <div class="helper-grid">
                          <span>Accepted: team-code blocks, named squads, substitutes, injuries, formations.</span>
                          <span>Parser preview shows the scenario before prediction.</span>
                        </div>
                        """
                    )

                with gr.Column(elem_classes=["step-block"]):
                    gr.HTML(
                        """
                        <div class="step-head">
                          <div>
                            <div class="micro-title">Step 3</div>
                            <h3>Run Options</h3>
                          </div>
                        </div>
                        """
                    )
                    with gr.Row(elem_classes=["option-grid"]):
                        venue_mode = gr.Dropdown(
                            label="Venue context",
                            choices=["Neutral", "Team 1 Home", "Team 2 Home"],
                            value="Neutral",
                            elem_classes=["field-control"],
                        )
                        use_live_news = gr.Checkbox(label="Include live news", value=False)

                    with gr.Row(elem_classes=["actions"]):
                        predict_button = gr.Button("Run prediction", elem_classes=["primary-button"])
                        parse_button = gr.Button("Parse preview", elem_classes=["secondary-button"])
                        load_button = gr.Button("Load example", elem_classes=["secondary-button"])
                        clear_button = gr.Button("Clear", elem_classes=["secondary-button"])

                with gr.Column(elem_classes=["step-block"]):
                    gr.HTML('<div class="micro-title">Parsed Scenario Preview</div>')
                    scenario_preview = gr.Textbox(
                        label="",
                        value="Parser output appears here.",
                        lines=8,
                        max_lines=12,
                        interactive=False,
                        elem_classes=["preview-box"],
                    )

            with gr.Column(elem_classes=["work-card"]):
                gr.HTML(
                    """
                    <div class="card-head">
                      <div>
                        <h2>Prediction Board</h2>
                        <p>Forecast results update here after inference completes.</p>
                      </div>
                    </div>
                    """
                )
                status = gr.HTML(value=_status("Idle", "amber"))
                results = gr.HTML(value=_empty_results())

    predict_button.click(
        run_prediction,
        inputs=[team_1, team_2, match_date, scenario_path, lineup_text, venue_mode, use_live_news],
        outputs=[results, scenario_preview, status],
        show_progress="full",
    )
    parse_button.click(
        parse_preview,
        inputs=[lineup_text, team_1, team_2, venue_mode],
        outputs=[scenario_preview, status],
        show_progress="minimal",
    )
    load_button.click(
        load_example,
        inputs=[],
        outputs=[team_1, team_2, match_date, scenario_path, lineup_text],
    )
    clear_button.click(
        clear_form,
        inputs=[],
        outputs=[team_1, team_2, match_date, scenario_path, lineup_text, results, status],
    )


if __name__ == "__main__":
    host = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    port = int(os.environ["GRADIO_SERVER_PORT"]) if "GRADIO_SERVER_PORT" in os.environ else _find_free_port()
    demo.launch(server_name=host, server_port=port, css=PRO_CSS)
