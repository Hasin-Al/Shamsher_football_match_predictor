# Football Goal Time Gradio App

This folder is a self-contained export of the football prediction app. It includes the copied foundation checkpoint, team profile data, compact player-prior data, the required `football_predictor` source package, and a Gradio UI.

## Run

```bash
cd gradio_goal_time_app
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

If your system only exposes Python 3 as `python3`, use `python3 -m venv venv` and `python3 app.py`.

The app opens a local Gradio server. It does not need the original root project folder.

## Included Artifacts

- `data/models/football_foundation.pt`: copied current model checkpoint.
- `data/processed/team_profiles.json`: copied team profiles used for squad and team features.
- `data/external/fbref/` and `data/external/understat/`: compact player-prior CSVs for better handling of lineup players.
- `data/manual/team_context.example.json`: example manual context format.
- `config/knowledge_config.json`: copied knowledge encoder config.
- `src/football_predictor/`: copied inference, parser, feature, and model code.

## Model Architecture

Checkpoint: `data/models/football_foundation.pt`

Architecture class: `FootballFoundationModel`

Trainable parameters in the saved model state: `3,304,557`

Core dimensions from the copied checkpoint:

- Player graph node features: `20`
- Squad row features: `15`
- Match context features: `52`
- News vector dimension: `128`
- Knowledge vector dimension: `128`
- Hidden dimension: `128`
- Transformer layers: `1`
- Attention heads: `4`
- Mixture-of-experts heads: `4` experts
- Dropout: `0.3`
- Knowledge dropout: `0.1`

High-level flow:

1. Each team squad is converted into a player graph and encoded by two lightweight graph-attention layers.
2. Home and away squad availability matrices are encoded with an attention-pooling squad encoder.
3. Match context, hashed news vectors, and team knowledge text vectors are encoded as separate modalities.
4. Ten modality tokens are built: home team, away team, home squad, away squad, match context, home news, away news, home knowledge, away knowledge, and an interaction token.
5. A Transformer encoder fuses these modality tokens.
6. Task heads predict win/draw/loss logits, match regression metrics, goal rates, scorer logits, and confidence.
7. The inference layer blends model goal rates with xG/scoreline consistency heuristics.
8. A scoreline-aligned timeline is derived from expected goals, possession, and scorer probabilities for display.

Training configuration stored in the checkpoint:

- Epochs: `90`
- Batch size: `32`
- Learning rate: `0.0002`
- Weight decay: `0.0005`
- Random seed: `42`
- Outcome loss weight: `1.4`
- Regression loss weight: `0.3`
- Score loss weight: `0.25`
- Scorer loss weight: `0.15`
- Confidence loss weight: `0.05`
- Label smoothing: `0.1`
- Focal gamma: `1.5`

## Prediction Inputs

The Gradio app supports two input styles.

### 1. Team Match Input

Use this when you only want to select teams:

- `Home Team`: team name present in `data/processed/team_profiles.json`, for example `Argentina`.
- `Away Team`: team name present in `data/processed/team_profiles.json`, for example `France`.
- `Match Date`: optional ISO date, `YYYY-MM-DD`.
- `Neutral venue`: boolean.
- `Tournament match`: boolean.
- `Use live/news context`: optional; leave off for fully local deterministic-style runs.

The app resolves common aliases such as `USA` / `United States`, `South Korea` / `Korea Republic`, and normalized team names.

### 2. Lineup Text Input

Use this when you want lineup, formation, coach, bench, and injury context to affect the prediction.

Recommended format:

```text
Argentina
Coach: Lionel Scaloni
Formation: 4-3-3
GK: Emiliano Martinez
Defenders: Nahuel Molina, Cristian Romero, Nicolas Otamendi, Nicolas Tagliafico
Midfielders: Rodrigo De Paul, Enzo Fernandez, Alexis Mac Allister
Forwards: Lionel Messi, Lautaro Martinez, Julian Alvarez
Substitutes: Angel Di Maria, Leandro Paredes, Paulo Dybala
Injuries: Player Name

France
Coach: Didier Deschamps
Formation: 4-2-3-1
GK: Hugo Lloris
Defenders: Jules Kounde, Raphael Varane, Dayot Upamecano, Theo Hernandez
Midfielders: Aurelien Tchouameni, Adrien Rabiot, Antoine Griezmann
Forwards: Kylian Mbappe, Olivier Giroud, Ousmane Dembele
Substitutes: Kingsley Coman, Marcus Thuram, Eduardo Camavinga
```

Accepted lineup details:

- Team heading on its own line.
- `Coach: Name`
- `Formation: 4-3-3`, `4-2-3-1`, `3-5-2`, etc.
- Position lines: `GK:`, `RB:`, `CB:`, `LB:`, `DM:`, `CM:`, `AM:`, `LW:`, `RW:`, `ST:`, `FW:`
- Group lines: `Goalkeepers:`, `Defenders:`, `Midfielders:`, `Forwards:`
- `Substitutes:` or `Bench:` followed by comma-separated names.
- `Injuries:` followed by comma-separated names.

You can also paste lineup blocks in the simpler code-style format used by the original parser:

```text
ARG
-
Line up
23
Emiliano Martinez
4
Cristian Romero
...
Substitutes
11
Angel Di Maria

FRA
-
Line up
10
Kylian Mbappe
...
```

If the parser cannot infer the teams, fill `Home Team override` and `Away Team override`.

## Outputs

The app returns:

- Win/draw/loss probabilities.
- Expected goals, xG, shots on target, and other match profile metrics.
- Most likely scoreline and top scorelines.
- Probable scorers per team.
- Scoreline-aligned likely goal timeline.
- Raw JSON payload for downstream use.

## Notes

- The foundation model is learned from the project training data; the displayed scoreline timeline is derived during inference.
- New player names that are not in the saved team profile can still be injected from lineup text. If no prior stats are found, the app uses zero/default priors for those players.
- For reproducibility, run from this folder so local relative paths resolve to the copied `data/` and `config/` folders. `app.py` also changes the working directory to its own folder at startup.
