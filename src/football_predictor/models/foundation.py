from __future__ import annotations

import torch
from torch import nn

from .gnn import AttentionPooling, SimpleGATLayer


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class TeamGraphEncoder(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.gat1 = SimpleGATLayer(node_dim, hidden_dim, heads=2, dropout=dropout)
        self.gat2 = SimpleGATLayer(hidden_dim * 2, hidden_dim, heads=2, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim * 2)
        self.norm2 = nn.LayerNorm(hidden_dim * 2)
        self.pool = AttentionPooling(hidden_dim * 2)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        h = self.gat1(x, edge_index, edge_weight)
        h = self.norm1(nn.functional.gelu(h))
        h = self.gat2(h, edge_index, edge_weight)
        h = self.norm2(nn.functional.gelu(h))
        pooled = self.pool(h, batch)
        return self.proj(pooled)


class MatchContextEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SquadAvailabilityEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.player_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.pool = AttentionPooling(hidden_dim)

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        projected = self.player_proj(x)
        return self.pool(projected, batch)


class NewsEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class KnowledgeEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ModalityProjector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ExpertMLP(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TaskMoEHead(nn.Module):
    def __init__(self, dim: int, output_dim: int, num_experts: int, dropout: float, positive: bool = False) -> None:
        super().__init__()
        self.experts = nn.ModuleList([ExpertMLP(dim, dropout) for _ in range(num_experts)])
        self.gate = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_experts),
        )
        self.out = nn.Linear(dim, output_dim)
        self.positive = positive

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        gate_logits = self.gate(x)
        gate_weights = torch.softmax(gate_logits, dim=-1)
        mixed = torch.sum(expert_outputs * gate_weights.unsqueeze(-1), dim=1)
        output = self.out(mixed)
        if self.positive:
            output = nn.functional.softplus(output)
        return output, gate_weights


class FootballFoundationModel(nn.Module):
    def __init__(
        self,
        node_dim: int,
        squad_dim: int,
        context_dim: int,
        news_dim: int,
        knowledge_dim: int,
        hidden_dim: int = 64,
        regression_dim: int = 8,
        dropout: float = 0.1,
        num_experts: int = 4,
        transformer_layers: int = 2,
        attention_heads: int = 4,
        knowledge_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        team_embed_dim = hidden_dim
        squad_embed_dim = hidden_dim // 2
        news_embed_dim = hidden_dim // 2
        knowledge_embed_dim = hidden_dim // 2

        self.team_encoder = TeamGraphEncoder(node_dim=node_dim, hidden_dim=hidden_dim // 2, dropout=dropout)
        self.squad_encoder = SquadAvailabilityEncoder(input_dim=squad_dim, hidden_dim=squad_embed_dim, dropout=dropout)
        self.context_encoder = MatchContextEncoder(input_dim=context_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.news_encoder = NewsEncoder(input_dim=news_dim, hidden_dim=news_embed_dim, dropout=dropout)
        self.knowledge_encoder = KnowledgeEncoder(input_dim=knowledge_dim, hidden_dim=knowledge_embed_dim, dropout=knowledge_dropout)

        self.home_team_token = ModalityProjector(team_embed_dim, hidden_dim, dropout)
        self.away_team_token = ModalityProjector(team_embed_dim, hidden_dim, dropout)
        self.home_squad_token = ModalityProjector(squad_embed_dim, hidden_dim, dropout)
        self.away_squad_token = ModalityProjector(squad_embed_dim, hidden_dim, dropout)
        self.context_token = ModalityProjector(hidden_dim, hidden_dim, dropout)
        self.home_news_token = ModalityProjector(news_embed_dim, hidden_dim, dropout)
        self.away_news_token = ModalityProjector(news_embed_dim, hidden_dim, dropout)
        self.home_knowledge_token = ModalityProjector(knowledge_embed_dim, hidden_dim, dropout)
        self.away_knowledge_token = ModalityProjector(knowledge_embed_dim, hidden_dim, dropout)
        self.interaction_token = ModalityProjector(
            (team_embed_dim * 4) + (squad_embed_dim * 3) + (news_embed_dim * 2) + (knowledge_embed_dim * 2),
            hidden_dim,
            dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=attention_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.modality_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.token_importance = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
        self.knowledge_gate = nn.Sequential(
            nn.LayerNorm(knowledge_embed_dim * 2),
            nn.Linear(knowledge_embed_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
            nn.Sigmoid(),
        )

        fusion_dim = hidden_dim * 3
        self.pre_fusion = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
        )
        self.fusion_residual = nn.Sequential(
            ResidualMLPBlock(hidden_dim, dropout),
            ResidualMLPBlock(hidden_dim, dropout),
            nn.LayerNorm(hidden_dim),
        )

        self.outcome_head = TaskMoEHead(hidden_dim, 3, num_experts=num_experts, dropout=dropout)
        self.regression_head = TaskMoEHead(hidden_dim, regression_dim, num_experts=num_experts, dropout=dropout)
        self.score_feature_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.score_head = TaskMoEHead(hidden_dim, 2, num_experts=num_experts, dropout=dropout, positive=True)
        self.outcome_refiner = nn.Sequential(
            nn.LayerNorm(hidden_dim + hidden_dim + 2),
            nn.Linear(hidden_dim + hidden_dim + 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.outcome_refine_head = nn.Linear(hidden_dim, 3)
        self.confidence_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.scorer_player_encoder = nn.Sequential(
            nn.Linear(squad_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.scorer_head = TaskMoEHead(hidden_dim + hidden_dim + hidden_dim // 2, 1, num_experts=num_experts, dropout=dropout)

    def _pool_tokens(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        token_scores = self.token_importance(tokens).squeeze(-1)
        token_weights = torch.softmax(token_scores, dim=-1)
        pooled = torch.sum(tokens * token_weights.unsqueeze(-1), dim=1)
        return pooled, token_weights

    def _score_player_rows(
        self,
        fused: torch.Tensor,
        team_embed: torch.Tensor,
        squad_x: torch.Tensor,
        squad_batch_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        repeated_match_context = fused[squad_batch_index]
        repeated_team_embed = team_embed[squad_batch_index]
        player_embed = self.scorer_player_encoder(squad_x)
        logits, gate_weights = self.scorer_head(
            torch.cat([repeated_match_context, repeated_team_embed, player_embed], dim=-1)
        )
        return logits.squeeze(-1), gate_weights

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        home_embed = self.team_encoder(
            batch["home_x"],
            batch["home_edge_index"],
            batch["home_edge_weight"],
            batch["home_batch_index"],
        )
        away_embed = self.team_encoder(
            batch["away_x"],
            batch["away_edge_index"],
            batch["away_edge_weight"],
            batch["away_batch_index"],
        )
        home_squad_embed = self.squad_encoder(batch["home_squad_x"], batch["home_squad_batch_index"])
        away_squad_embed = self.squad_encoder(batch["away_squad_x"], batch["away_squad_batch_index"])
        context_embed = self.context_encoder(batch["context"])
        home_news_embed = self.news_encoder(batch["home_news"])
        away_news_embed = self.news_encoder(batch["away_news"])
        home_knowledge_embed = self.knowledge_encoder(batch["home_knowledge"])
        away_knowledge_embed = self.knowledge_encoder(batch["away_knowledge"])
        knowledge_gates = self.knowledge_gate(torch.cat([home_knowledge_embed, away_knowledge_embed], dim=-1))
        home_knowledge_embed = home_knowledge_embed * knowledge_gates[:, :1]
        away_knowledge_embed = away_knowledge_embed * knowledge_gates[:, 1:]

        interaction = torch.cat(
            [
                home_embed - away_embed,
                torch.abs(home_embed - away_embed),
                home_embed * away_embed,
                home_embed + away_embed,
                home_squad_embed - away_squad_embed,
                torch.abs(home_squad_embed - away_squad_embed),
                home_squad_embed * away_squad_embed,
                home_news_embed - away_news_embed,
                torch.abs(home_news_embed - away_news_embed),
                home_knowledge_embed - away_knowledge_embed,
                torch.abs(home_knowledge_embed - away_knowledge_embed),
            ],
            dim=-1,
        )

        modality_tokens = torch.stack(
            [
                self.home_team_token(home_embed),
                self.away_team_token(away_embed),
                self.home_squad_token(home_squad_embed),
                self.away_squad_token(away_squad_embed),
                self.context_token(context_embed),
                self.home_news_token(home_news_embed),
                self.away_news_token(away_news_embed),
                self.home_knowledge_token(home_knowledge_embed),
                self.away_knowledge_token(away_knowledge_embed),
                self.interaction_token(interaction),
            ],
            dim=1,
        )
        encoded_tokens = self.modality_encoder(modality_tokens)
        pooled_tokens, token_weights = self._pool_tokens(encoded_tokens)
        fused = self.pre_fusion(
            torch.cat([pooled_tokens, encoded_tokens[:, 0], encoded_tokens[:, 1]], dim=-1)
        )
        fused = self.fusion_residual(fused)

        base_logits, outcome_gate = self.outcome_head(fused)
        regression, regression_gate = self.regression_head(fused)
        score_features = self.score_feature_head(fused)
        score_rates, score_gate = self.score_head(score_features)
        refined_outcome = self.outcome_refiner(torch.cat([fused, score_features, score_rates], dim=-1))
        logits = base_logits + self.outcome_refine_head(refined_outcome)
        confidence = self.confidence_head(fused).squeeze(-1)
        home_scorer_logits, home_scorer_gate = self._score_player_rows(
            fused,
            home_embed,
            batch["home_squad_x"],
            batch["home_squad_batch_index"],
        )
        away_scorer_logits, away_scorer_gate = self._score_player_rows(
            fused,
            away_embed,
            batch["away_squad_x"],
            batch["away_squad_batch_index"],
        )
        return {
            "logits": logits,
            "regression": regression,
            "score_rates": score_rates,
            "confidence": confidence,
            "home_scorer_logits": home_scorer_logits,
            "away_scorer_logits": away_scorer_logits,
            "token_weights": token_weights,
            "outcome_gate": outcome_gate,
            "regression_gate": regression_gate,
            "score_gate": score_gate,
            "knowledge_gate": knowledge_gates,
            "home_scorer_gate": home_scorer_gate,
            "away_scorer_gate": away_scorer_gate,
        }
