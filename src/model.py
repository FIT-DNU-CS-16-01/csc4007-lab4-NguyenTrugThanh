from __future__ import annotations

import torch
from torch import nn


class SequenceClassifier(nn.Module):
    """Embedding + RNN/LSTM/GRU classifier."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_classes: int = 2,
        model_type: str = "lstm",
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        model_type = model_type.lower()
        if model_type not in {"rnn", "lstm", "gru"}:
            raise ValueError("model_type must be one of: rnn, lstm, gru")
        self.model_type = model_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        rnn_cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[model_type]
        self.encoder = rnn_cls(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * self.num_directions, num_classes)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        lengths_cpu = lengths.detach().cpu().clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        _, hidden = self.encoder(packed)
        if self.model_type == "lstm":
            hidden = hidden[0]
        if self.bidirectional:
            last_forward = hidden[-2]
            last_backward = hidden[-1]
            features = torch.cat([last_forward, last_backward], dim=1)
        else:
            features = hidden[-1]
        logits = self.classifier(self.dropout(features))
        return logits
