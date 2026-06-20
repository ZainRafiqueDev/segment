# backend/lstm_model.py
# LSTM Model Definition — must match exactly what was trained in Colab
# Week 7-9: RNNs → LSTM
# EXPLAIN: "BiLSTM reads the review token by token, forward and backward.
#           At each step the forget gate decides what to discard,
#           the input gate decides what to remember, and the output gate
#           controls what the next layer sees. The final hidden state is
#           the compressed 512-dim representation of the entire review."

import torch
import torch.nn as nn


class SentimentLSTM(nn.Module):
    """
    Bidirectional LSTM for binary sentiment classification.

    Architecture:
        Embedding(vocab_size, 128)
        → BiLSTM(128→256, 2 layers, dropout=0.3)
        → Concat [forward_hidden, backward_hidden]  → 512-dim latent vector
        → Dropout(0.3)
        → Linear(512, 128) + ReLU          (Week 4)
        → Dropout(0.3)
        → Linear(128, 1) + Sigmoid          (Week 4)

    Key concepts by week:
        Week 4:  ReLU in FC layer, Sigmoid output activation
        Week 7:  RNN foundation — sequence processing
        Week 8:  BPTT — gradients flow backward through LSTM gates
        Week 9:  LSTM gates — forget, input, cell state, output
        Week 10: Latent space — 512-dim hidden state as compressed review repr
        Week 12: Latent vector used in cosine similarity RAG search
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim:  int = 128,
        hidden_dim: int = 256,
        n_layers:   int = 2,
        dropout:    float = 0.3,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc1     = nn.Linear(hidden_dim * 2, 128)   # 512 → 128
        self.fc2     = nn.Linear(128, 1)
        self.relu    = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, return_hidden: bool = False):
        """
        Args:
            x:             (batch, seq_len) — token IDs
            return_hidden: if True, also return the 512-dim latent vector

        Returns:
            prob:   (batch,) — probability of positive sentiment
            latent: (batch, 512) — only if return_hidden=True
        """
        embedded = self.dropout(self.embedding(x))           # (B, L, 128)
        lstm_out, (hidden, _) = self.lstm(embedded)          # hidden: (4, B, 256)

        # Concatenate last layer's forward and backward hidden states → latent space
        hidden_fwd = hidden[-2]                              # (B, 256)
        hidden_bwd = hidden[-1]                              # (B, 256)
        latent     = torch.cat([hidden_fwd, hidden_bwd], dim=1)  # (B, 512)

        out   = self.dropout(latent)
        out   = self.relu(self.fc1(out))
        out   = self.dropout(out)
        prob  = self.sigmoid(self.fc2(out)).squeeze(1)       # (B,)

        if return_hidden:
            return prob, latent
        return prob
