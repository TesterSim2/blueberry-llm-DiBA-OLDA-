import torch
import torch.nn as nn
import torch.nn.functional as F

from .diba_olda import Gated_DiBA_OLDA_Attention, precompute_freqs_cis


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = F.silu(x)
        x = self.dropout(x)
        return self.linear2(x)


class TransformerBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.d_model = args.dim
        self.n_heads = args.n_heads
        self.head_dim = getattr(args, "head_dim", self.d_model // self.n_heads)
        self.dropout = getattr(args, "dropout", 0.0)
        self.ff_mult = getattr(args, "ff_mult", 4)
        self.max_seq_len = getattr(args, "max_seq_len", 2048)
        self.rope_theta = getattr(args, "rope_theta", 10000.0)

        # Swap the engine to the DiBA-OLDA attention
        self.attention = Gated_DiBA_OLDA_Attention(args)

        # Standard transformer components stay intact
        hidden_dim = int(self.d_model * self.ff_mult)
        self.feed_forward = FeedForward(self.d_model, hidden_dim, dropout=self.dropout)
        self.attn_norm = nn.LayerNorm(self.d_model)
        self.ff_norm = nn.LayerNorm(self.d_model)
        self.dropout_layer = nn.Dropout(self.dropout)

        # Precompute RoPE cache for the custom attention
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(self.head_dim, self.max_seq_len, theta=self.rope_theta),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attention(self.attn_norm(x), self.freqs_cis)
        x = x + self.dropout_layer(attn_out)

        ff_out = self.feed_forward(self.ff_norm(x))
        x = x + self.dropout_layer(ff_out)
        return x
