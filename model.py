"""
Pure PyTorch Causal Decoder-Only Transformer built from scratch.
Zero pre-trained weights — initialized with standard normal distributions.
Designed specifically for Voice Command -> Structured Task JSON auto-regressive generation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Llama/Mistral style)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class MultiHeadCausalAttention(nn.Module):
    """Multi-Head Attention with strict Causal Masking (prevents attending to future tokens)."""
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Linear projections for Query, Key, Value in one batched matrix
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Pre-compute causal triangular mask
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).unsqueeze(0).unsqueeze(0)
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # Batch, Sequence Length, Embedding Dim

        # Project and split into Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to (Batch, Heads, Seq_Len, Head_Dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention: (Q * K^T) / sqrt(d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply causal mask (set future positions to -infinity)
        scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))

        # Softmax & Dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values
        context = torch.matmul(attn_weights, v)  # (B, n_heads, T, head_dim)

        # Concatenate heads back to d_model
        context = context.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(context)


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network with GeLU activation."""
    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.linear2(self.gelu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Standard Decoder Transformer block with Pre-LayerNorm residuals."""
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadCausalAttention(d_model, n_heads, max_seq_len, dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn = FeedForward(d_model, hidden_dim=4 * d_model, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm residual connection for attention
        x = x + self.attn(self.norm1(x))
        # Pre-norm residual connection for feed-forward
        x = x + self.ffn(self.norm2(x))
        return x


class VoiceTaskSLM(nn.Module):
    """
    Complete Small Language Model (SLM) for Voice-to-Action parsing.
    Total Parameters: ~6.5 Million with default configs.
    """
    def __init__(
        self,
        vocab_size: int = 2048,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        max_seq_len: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        # Stacked Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, max_seq_len, dropout)
            for _ in range(n_layers)
        ])

        # Final Normalization and Output Projection Head
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying (tying token embeddings with output weights improves convergence)
        self.lm_head.weight = self.token_emb.weight

        # Weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        """
        Args:
            idx: (Batch, Seq_Len) token indices
            targets: (Batch, Seq_Len) target indices for training loss
        Returns:
            logits: (Batch, Seq_Len, Vocab_Size)
            loss: scalar cross-entropy loss if targets provided
        """
        B, T = idx.shape
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max {self.max_seq_len}"

        positions = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(positions)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Flatten to compute CrossEntropy
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-100  # Mask out pad tokens
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 120,
        temperature: float = 0.2,
        eos_id: int = 3,
    ) -> torch.Tensor:
        """
        Auto-regressive token generation with greedy / temperature sampling.
        """
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]
            logits, _ = self(idx_cond)
            # Pick logits of the last token
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            probs = F.softmax(logits, dim=-1)

            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)

            if next_token.item() == eos_id:
                break

        return idx
