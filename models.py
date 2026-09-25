"""Parameter-matched Transformer and Mamba-2 models for training from scratch.

Adapted from the summer SLM notebooks. Mamba projections execute module hooks
for every optimizer, including the AdamW control.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

SIZES = {"150m": (768, 16, 30), "300m": (1024, 20, 38)}


def model_for(arch, size, sequence, tiny=False, *, vocab_size=50000, checkpointing=True):
    if arch not in ("transformer", "mamba") or size not in SIZES:
        raise ValueError("Use transformer/mamba and 150m/300m")
    width, transformer_layers, mamba_layers = (64, 1, 1) if tiny else SIZES[size]
    if arch == "transformer":
        return GPT(vocab_size, width, width // 64, transformer_layers, 4 * width,
                   sequence, 0.0, checkpointing)
    return Mamba2LM(vocab_size, width, mamba_layers, 16 if tiny else 128,
                    16 if tiny else 64, 2, 4, 16 if tiny else 256, checkpointing)

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim)   # QK-norm
        self.k_norm = nn.RMSNorm(self.head_dim)   # QK-norm
        self.dropout = dropout

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.proj(out.transpose(1, 2).contiguous().view(B, T, C))


class FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, use_checkpoint):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, d_ff)
        self.use_checkpoint = use_checkpoint

    def forward(self, x):
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)

    def _forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, context_length, dropout, use_checkpoint):
        super().__init__()
        self.context_length = context_length
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(context_length, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, d_ff, dropout, use_checkpoint)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, std=0.02 / (2 * n_layers) ** 0.5)
            nn.init.normal_(block.ffn.down.weight, std=0.02 / (2 * n_layers) ** 0.5)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        x = self.drop(self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device)))
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln_f(x))


class MambaBlock(nn.Module):
    """Pre-norm residual Mamba-2 block: x + Mamba2(RMSNorm(x)).

    There is no FFN: the mixer is the whole layer, which is why a parameter-matched Mamba-2 has about
    twice the transformer's layer count.
    """
    def __init__(self, d_model, layer_idx, mixer_kwargs, use_checkpoint):
        super().__init__()
        from mamba_ssm.modules.mamba2 import Mamba2
        self.norm = nn.RMSNorm(d_model)
        self.mixer = Mamba2(d_model, layer_idx=layer_idx, **mixer_kwargs)
        self.use_checkpoint = use_checkpoint

    def forward(self, x):
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)

    def _forward(self, x):
        return x + self.mixer(self.norm(x))


class Mamba2LM(nn.Module):
    """Mamba-2 LM (Dao & Gu 2024): embedding -> n_layers x MambaBlock -> RMSNorm -> tied LM head.

    No positional embedding: the causal conv and the selective state carry position, so
    context_length is a training block length rather than a hard limit.
    """
    def __init__(self, vocab_size, d_model, n_layers, d_state, headdim, expand, d_conv, chunk_size,
                 use_checkpoint):
        super().__init__()
        mixer_kwargs = dict(
            d_state=d_state, headdim=headdim, expand=expand, d_conv=d_conv, chunk_size=chunk_size,
            rmsnorm=True,             # gated RMSNorm before out_proj (Mamba-2 default), Triton kernel
            use_mem_eff_path=False,   # selected projections must execute module hooks
        )
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, i, mixer_kwargs, use_checkpoint) for i in range(n_layers)
        ])
        self.norm_f = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight

        # Init follows mamba_ssm's convention: embedding N(0, 0.02); Mamba2 initialises its own A_log,
        # dt_bias, D and conv; the residual-branch output projection keeps PyTorch's default init but is
        # scaled by 1/sqrt(n_layers) so the residual stream does not grow with depth.
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        with torch.no_grad():
            for block in self.blocks:
                block.mixer.out_proj.weight /= math.sqrt(n_layers)

    def forward(self, idx):
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm_f(x))
