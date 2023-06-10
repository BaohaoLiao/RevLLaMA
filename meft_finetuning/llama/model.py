# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Embedding, Linear, Dropout


@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048
    adapter_layer: int = 32
    adapter_dropout: float = 0.05
    adapter_dim: int = 8
    reversible_layer: int = 0 # No reversible layers by default. I.e. treat MEFT as a traditional PEFT
    x1_factor: float = 0.1
    x2_factor: float = 1
    sum_factor: float = 0


class Adapter(nn.Module):
    def __init__(self, dim: int, adapter_dim: int, dropout: float):
        super().__init__()
        self.dense1 = Linear(dim, adapter_dim)
        self.dense2 = Linear(adapter_dim, dim)
        self.dropout = Dropout(dropout)

    def forward(self, x):
        x = F.linear(x, self.dense1.weight.half(), bias=self.dense1.bias.half())
        x = F.silu(x)
        x = self.dropout(x)
        x = F.linear(x, self.dense2.weight.half(), bias=self.dense2.bias.half())
        return x


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_local_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wv = Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wo = Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)).cuda()
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)).cuda()

    def forward(
        self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]
    ):

        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        keys = xk
        values = xv

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = Linear(dim, hidden_dim, bias=False)
        self.w2 = Linear(hidden_dim, dim, bias=False)
        self.w3 = Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of)
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask, adapter)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class RevTransformerBlock(TransformerBlock):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__(layer_id, args)
        # for MEFT_1
        self.x1_factor = args.x1_factor
        self.x2_factor = args.x2_factor
        self.f_adapter = Adapter(dim=args.dim, adapter_dim=args.adapter_dim, dropout = args.adapter_dropout)
        self.rev_adapter = Adapter(dim=args.dim, adapter_dim=args.adapter_dim, dropout=args.adapter_dropout)

        self.F = self.forward_layer
        self.G = self.forward_adapter

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        x1, x2 = torch.chunk(x, 2, dim=-1)
        f_x2 = self.F(x2, start_pos, freqs_cis, mask)
        y1 = self.x1_factor * x1 + f_x2

        g_y1 = self.G(y1)
        y2 = self.x2_factor * x2 + g_y1

        out = torch.cat([y2, y1], dim=-1)
        return out

    def forward_layer(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        residual = h
        h = self.ffn_norm(h)
        h = self.feed_forward(h) + self.f_adapter(h)
        out = h + residual
        return out

    def forward_adapter(self, x: torch.Tensor):
        return self.rev_adapter(x)


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.tok_embeddings = Embedding(params.vocab_size, params.dim)

        self.adapter_layer = params.adapter_layer
        self.reversible_layer = params.reversible_layer
        #self.sum_factor = nn.Parameter(torch.tensor(params.sum_factor))
        self.sum_factor = params.sum_factor
        assert (self.adapter_layer <= self.n_layers) and (self.reversible_layer <= self.adapter_layer)

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

        self.layers = torch.nn.ModuleList()


        self.num_freeze_layer = self.n_layers - self.adapter_layer
        for layer_id in range(self.num_freeze_layer):
            self.layers.append(TransformerBlock(layer_id, params))
        for layer_id in range(self.num_freeze_layer, self.n_layers):
            self.layers.append(RevTransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = Linear(params.dim, params.vocab_size, bias=False)
        self.freqs_cis = precompute_freqs_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len * 2)

    def forward(self, examples, labels):

        _bsz, seqlen = examples.shape

        with torch.no_grad():
            h = self.tok_embeddings(examples)
            freqs_cis = self.freqs_cis.to(h.device)
            freqs_cis = freqs_cis[:seqlen]
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
            mask = torch.triu(mask, diagonal=0 + 1).type_as(h)
            start_pos = 0
            for layer in self.layers[: -1 * self.adapter_layer]:
                h = layer(h, start_pos, freqs_cis, mask)

        h = torch.cat([h, h], dim=-1)
        for layer in self.layers[-1 * self.adapter_layer :]:
            h = layer(h, start_pos, freqs_cis, mask)

        h1, h2 = torch.chunk(h, 2, dim=-1)
        h = self.sum_factor.half() * h1 + h2

        h = self.norm(h)
        output = self.output(h)
        output = output[:, :-1, :].reshape(-1, self.vocab_size)
        labels = labels[:, 1:].flatten()

        c_loss = self.criterion(output, labels)
        return c_loss
