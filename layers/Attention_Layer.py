import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale      # scaled
        self.mask_flag = mask_flag      # is mask or not
        self.output_attention = output_attention    # attention scores to visualize
        self.dropout = nn.Dropout(attention_dropout)    # dropout rate

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape  # [batch x query_len x n_head x n_embed]
        _, S, _, D = values.shape   # [batch x value_len x n_head x n_embed]
        scale = self.scale or 1. / sqrt(E)

        # compute attention scores  [batch x n_head x query_len x value_len]
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                # 三角因果掩码通常用于 decoder self attention
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        # scaled
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum('bhls,bshd->blhd', A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)    # wq
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)      # wk
        self.value_projection = nn.Linear(d_model, d_values * n_heads)  # wv
        self.out_projection = nn.Linear(d_values * n_heads, d_model)    # wo
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(queries, keys, values, attn_mask)

        out = out.view(B, L, -1)    # out.shape (B, L, D)

        return self.out_projection(out), attn