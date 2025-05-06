import torch
import torch.nn as nn
from math import sqrt
import torch.nn.functional as F

# Bidirectional Multimodel Attention

class BMAttention(nn.Module):
    def __init__(self, scale=None, attention_dropout=0.1, output_attention=False):
        super(BMAttention, self).__init__()
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def compute_attention(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale if self.scale is not None else 1. / sqrt(E)
        scores = torch.einsum('blhe,bshe->bhls', queries, keys)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous().view(B, L, -1), A
        else:
            return V.contiguous().view(B, L, -1), None

    # w: water, m: mete
    def forward(self, queries_w, keys_w, values_w, queries_m, keys_m, values_m, attn_mask):
        # queries: [B x L x H x E]
        contexts = []
        attns = []

        # water self attention, q: water, k: water, v: water
        context, attn = self.compute_attention(queries_w, keys_w, values_w)
        contexts.append(context)
        attns.append(attn)

        # mete self attention, q: mete, k: mete, v: mete
        context, attn = self.compute_attention(queries_m, keys_m, values_m)
        contexts.append(context)
        attns.append(attn)

        # cross attention wm, q: water, k: mete, v mete
        context, attn = self.compute_attention(queries_w, keys_m, values_m)
        contexts.append(context)
        attns.append(attn)

        # cross attention mw, q: mete, k: water, v: water
        context, attn = self.compute_attention(queries_m, keys_w, values_w)
        contexts.append(context)
        attns.append(attn)

        return contexts, attns

class BMAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None, dropout_rate=0.1):
        super(BMAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.n_heads = n_heads
        self.inner_attention = attention

        # water qkv 矩阵
        self.query_w = nn.Linear(d_model, d_keys * n_heads)
        self.key_w = nn.Linear(d_model, d_keys * n_heads)
        self.value_w = nn.Linear(d_model, d_values * n_heads)

        # mete qkv 矩阵
        self.query_m = nn.Linear(d_model, d_keys * n_heads)
        self.key_m = nn.Linear(d_model, d_keys * n_heads)
        self.value_m = nn.Linear(d_model, d_values * n_heads)

        # self.out = nn.Linear(d_model, d_model)
        self.projection_water = nn.Linear(d_model, d_model)
        self.projection_mete = nn.Linear(d_model, d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, queries_w, keys_w, values_w, queries_m, keys_m, values_m, attn_mask=None):
        B, w_L, _ = queries_w.shape
        _, w_S, _ = keys_w.shape

        _, m_L, _ = queries_m.shape
        _, m_S, _ = keys_m.shape
        H = self.n_heads

        hidden_states_w = queries_w
        hidden_states_m = queries_m

        # water
        queries_w = self.query_w(queries_w).view(B, w_L, H, -1)
        keys_w = self.key_w(keys_w).view(B, w_S, H, -1)
        values_w = self.value_w(values_w).view(B, w_S, H, -1)

        # mete
        queries_m = self.query_m(queries_m).view(B, m_L, H, -1)
        keys_m = self.key_m(keys_m).view(B, m_S, H, -1)
        values_m = self.value_m(values_m).view(B, m_S, H, -1)

        # 双向多模态注意力
        contexts, attns = self.inner_attention(
            queries_w,  keys_w, values_w,
            queries_m, keys_m, values_m,
            attn_mask
        )

        # hidden_states_w residual connect
        # contexts[0] = contexts[0] + hidden_states_w
        # contexts[2] = contexts[2] + hidden_states_w

        # hidden_states_w residual connect
        # contexts[1] = contexts[1] + hidden_states_m
        # contexts[3] = contexts[3] + hidden_states_m

        # Combine Multimodel outputs
        contexts_water = self.dropout(self.projection_water((contexts[0] + contexts[2]) / 2.) + hidden_states_w)
        contexts_mete = self.dropout(self.projection_mete((contexts[1] + contexts[3]) / 2.) + hidden_states_m)

        return contexts_water, contexts_mete, attns


class BMAttentionBlock(nn.Module):
    def __init__(self, bma_layers, attn_layers, norm_layers=None):
        super(BMAttentionBlock, self).__init__()
        self.bma_layers = nn.ModuleList(bma_layers)
        self.attn_layers = nn.ModuleList(attn_layers)

        # self.norm = norm_layer
        self.norm_layers = nn.ModuleList(norm_layers)

    def forward(self, water_x, mete_x):
        # water_x: [B, L, D]
        # mete_x: [B, S, D]
        bma_attns = []; attns = []

        for bma_layer in self.bma_layers:
            water_x, mete_x, bma_attn = bma_layer(water_x, water_x, water_x, mete_x, mete_x, mete_x)
            bma_attns.append(bma_attn)

        # water_x.shape [B x L x D]
        # mete_x.shape [B x S x D]
        hidden_states = torch.cat((water_x, mete_x), dim=1)     # [B x (L + S) x D]
        hidden_states = self.norm_layers[0](hidden_states)

        for attn_layer in self.attn_layers:
            y = hidden_states
            hidden_states, attn = attn_layer(hidden_states, hidden_states, hidden_states, attn_mask=None)
            hidden_states = hidden_states + y
            attns.append(attn)

        # if self.norm is not None:
        hidden_states = self.norm_layers[1](hidden_states)

        return hidden_states, bma_attns, attns


if __name__ == '__main__':
    batch = 8
    w_seq_len = 24
    m_seq_len = 48
    d_model = 128
    n_heads = 8

    water_tensor = torch.rand((batch, w_seq_len, d_model))
    mete_tensor = torch.rand((batch, m_seq_len, d_model))

    layer = BMAttentionLayer(
        BMAttention(output_attention=True),
        d_model=d_model,
        n_heads=n_heads
    )

    results = layer(water_tensor, water_tensor, water_tensor, mete_tensor, mete_tensor, mete_tensor, attn_mask=None)

    print(results[0].shape)
    print(results[1].shape)
