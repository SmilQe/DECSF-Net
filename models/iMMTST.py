from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from layers.BMAttention_Layer import BMAttentionBlock, BMAttentionLayer, BMAttention
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.Attention_Layer import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted

@dataclass
class Configs():
    pred_len: int = 96
    output_attention: bool = False
    enc_water_in: int = 4
    enc_mete_in: int = 4
    d_model: int = 512
    water_seq_len: int = 336
    mete_seq_len: int = 432
    embed: str = 'timeF'
    freq: str = 'h'
    dropout: float = 0.1
    n_heads: int = 8
    d_ff: int = 2048
    activation: str = 'gelu'
    ew_layers: int = 1
    em_layers: int = 1
    b_layers: int = 1
    ba_layers: int = 1

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        # self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        # self.use_norm = configs.use_norm

        # Embedding
        self.enc_water_embedding = DataEmbedding_inverted(
            configs.water_seq_len, configs.d_model, configs.embed, freq=configs.freq,
            dropout=configs.dropout
        )

        self.enc_mete_embedding = DataEmbedding_inverted(
            configs.mete_seq_len, configs.d_model, configs.embed, freq=configs.freq,
            dropout=configs.dropout
        )

        # water encoder
        self.water_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        attention=FullAttention(
                            mask_flag=False,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),
                        d_model=configs.d_model,
                        n_heads=configs.n_heads,
                    ),
                    d_model=configs.d_model,
                    d_ff=configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for _ in range(configs.ew_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
        )

        # mete encoder
        self.mete_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        attention=FullAttention(
                            mask_flag=False,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),
                        d_model=configs.d_model,
                        n_heads=configs.n_heads
                    ),
                    d_model=configs.d_model,
                    d_ff=configs.d_ff,
                    dropout=configs.dropout,
                ) for _ in range(configs.em_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
        )

        # MMBlock
        self.MMBlock = BMAttentionBlock(
            [
                BMAttentionLayer(
                    attention=BMAttention(
                        attention_dropout=configs.dropout,
                        output_attention=configs.output_attention,
                    ),
                    d_model=configs.d_model,
                    n_heads=configs.n_heads,
                    dropout_rate=configs.dropout,
                ) for _ in range(configs.b_layers)
            ],
            [
                AttentionLayer(
                    attention=FullAttention(
                        mask_flag=False,
                        attention_dropout=configs.dropout,
                        output_attention=configs.output_attention,
                    ),
                    d_model=configs.d_model,
                    n_heads=configs.n_heads,
                ) for _ in range(configs.ba_layers)
            ],
            norm_layers=[nn.LayerNorm(configs.d_model), nn.LayerNorm(configs.d_model)],
        )

        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forward(self, water_enc, water_enc_mark, mete_enc, mete_enc_mark):
        
        water_means = water_enc.mean(1, keepdim=True).detach()
        water_enc = water_enc - water_means
        water_stdev = torch.sqrt(torch.var(water_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        water_enc /= water_stdev
        
        mete_means = mete_enc.mean(1, keepdim=True).detach()
        mete_enc = mete_enc - mete_means
        mete_stdev = torch.sqrt(torch.var(mete_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        mete_enc /= mete_stdev
        
        _, _, N = water_enc.shape

        # water_enc.shape [B, L, N]
        # water_enc_mark.shape [B, L, 4]
        # Embedding
        # [B, 336, 4], [B, 336, 4] -> torch.cat: [B, 8, 336] -nn.Linear-> [B, 8, 512]
        enc_water_out = self.enc_water_embedding(water_enc, water_enc_mark)
        enc_water_out, enc_water_attns = self.water_encoder(enc_water_out)

        # [B, 432, 10], [B, 432, 4] -> torch.cat: [B, 8, 432] -nn.Linear-> [B, 14, 512]
        enc_mete_out = self.enc_mete_embedding(mete_enc, mete_enc_mark)
        enc_mete_out, enc_mete_attns = self.mete_encoder(enc_mete_out)

        # MMBlock hidden_states.shape [B, 22, 512]
        hidden_states, MM_bma_attns, MM_attns = self.MMBlock(enc_water_out, enc_mete_out)

        # hidden_states.shape [B, 22, 512]
        # decoder
        dec_water_out = self.projector(hidden_states).permute(0, 2, 1)[:, :, :N]
        dec_water_out = dec_water_out * (water_stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_water_out = dec_water_out + (water_means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_water_out[:, -self.pred_len:, :]

if __name__ == '__main__':
    # test code
    water_enc = torch.rand((32, 336, 4))
    water_mark_enc = torch.rand((32, 336, 4))

    mete_enc = torch.rand((32, 432, 10))
    mete_mark_enc = torch.rand((32, 432, 4))

    configs = Configs()
    model = Model(configs)

    dec_water_out = model(water_enc, water_mark_enc, mete_enc, mete_mark_enc)

    print(dec_water_out.shape)
    print("Done")

