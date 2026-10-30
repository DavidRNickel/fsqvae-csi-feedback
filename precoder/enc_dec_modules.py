import torch

from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import math

#
#
class EncoderHead(nn.Module):
    def __init__(self, encoder_layer_config: dict, encoder_config: dict, pos_enc):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(**encoder_layer_config)
        self.encoder = TransformerEncoder(encoder_layer, **encoder_config)
        self.pos_enc = pos_enc

    def forward(self, x, src_key_padding_mask=None):
        return self.encoder(self.pos_enc(x), src_key_padding_mask=src_key_padding_mask)

#
# Standard positional encoding from torch.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:x.size(1),:]