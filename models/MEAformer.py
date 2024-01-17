# encoding=utf-8
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from layers.AutoCorrelation import AutoCorrelationLayer, AutoCorrelation
from layers.Embed import DataEmbedding_wo_pos
from layers.RevIN import RevIN
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from pytorch_wavelets import DWTForward, DWTInverse



class Model(nn.Module):
    """
    Decomposition-Linear
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        self.channels = configs.enc_in

        self.encoder = Encoder(
            [LEncoderLayer(self.seq_len, configs.S) for l in range(configs.e_layers)]
        )

        self.decoder = Decoder(
            [
                MLPLayer(self.seq_len if l == 0 else self.pred_len, self.pred_len, configs.d_ff) for l in range(configs.d_layers)
            ]
        )

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        rev = RevIN(x.size(2)).cuda()
        x = rev(x, 'norm')
        x = self.encoder(x)
        x = self.decoder(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = rev(x, 'denorm')
        return x  # to [Batch, Output length, Channel]


class Encoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class LEncoderLayer(nn.Module):
    def __init__(self, seq_len, S, norm='ln'):
        super().__init__()
        self.seq_len = seq_len

        # self.attn = nn.LSTM(seq_len, hidden_size)
        self.attn = TemporalExternalAttn(seq_len, S)
        self.drop1 = nn.Dropout(0.2)
        self.feed2 = nn.Linear(seq_len, seq_len)
        self.norm = norm
        if norm == 'ln':
            self.norm1 = nn.LayerNorm(seq_len)
            self.norm2 = nn.LayerNorm(seq_len)
        else:
            self.norm1 = nn.BatchNorm1d(seq_len)
            self.norm2 = nn.BatchNorm1d(seq_len)

    def forward(self, x):
        # x = self.feed1(x)
        attn = self.attn(x.permute(0, 2, 1))
        x = x + attn.permute(0, 2, 1)
        if self.norm == 'ln':
            x = self.norm1(x.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            x = self.norm1(x.permute(0, 2, 1))
        x = x + self.feed2(x.permute(0, 2, 1)).permute(0, 2, 1)
        if self.norm == 'ln':
            x = self.norm2(x.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            x = self.norm2(x.permute(0, 2, 1))
        return x


class TemporalExternalAttn(nn.Module):
    def __init__(self, d_model, S=256):
        super().__init__()

        self.mk = nn.Linear(d_model, S, bias=False)
        self.mv = nn.Linear(S, d_model, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, queries):

        attn = self.mk(queries)  # bs,n,S
        attn = self.softmax(attn)  # bs,n,S
        # attn = attn / torch.sum(attn, dim=2, keepdim=True)  # bs,n,S

        out = self.mv(attn)  # bs,n,d_model
        return out


class Decoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class MLPLayer(nn.Module):
    def __init__(self, seq_len, pred_len, hidden_size):
        super().__init__()
        self.l1 = nn.Linear(seq_len, hidden_size)
        self.gelu = nn.GELU()
        self.l2 = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        x = self.l1(x)
        x = self.gelu(x)
        x = self.l2(x)
        return x


