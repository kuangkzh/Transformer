import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from Utils import *


class EncoderLayer(nn.Module):
    # Encoder is made up of self-attn and feed forward (defined below)
    def __init__(self, d_model, dff, h, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(h, d_model, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, dff, dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(d_model)

    def forward(self, x, mask):
        x = self.norm(x+self.dropout(self.self_attn(x, x, x, mask)))
        return self.norm(x+self.dropout(self.feed_forward(x)))


class DecoderLayer(nn.Module):
    # Decoder is made of self-attn, src-attn, and feed forward (defined below)
    def __init__(self, d_model, dff, h, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(h, d_model, dropout)
        self.src_attn = MultiHeadedAttention(h, d_model, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, dff, dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(d_model)

    def forward(self, x, s, src_mask, tgt_mask):
        x = self.norm(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm(x + self.dropout(self.src_attn(x, s, s, src_mask)))
        return self.norm(x+self.dropout(self.feed_forward(x)))


class Encoder(nn.Module):
    # Core encoder is a stack of N layers
    def __init__(self, d_model, dff, h, dropout, N):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, dff, h, dropout) for _ in range(N)])

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class Decoder(nn.Module):
    # Generic N layer decoder with masking.
    def __init__(self, d_model, dff, h, dropout, N):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, dff, h, dropout) for _ in range(N)])

    def forward(self, x, s, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, s, src_mask, tgt_mask)
        return x


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, src_max_len, tgt_vocab_size, tgt_max_len,
                 N=6, d_model=512, h=8, dff=2048, dropout=0.1):
        super(Transformer, self).__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.src_pe = PositionalEncoding(d_model, dropout, src_max_len)
        self.tgt_pe = PositionalEncoding(d_model, dropout, tgt_max_len)
        self.encoder = Encoder(d_model, dff, h, dropout, N)
        self.decoder = Decoder(d_model, dff, h, dropout, N)
        self.linear = nn.Linear(d_model, tgt_vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, src_seq, tgt_seq, src_mask=None, tgt_mask=None):
        src_seq = self.src_pe(self.src_embedding(src_seq))
        tgt_seq = self.src_pe(self.src_embedding(tgt_seq))
        enc_src = self.encoder(src_seq, src_mask)
        output = self.decoder(tgt_seq, enc_src, src_mask, tgt_mask)
        output = self.softmax(self.linear(output))
        return output
