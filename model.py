import numpy as np
import pandas as pd
import torch
from torch import  nn
import torch.nn.functional as F
import copy
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
from dgllife.model.model_zoo.gcn_predictor import GCNPredictor
batch_size= 128

# 128
def get_position_encoding(seq_len, embed):
    pe = np.array([[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(seq_len)])
    pe[:, 0::2] = np.sin(pe[:, 0::2])
    pe[:, 1::2] = np.cos(pe[:, 1::2])
    return pe

class Positional_Encoding(nn.Module):
    def __init__(self, embed, pad_size, dropout):
        super(Positional_Encoding, self).__init__()
        self.pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = x + nn.Parameter(self.pe, requires_grad=False).to("cuda")
        out = self.dropout(out)
        return out



class Scaled_Dot_Product_Attention(nn.Module):
    '''Scaled Dot-Product'''
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        attention = torch.matmul(Q, K.permute(0, 2, 1))  # Q*K^T
        if scale:
            attention = attention * scale
        # if mask:  # TODO change this
        #     attention = attention.masked_fill_(mask == 0, -1e9)
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context

class Multi_Head_Attention(nn.Module):
    def __init__(self, dim_model, num_head, dropout=0.0):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0
        self.dim_head = dim_model // self.num_head
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        # if mask:  # TODO
        #     mask = mask.repeat(self.num_head, 1, 1)  # TODO change this
        scale = K.size(-1) ** -0.5
        context = self.attention(Q, K, V, scale) # Scaled_Dot_Product_Attention
        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        out = self.fc(context)
        out = self.dropout(out)
        out = out + x
        out = self.layer_norm(out)
        return out

class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0.0):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x
        out = self.layer_norm(out)
        return out

class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head, dropout)
        self.feed_forward = Position_wise_Feed_Forward(dim_model, hidden, dropout)

    def forward(self, x):
        out = self.attention(x)
        out = self.feed_forward(out)
        return out

class ConfigTrans(object):
    def __init__(self):
        self.model_name = 'Transformer'
        self.dropout = 0.5
        self.num_classes = 1
        self.num_epochs = 100
        self.batch_size = 128
        self.pad_size = 1
        self.learning_rate = 0.001
        self.embed = 50
        # self.embed = 256
        self.dim_model = 50
        # self.dim_model = 256
        self.hidden = 1024
        # self.hidden = 512
        self.last_hidden = 512
        # self.num_head = 8
        self.num_head = 2
        self.num_encoder = 2

config = ConfigTrans()

class Transformer_test(nn.Module):
    def __init__(self):
        super(Transformer_test, self).__init__()
        self.postion_embedding = Positional_Encoding(config.embed, config.pad_size, config.dropout)
        self.encoder = Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(config.num_encoder)])
        self.before_list = []
        self.after_list = []
        # feature_dim = config.embed * config.dim_model
        # self.projection1 = nn.Linear(feature_dim, 128)
        # self.projection2 = nn.Linear(feature_dim, 40)


        self.projection1 = nn.Linear(batch_size * 50, 128)
        self.projection2 = nn.Linear(batch_size * 50, 40)

    def forward(self, x):
        #Transformer
        out = self.postion_embedding(x)
        for encoder in self.encoders:
            out = encoder(out)
        out = out.view(out.size(0), -1)
        rep = self.projection1(out)
        y_p=self.projection2(out)
        return rep,y_p

class SimpleSelfAttention(nn.Module):
    def __init__(self, embed_dim, dropout=0.5):
        super(SimpleSelfAttention, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.float()
        if x.dim() == 2:
            x = x.unsqueeze(1)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Attention score: (batch_size, seq_len, seq_len)
        score = torch.bmm(Q, K.transpose(1, 2)) / (x.size(-1) ** 0.5)
        attn = F.softmax(score, dim=-1)
        attn = self.dropout(attn)

        # Weighted sum: (batch_size, seq_len, embed_dim)
        context = torch.bmm(attn, V)

        # Optional: aggregate over sequence (mean pooling)
        out = torch.mean(context, dim=1)  # (batch_size, embed_dim)
        return out



class GatedFusionUnit(nn.Module):
    def __init__(self, feature_dim, dropout=0.3, use_residual=True):
        super(GatedFusionUnit, self).__init__()

        # Scalar gate
        self.scalar_gate = nn.Sequential(
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )

        # Feature-wise gate
        self.feature_gate = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid()
        )

        self.use_residual = use_residual

    def forward(self, x):
        scalar = self.scalar_gate(x)  # (batch, 1)
        feature = self.feature_gate(x)  # (batch, feature_dim)

        scalar = scalar.expand_as(x)

        out = x * scalar * feature

        if self.use_residual:
            out = out + x  # residual connection

        return out







# class Model_TGCN_NoGNN(nn.Module):
#     def __init__(self, hidden_dim_t=256, dropout=0.3, use_residual=True):
#         super(Model_TGCN_NoGNN, self).__init__()
#         # === 数值特征分支 ===
#         self.num_fc = nn.Linear(20, 32)
#         self.nrom = nn.BatchNorm1d(32)
#         self.num_scaler = nn.Parameter(torch.ones(1, 32))  # Feature-wise Scaling
#
#         # === ProtT5 MLP + Attention Pooling ===
#         self.prott5_proj = nn.Sequential(
#             nn.Linear(1024, 256),
#             nn.ReLU(),
#             nn.Linear(256, 64)
#         )
#         self.prot_attn = nn.Linear(64, 1)
#
#         # === Transformer 分支投影 ===
#         self.trans_proj = nn.Sequential(
#             nn.Linear(128, hidden_dim_t),
#             nn.ReLU(),
#             nn.Dropout(p=dropout)
#         )
#
#         # === Gated Fusion ===
#         self.gate_x_t = GatedFusionUnit(hidden_dim_t, dropout, use_residual)
#         # self.gate_x_g = GatedFusionUnit(10, dropout, use_residual)
#         self.gate_x_l = GatedFusionUnit(32, dropout, use_residual)
#         self.gate_x_p = GatedFusionUnit(64, dropout, use_residual)
#
#         # === 融合部分 ===
#         fusion_dim = hidden_dim_t + 32 + 64
#         self.res_fc = nn.Linear(fusion_dim, fusion_dim)
#         self.layernorm = nn.LayerNorm(fusion_dim)
#
#         # === 最终输出 ===
#         self.fc1 = nn.Linear(fusion_dim, 1)
#
#         # === t-SNE 保存用 ===
#         self.tsne_list = []
#         self.out_list = []
#
#         self.sig = nn.Sigmoid()
#
#     def forward(self, x_t, x_l, x_p):
#         # === 输入送 GPU ===
#         # gcn_i = x_g.to("cuda")
#         out = x_t.to("cuda")
#         x_l = x_l.to("cuda")
#         x_p = x_p.to("cuda")
#
#         # === 数值特征分支 ===
#         num_out = self.num_fc(x_l)
#         num_out = self.nrom(num_out).to("cuda")
#         num_out = num_out * self.num_scaler.to("cuda")
#
#         # === ProtT5 Attention Pooling ===
#
#         prot_proj = self.prott5_proj(x_p.unsqueeze(1))
#         attn_score = torch.softmax(self.prot_attn(prot_proj), dim=1)
#         prot_out = (attn_score * prot_proj).sum(dim=1)
#
#         # === Transformer 分支投影 + Gate ===
#         out = self.trans_proj(out)
#         out_gated = self.gate_x_t(out)
#
#         # === 其他 Gate ===
#         # gcn_gated = self.gate_x_g(gcn_i)
#         num_gated = self.gate_x_l(num_out)
#         prot_gated = self.gate_x_p(prot_out)
#
#         # === 融合 ===
#         fusion_out = torch.cat([out_gated, num_gated, prot_gated], dim=-1)
#
#         # === Residual + LayerNorm ===
#         fusion_out = self.layernorm(fusion_out + self.res_fc(fusion_out))
#
#         # === t-SNE 保存 ===
#         tsne = fusion_out.detach().to("cpu").numpy()
#         self.tsne_list.append(tsne)
#
#         # === 最终输出 ===
#         out = self.fc1(fusion_out)
#         out = self.sig(out)
#
#         return out


# class Model_TGCN_NoPretrain(nn.Module):
#     def __init__(self, hidden_dim_t=256, dropout=0.3, use_residual=True):
#         super(Model_TGCN_NoPretrain, self).__init__()
#
#         # === 数值特征分支 ===
#         self.num_fc = nn.Linear(20, 32)
#         self.nrom = nn.BatchNorm1d(32)
#         self.num_scaler = nn.Parameter(torch.ones(1, 32))  # Feature-wise Scaling
#
#         # === ProtT5 MLP + Attention Pooling ===
#         self.prott5_proj = nn.Sequential(
#             nn.Linear(1024, 256),
#             nn.ReLU(),
#             nn.Linear(256, 64)
#         )
#         self.prot_attn = nn.Linear(64, 1)
#
#         # === Transformer 分支投影 ===
#         self.trans_proj = nn.Sequential(
#             nn.Linear(128, hidden_dim_t),
#             nn.ReLU(),
#             nn.Dropout(p=dropout)
#         )
#
#         # === Gated Fusion ===
#         self.gate_x_t = GatedFusionUnit(hidden_dim_t, dropout, use_residual)
#         self.gate_x_g = GatedFusionUnit(40, dropout, use_residual)
#         self.gate_x_l = GatedFusionUnit(32, dropout, use_residual)
#         # self.gate_x_p = GatedFusionUnit(128, dropout, use_residual)
#
#         # === 融合部分 ===
#         fusion_dim = hidden_dim_t + 40 + 32
#         self.res_fc = nn.Linear(fusion_dim, fusion_dim)
#         self.layernorm = nn.LayerNorm(fusion_dim)
#
#         # === 最终输出 ===
#         self.fc1 = nn.Linear(fusion_dim, 1)  # Regression → 不加 Sigmoid
#
#         # === t-SNE 保存用 ===
#         self.tsne_list = []
#         self.out_list = []
#         self.sig = nn.Sigmoid()
#
#     def forward(self, x_t, x_g, x_l,  prot_embeddings=None):
#         # === 输入送 GPU ===
#         gcn_i = x_g.to("cuda")
#         out = x_t.to("cuda")
#         x_l = x_l.to("cuda")
#         # x_p = x_p.to("cuda")
#
#         # === 数值特征分支 ===
#         num_out = self.num_fc(x_l)
#         num_out = self.nrom(num_out).to("cuda")
#         num_out = num_out * self.num_scaler.to("cuda")
#
#
#
#         # === Transformer 分支投影 + Gate ===
#         out = self.trans_proj(out)
#         out_gated = self.gate_x_t(out)
#
#         # === 其他 Gate ===
#         gcn_gated = self.gate_x_g(gcn_i)
#         num_gated = self.gate_x_l(num_out)
#         # prot_gated = self.gate_x_p(prot_out)
#
#         # === 融合 ===
#         fusion_out = torch.cat([out_gated, gcn_gated, num_gated], dim=-1)
#
#         # === Residual + LayerNorm ===
#         fusion_out = self.layernorm(fusion_out + self.res_fc(fusion_out))
#
#         # === t-SNE 保存 ===
#         tsne = fusion_out.detach().to("cpu").numpy()
#         self.tsne_list.append(tsne)
#
#         # === 最终输出 ===
#         out = self.fc1(fusion_out)
#         out = self.sig(out)
#
#         return out


# class Model_TGCN(nn.Module):
#     def __init__(self, hidden_dim_t=256, dropout=0.3, use_residual=True):
#         super(Model_TGCN, self).__init__()
#
#         # === 数值特征分支 ===
#         self.num_fc = nn.Linear(20, 32)
#         self.nrom = nn.BatchNorm1d(32)
#         self.num_scaler = nn.Parameter(torch.ones(1, 32))  # Feature-wise Scaling
#
#         # === ProtT5 MLP + Attention Pooling ===
#         self.prott5_proj = nn.Sequential(
#             nn.Linear(1024, 256),
#             nn.ReLU(),
#             nn.Linear(256, 64)
#         )
#         self.prot_attn = nn.Linear(64, 1)
#
#         # === Transformer 分支投影 ===
#         self.trans_proj = nn.Sequential(
#             nn.Linear(128, hidden_dim_t),
#             nn.ReLU(),
#             nn.Dropout(p=dropout)
#         )
#
#         # === Gated Fusion ===
#         self.gate_x_t = GatedFusionUnit(hidden_dim_t, dropout, use_residual)
#         self.gate_x_g = GatedFusionUnit(128, dropout, use_residual)
#         self.gate_x_l = GatedFusionUnit(32, dropout, use_residual)
#         self.gate_x_p = GatedFusionUnit(64, dropout, use_residual)
#
#         # === 融合部分 ===
#         fusion_dim = hidden_dim_t + 128 + 32 + 64
#         self.res_fc = nn.Linear(fusion_dim, fusion_dim)
#         self.layernorm = nn.LayerNorm(fusion_dim)
#
#         # === 最终输出 ===
#         self.fc1 = nn.Linear(fusion_dim, 1)
#
#         # === t-SNE 保存用 ===
#         self.tsne_list = []
#         self.out_list = []
#
#         self.sig = nn.Sigmoid()
#
#     def forward(self, x_t, x_g, x_l, x_p, prot_embeddings=None):
#         # === 输入送 GPU ===
#         gcn_i = x_g.to("cuda")
#         out = x_t.to("cuda")
#         x_l = x_l.to("cuda")
#         x_p = x_p.to("cuda")
#
#         # === 数值特征分支 ===
#         num_out = self.num_fc(x_l)
#         num_out = self.nrom(num_out).to("cuda")
#         num_out = num_out * self.num_scaler.to("cuda")
#
#         # === ProtT5 Attention Pooling ===
#         if prot_embeddings is not None:
#             prot_embeddings = prot_embeddings.to("cuda")
#             prot_proj = self.prott5_proj(prot_embeddings)
#             attn_score = torch.softmax(self.prot_attn(prot_proj), dim=1)
#             prot_out = (attn_score * prot_proj).sum(dim=1)
#         else:
#             prot_proj = self.prott5_proj(x_p.unsqueeze(1))
#             attn_score = torch.softmax(self.prot_attn(prot_proj), dim=1)
#             prot_out = (attn_score * prot_proj).sum(dim=1)
#
#         # === Transformer 分支投影 + Gate ===
#         out = self.trans_proj(out)
#         out_gated = self.gate_x_t(out)
#
#         # === 其他 Gate ===
#         gcn_gated = self.gate_x_g(gcn_i)
#         num_gated = self.gate_x_l(num_out)
#         prot_gated = self.gate_x_p(prot_out)
#
#         # === 融合 ===
#         fusion_out = torch.cat([out_gated, gcn_gated, num_gated, prot_gated], dim=-1)
#
#         # === Residual + LayerNorm ===
#         fusion_out = self.layernorm(fusion_out + self.res_fc(fusion_out))
#
#         # === t-SNE 保存 ===
#         # tsne = fusion_out.detach().to("cpu").numpy()
#         # self.tsne_list.append(tsne)
#
#         # === 最终输出 ===
#         out = self.fc1(fusion_out)
#
#         out = self.sig(out)
#
#         return out


import torch
import torch.nn as nn


import torch
import torch.nn as nn


class MMTMModule(nn.Module):
    def __init__(self, dim_a, dim_b, reduction_ratio=4):
        super(MMTMModule, self).__init__()
        reduced_dim = int((dim_a + dim_b) / reduction_ratio)

        self.fc_squeeze = nn.Sequential(
            nn.Linear(dim_a + dim_b, reduced_dim),
            nn.ReLU()
        )

        self.fc_a = nn.Sequential(nn.Linear(reduced_dim, dim_a), nn.Sigmoid())
        self.fc_b = nn.Sequential(nn.Linear(reduced_dim, dim_b), nn.Sigmoid())

    def forward(self, a, b):
        z = torch.cat([a, b], dim=-1)
        z = self.fc_squeeze(z)
        return 5*a * self.fc_a(z), 2*b * self.fc_b(z)


class Model_TGCN(nn.Module):
    def __init__(self, hidden_dim_t=256, dropout=0.5):
        super(Model_TGCN, self).__init__()

        # 数值特征
        self.num_fc = nn.Linear(20, 32)
        self.nrom = nn.BatchNorm1d(32)
        self.num_scaler = nn.Parameter(torch.ones(1, 32))

        # ProtT5 表征
        self.prott5_proj = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64)
        )
        self.prot_attn = nn.Linear(64, 1)

        # Transformer 投影
        self.trans_proj = nn.Sequential(
            nn.Linear(128, hidden_dim_t),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        # 仅对 Transformer + GCN 做门控融合
        self.mmtm = MMTMModule(dim_a=hidden_dim_t, dim_b=40)


        fusion_dim = hidden_dim_t + 40 + 64 + 32
        self.res_fc = nn.Linear(fusion_dim, fusion_dim)
        self.layernorm = nn.LayerNorm(fusion_dim)
        self.fc1 = nn.Linear(fusion_dim, 1)
        self.sigmoid = nn.Sigmoid()

        self.tsne_list = []

    def forward(self, x_t, x_g, x_l, x_p, prot_embeddings=None):
        x_t = x_t.to("cuda")
        x_g = x_g.to("cuda")
        x_l = x_l.to("cuda")
        x_p = x_p.to("cuda")

        # 数值特征
        num_out = self.num_fc(x_l)
        num_out = self.nrom(num_out)
        num_out = num_out * self.num_scaler.to("cuda")

        # ProtT5 注意力池化
        if prot_embeddings is not None:
            prot_proj = self.prott5_proj(prot_embeddings.to("cuda"))
        else:
            prot_proj = self.prott5_proj(x_p.unsqueeze(1))
        attn_score = torch.softmax(self.prot_attn(prot_proj), dim=1)
        prot_out = (attn_score * prot_proj).sum(dim=1)

        # Transformer 投影
        trans_out = self.trans_proj(x_t)

        # GCN + Transformer 融合（MMTM门控）
        trans_gated, gcn_gated = self.mmtm(trans_out, x_g)

        # 总融合
        fusion_out = torch.cat([trans_gated, gcn_gated, prot_out, num_out], dim=-1)
        fusion_out = self.layernorm(fusion_out + self.res_fc(fusion_out))

        out = self.sigmoid(self.fc1(fusion_out))
        return out,gcn_gated,trans_gated,fusion_out


class Model_TGCN_NoTransformer(nn.Module):
    def __init__(self, dropout=0.5):
        super(Model_TGCN_NoTransformer, self).__init__()

        # 数值特征分支（20 -> 32）
        self.num_fc = nn.Linear(20, 32)
        self.nrom = nn.BatchNorm1d(32)
        self.num_scaler = nn.Parameter(torch.ones(1, 32))

        # ProtT5 表征分支（1024 -> 256 -> 64）
        self.prott5_proj = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64)
        )
        self.prot_attn = nn.Linear(64, 1)

        # 融合维度为：GCN (40) + ProtT5 (64) + 数值 (32)
        fusion_dim = 40 + 64 + 32
        self.res_fc = nn.Linear(fusion_dim, fusion_dim)
        self.layernorm = nn.LayerNorm(fusion_dim)
        self.fc1 = nn.Linear(fusion_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_g, x_l, x_p, prot_embeddings=None):
        x_g = x_g.to("cuda")
        x_l = x_l.to("cuda")
        x_p = x_p.to("cuda")

        # 数值特征处理
        num_out = self.num_fc(x_l)
        num_out = self.nrom(num_out)
        num_out = num_out * self.num_scaler.to("cuda")

        # ProtT5 注意力池化处理
        if prot_embeddings is not None:
            prot_proj = self.prott5_proj(prot_embeddings.to("cuda"))
        else:
            prot_proj = self.prott5_proj(x_p.unsqueeze(1))
        attn_score = torch.softmax(self.prot_attn(prot_proj), dim=1)
        prot_out = (attn_score * prot_proj).sum(dim=1)

        # 融合所有特征
        fusion_out = torch.cat([x_g, prot_out, num_out], dim=-1)
        fusion_out = self.layernorm(fusion_out + self.res_fc(fusion_out))

        out = self.sigmoid(self.fc1(fusion_out))
        return out

import torch
import torch.nn as nn

class Model_TGCN_NoPretrain(nn.Module):
    def __init__(self, hidden_dim_t=256, dropout=0.5):
        super(Model_TGCN_NoPretrain, self).__init__()

        # 数值特征分支
        self.num_fc = nn.Linear(20, 32)
        self.nrom = nn.BatchNorm1d(32)
        self.num_scaler = nn.Parameter(torch.ones(1, 32))

        # Transformer 投影分支
        self.trans_proj = nn.Sequential(
            nn.Linear(128, hidden_dim_t),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        # MMTM门控融合（Transformer + GCN）
        self.mmtm = MMTMModule(dim_a=hidden_dim_t, dim_b=40)

        # 总融合维度（无ProtT5，原来减去64）
        fusion_dim = hidden_dim_t + 40 + 32
        self.res_fc = nn.Linear(fusion_dim, fusion_dim)
        self.layernorm = nn.LayerNorm(fusion_dim)
        self.fc1 = nn.Linear(fusion_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_t, x_g, x_l):
        x_t = x_t.to("cuda")
        x_g = x_g.to("cuda")
        x_l = x_l.to("cuda")

        # 数值特征分支处理
        num_out = self.num_fc(x_l)
        num_out = self.nrom(num_out)
        num_out = num_out * self.num_scaler.to("cuda")

        # Transformer 投影
        trans_out = self.trans_proj(x_t)

        # 门控融合（Transformer + GCN）
        trans_gated, gcn_gated = self.mmtm(trans_out, x_g)

        # 特征融合
        fusion_out = torch.cat([trans_gated, gcn_gated, num_out], dim=-1)
        fusion_out = self.layernorm(fusion_out + self.res_fc(fusion_out))

        out = self.sigmoid(self.fc1(fusion_out))
        return out


class Model_TGCN_NoGNN(nn.Module):
    def __init__(self, hidden_dim_t=256, dropout=0.5):
        super(Model_TGCN_NoGNN, self).__init__()

        # 数值特征
        self.num_fc = nn.Linear(20, 32)
        self.nrom = nn.BatchNorm1d(32)
        self.num_scaler = nn.Parameter(torch.ones(1, 32))

        # ProtT5 表征
        self.prott5_proj = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64)
        )
        self.prot_attn = nn.Linear(64, 1)

        # Transformer 投影
        self.trans_proj = nn.Sequential(
            nn.Linear(128, hidden_dim_t),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        # 总融合维度（去掉原来的 GCN 特征 40 维）
        fusion_dim = hidden_dim_t + 64 + 32
        self.res_fc = nn.Linear(fusion_dim, fusion_dim)
        self.layernorm = nn.LayerNorm(fusion_dim)
        self.fc1 = nn.Linear(fusion_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_t, x_l, x_p, prot_embeddings=None):
        x_t = x_t.to("cuda")
        x_l = x_l.to("cuda")
        x_p = x_p.to("cuda")

        # 数值特征
        num_out = self.num_fc(x_l)
        num_out = self.nrom(num_out)
        num_out = num_out * self.num_scaler.to("cuda")

        # ProtT5 注意力池化
        if prot_embeddings is not None:
            prot_proj = self.prott5_proj(prot_embeddings.to("cuda"))
        else:
            prot_proj = self.prott5_proj(x_p.unsqueeze(1))
        attn_score = torch.softmax(self.prot_attn(prot_proj), dim=1)
        prot_out = (attn_score * prot_proj).sum(dim=1)

        # Transformer 投影
        trans_out = self.trans_proj(x_t)

        # 总融合（不含 GCN 特征）
        fusion_out = torch.cat([trans_out, prot_out, num_out], dim=-1)
        fusion_out = self.layernorm(fusion_out + self.res_fc(fusion_out))

        out = self.sigmoid(self.fc1(fusion_out))
        return out


import torch
import torch.nn as nn

class Model_TGCN_NoGate(nn.Module):
    def __init__(self, hidden_dim_t=256, dropout=0.5):
        super(Model_TGCN_NoGate, self).__init__()

        # 数值特征
        self.num_fc = nn.Linear(20, 32)
        self.nrom = nn.BatchNorm1d(32)
        self.num_scaler = nn.Parameter(torch.ones(1, 32))

        # ProtT5 表征
        self.prott5_proj = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64)
        )
        self.prot_attn = nn.Linear(64, 1)

        # Transformer 投影
        self.trans_proj = nn.Sequential(
            nn.Linear(128, hidden_dim_t),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        # 融合层（不再经过 MMTM，仅直接拼接）
        fusion_dim = hidden_dim_t + 40 + 64 + 32
        self.res_fc = nn.Linear(fusion_dim, fusion_dim)
        self.layernorm = nn.LayerNorm(fusion_dim)
        self.fc1 = nn.Linear(fusion_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_t, x_g, x_l, x_p, prot_embeddings=None):
        x_t = x_t.to("cuda")
        x_g = x_g.to("cuda")
        x_l = x_l.to("cuda")
        x_p = x_p.to("cuda")

        # 数值特征
        num_out = self.num_fc(x_l)
        num_out = self.nrom(num_out)
        num_out = num_out * self.num_scaler.to("cuda")

        # ProtT5 注意力池化
        if prot_embeddings is not None:
            prot_proj = self.prott5_proj(prot_embeddings.to("cuda"))
        else:
            prot_proj = self.prott5_proj(x_p.unsqueeze(1))
        attn_score = torch.softmax(self.prot_attn(prot_proj), dim=1)
        prot_out = (attn_score * prot_proj).sum(dim=1)

        # Transformer 投影
        trans_out = self.trans_proj(x_t)

        # 直接拼接 GCN + Transformer（无门控）
        fusion_out = torch.cat([trans_out, x_g, prot_out, num_out], dim=-1)
        fusion_out = self.layernorm(fusion_out + self.res_fc(fusion_out))

        out = self.sigmoid(self.fc1(fusion_out))
        return out



# class Model_TGCN_NoGate(nn.Module):
#     def __init__(self, hidden_dim_t=256, dropout=0.3):
#         super(Model_TGCN_NoGate, self).__init__()
#
#         # === 数值特征分支 ===
#         self.num_fc = nn.Linear(20, 32)
#         self.nrom = nn.BatchNorm1d(32)
#         self.num_scaler = nn.Parameter(torch.ones(1, 32))  # Feature-wise Scaling
#
#         # === ProtT5 MLP + Attention Pooling ===
#         self.prott5_proj = nn.Sequential(
#             nn.Linear(1024, 256),
#             nn.ReLU(),
#             nn.Linear(256, 64),
#             nn.ReLU()
#         )
#         self.prot_attn = nn.Linear(64, 1)
#
#         # === Transformer 分支投影 ===
#         self.trans_proj = nn.Sequential(
#             nn.Linear(128, hidden_dim_t),
#             nn.ReLU(),
#             nn.Dropout(p=dropout)
#         )
#
#         # === 融合部分 ===
#         fusion_dim = hidden_dim_t + 40 + 32 + 64  # 四个维度直接拼接
#         self.res_fc = nn.Linear(fusion_dim, fusion_dim)
#         self.layernorm = nn.LayerNorm(fusion_dim)
#
#         # === 最终输出 ===
#         self.fc1 = nn.Linear(fusion_dim, 1)
#         self.sig = nn.Sigmoid()
#
#         # === t-SNE 保存用 ===
#         self.tsne_list = []
#         self.out_list = []
#
#     def forward(self, x_t, x_g, x_l, x_p, prot_embeddings=None):
#         x_t = x_t.to("cuda")
#         x_g = x_g.to("cuda")
#         x_l = x_l.to("cuda")
#         x_p = x_p.to("cuda")
#
#         # 数值特征分支
#         num_out = self.num_fc(x_l)
#         num_out = self.nrom(num_out)
#         num_out = num_out * self.num_scaler.to("cuda")
#
#         # ProtT5 Attention Pooling
#         if prot_embeddings is not None:
#             prot_embeddings = prot_embeddings.to("cuda")
#             prot_proj = self.prott5_proj(prot_embeddings)
#             attn_score = torch.softmax(self.prot_attn(prot_proj), dim=1)
#             prot_out = (attn_score * prot_proj).sum(dim=1)
#         else:
#             prot_proj = self.prott5_proj(x_p.unsqueeze(1))
#             attn_score = torch.softmax(self.prot_attn(prot_proj), dim=1)
#             prot_out = (attn_score * prot_proj).sum(dim=1)
#
#         # Transformer 分支投影
#         trans_out = self.trans_proj(x_t)
#
#         # 直接拼接所有 embedding 表示
#         fusion_out = torch.cat([trans_out, x_g, num_out, prot_out], dim=-1)
#
#         # Residual + LayerNorm
#         fusion_out = self.layernorm(fusion_out + self.res_fc(fusion_out))
#
#         # t-SNE 保存
#         self.tsne_list.append(fusion_out.detach().cpu().numpy())
#
#         # 最终输出
#         out = self.fc1(fusion_out)
#         out = self.sig(out)
#
#         return out

# class Model_TGCN(nn.Module):
#     def __init__(self):
#         super(Model_TGCN,self).__init__()
#         self.num_fc = nn.Linear(20, 32)
#         self.nrom = nn.BatchNorm1d(32)
#         self.fc1 = nn.Linear(128+40+32+128, 1)
#         self.sig = nn.Sigmoid()
#         self.tsne_list = []
#         self.out_list = []
#
#         self.prott5_proj = nn.Sequential(
#             nn.Linear(1024, 128),
#             nn.ReLU(),
#             nn.Dropout(p=0.3)  # 可选
#         )
#
#     def forward(self,x_t,x_g,x_l,x_p):
#         gcn_i=x_g.to("cuda")
#         out=x_t.to("cuda")
#         x_l=x_l.to("cuda")
#         x_p = x_p.to("cuda")
#         num_out = self.num_fc(x_l)
#         num_out = self.nrom(num_out).to("cuda")
#         x_p_proj = self.prott5_proj(x_p).to("cuda")
#
#         # tsne = torch.cat([out,gcn_i,num_out,x_p_proj],dim=-1).detach().to('cpu').numpy()
#         # self.tsne_list.append(tsne)
#         p = torch.cat([out,gcn_i,num_out,x_p_proj],dim=-1)
#         out = self.fc1(p)
#         out = self.sig(out)
#         return out
