# -*- coding: utf-8 -*-
"""
@Project ：My_BP_code 
@Time    : 2023/10/6 9:37
@Author  : Rao Zhi
@File    : transformer_mapping.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm 

"""

import torch
import torch.nn as nn
import torch.nn.functional as nnf
from typing import Tuple, Optional, List


class MlpTransformer(nn.Module):

    def __init__(
            self,
            input_size: int,  # the input size of mlp
            hidden_size: int,  # the hidden layer size of mlp
            output_size: Optional[int] = None,  # the output size of mlp
            act=nnf.relu,
            dropout: float = 0.0
    ) -> None:
        super().__init__()
        output_size = output_size if output_size is not None else input_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.act = act
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MultiHeadAttention(nn.Module):

    def __init__(
            self,
            query_size: int,  # 768
            key_value_size: int,  # 768
            num_heads: int,
            bias=True,
            dropout: float = 0.0
    ) -> None:
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads  # 8
        self.head_size = query_size // num_heads  # the size of each head   # 768 // 8 == 96
        self.scale = self.head_size ** -0.5  # normalization factor for each head   # 0.10206207261596575
        self.to_queries = nn.Linear(query_size, query_size, bias=bias)
        #  projecting key and value together and spliting them for computing efficiently
        self.to_keys_values = nn.Linear(key_value_size, 2 * query_size, bias=bias)
        self.project = nn.Linear(query_size, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        key_value = key_value if key_value is not None else query
        b, n, d_query = query.shape  # b:80  n=20  d_query=768
        _, m, _ = key_value.shape  # m:20

        # a = self.to_queries(query)    # [80, 10, 768]

        queries = self.to_queries(query).reshape(b, n, self.num_heads, self.head_size)
        # (batch_size, n_seq, num_heads, head_size)   (80, 20, 8, 96)

        keys_values = self.to_keys_values(key_value).reshape(b, m, 2, self.num_heads, self.head_size)
        # (batch_size, m_seq, 2, num_heads, head_size)  (80, 20, 2, 8, 96)

        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        # keys:(batch_size, m_seq, num_heads, head_size)        # (80, 20, 8, 96)
        # values:(batch_size, m_seq, num_heads, head_size)      # (80, 20, 8, 96)

        attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale
        # (batch_size, n_seq, m_seq, num_heads)     # (80, 20, 20, 8)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(dim=1)  # expending dimension, shape: (batch_size, 1, m_seq)
            attention = attention.masked_fill(mask.unsqueeze(dim=3), float(
                "-inf"))  # expending dimension n_seq head and fill -inf according to mask

        attention = attention.softmax(dim=2)  # softmax alongside the dimension of key_value pairs
        outputs = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, d_query)
        # (batch_size, n_seq, d_query)   (80, 20, 768)

        outputs = self.project(outputs)  # (80, 20, 768)
        return outputs, attention


class TransformerLayer(nn.Module):

    def __init__(
            self,
            query_size: int,
            key_value_size: int,
            num_heads: int,
            mlp_ratio=4.0,
            bias=False,
            dropout: float = 0.0,
            act=nnf.relu,
            norm_layer: nn.Module = nn.LayerNorm
    ) -> None:
        super(TransformerLayer, self).__init__()
        self.norm1 = norm_layer(query_size)
        self.attn = MultiHeadAttention(query_size, key_value_size, num_heads, bias=bias, dropout=dropout)
        self.norm2 = norm_layer(query_size)
        self.mlp = MlpTransformer(query_size, int(query_size * mlp_ratio), act=act, dropout=dropout)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        query_, self.attention = self.attn(self.norm1(query), key_value, mask)
        query = query + query_
        query = query + self.mlp(self.norm2(query))
        return query


class Transformer(nn.Module):

    def __init__(
            self,
            query_size: int,  # query size
            num_layers: int,  # number of layer
            num_heads: int,  # number of head
            key_value_size: Optional[int] = None,  # key/value size
            mlp_ratio: float = 2.0,  # ratio for hidden size in mlp
            act=nnf.relu,  # activation
            norm_layer: nn.Module = nn.LayerNorm  # normalization
    ) -> None:
        super(Transformer, self).__init__()
        key_value_size = key_value_size if key_value_size is not None else query_size
        layers = []
        for _ in range(num_layers):
            layers.append(TransformerLayer(query_size, key_value_size, num_heads, mlp_ratio=mlp_ratio, act=act,
                                           norm_layer=norm_layer))
        self.layers = nn.Sequential(*layers)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        self.attentions = []
        for layer in self.layers:
            query = layer(query, key_value, mask)
            self.attentions.append(layer.attention)
        return query


class MappingNetwork(nn.Module):

    def __init__(
            self,
            clip_project_length: int,
            clip_hidden_size: int,
            prefix_length: int,
            d_model: int,  # the hidden size of language model
            num_layers: int = 8,
            num_heads: int = 8
    ) -> None:
        super(MappingNetwork, self).__init__()
        self.clip_project_length = clip_project_length
        # projector for input
        self.linear = nn.Linear(clip_hidden_size, clip_project_length * d_model)
        # learnable prefix embeddings
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, d_model), requires_grad=True)
        # self.prefix_const1 = nn.Parameter(torch.randn(4, 256), requires_grad=True)
        self.transformer = Transformer(d_model, num_layers, num_heads)
        """
        d_model = 768
        num_layers = 8
        num_heads = 8
        """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: clip cls feature with a shape of (batch_size, clip_hidden_size)
        Return:
            the embeddings of prefix with the shape of (batch_size, prefix_length, d_model)
        """
        # x shape: (80, 512)
        # t = self.linear(x)      # (80, 7680)
        x = self.linear(x).view(x.shape[0], self.clip_project_length, -1)

        # (b, clip_project_length, d_model)  (80, 10, 768)

        # g = self.prefix_const   # (10, 768)
        # h = self.prefix_const.unsqueeze(dim=0)   # (1, 10, 768)
        prefix = self.prefix_const.unsqueeze(dim=0).expand(x.shape[0],
                                                           *self.prefix_const.shape)

        # prefix = self.prefix_const1.unsqueeze(dim=0).expand(x.shape[0],
        #                                                     *self.prefix_const1.shape)
        # (b, prefix_length, d_model)   (80, 10, 768)

        inputs = torch.cat((x, prefix), dim=1)  # (b, clip_project_length + prefix_length, d_model)
        # (80, 20, 768)

        # outputs1 = self.transformer(inputs)     # (80, 20, 768)
        outputs = self.transformer(inputs)[:, self.clip_project_length:, :]  # (b, prefix_length, d_model)
        # (80, 10, 768)

        return outputs


if __name__ == '__main__':
    continuous_length = 10
    clip_project_length = 10
    clip_hidden_size = 512
    num_layers = 8
    num_heads = 8
    gpt_hidden_size = 768
    continuous_prompt = torch.rand(2048, 4, 256)
    mapping_network = MappingNetwork(clip_project_length, clip_hidden_size, continuous_length,
                                     gpt_hidden_size, num_layers, num_heads)

    continuous_embeddings = mapping_network(continuous_prompt).view(-1, continuous_length, gpt_hidden_size)
    print()
