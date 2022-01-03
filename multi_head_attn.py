import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        :param embed_dim: 词嵌入的维度
        :param num_heads: 多头的数量
        :param dropout:   一层神经元在每次迭代训练时不参与训练的概率
        """

        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0
        self.d_K = embed_dim // num_heads  # d_K = d_Q
        self.d_V = self.d_K  # 假设d_V = d_K
        self.num_heads = num_heads
        self.dropout = dropout

        # 以Query为例，W_Q为(embed_dim, d_Q=d_K)的矩阵，一共有num_heads个头，
        # 对应num_heads个W_Q矩阵，拼接起来为(embed_dim, d_q*num_heads=embed_dim)的矩阵
        self.W_Q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, embed_dim, bias=False)

        # 根据注意力公式得到的num_heads个(src_len, d_V)矩阵Z_i，并将其拼接起来，得到一个
        # (src_len, d_V*num_heads=embed_dim)的矩阵，再乘以(embed_dim, embed_dim)的矩阵W_O
        # 得到(src_len, embed_dim)的矩阵Z
        self.W_O = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        """
        :param query:     (batch_size, len_Q, embed_dim)的张量
        :param key:       (batch_size, len_K, embed_dim)的张量
        :param value:     (batch_size, len_V, embed_dim)的张量
        :param attn_mask: (batch_size, seq_len, seq_len)的张量
        :return:
        """

        print(query, key, value, attn_mask, key_padding_mask)
