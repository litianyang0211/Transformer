from Model import param
from Model.utils import attention
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.dropout = nn.Dropout(p=param.dropout)

        # 以query为例，W_Q为(embed_dim, d_q=d_k)的矩阵，一共有num_heads个头，
        # 对应num_heads个W_Q矩阵，拼接起来为(embed_dim, d_q*num_heads=embed_dim)的矩阵
        # 又例如，根据注意力公式得到的num_heads个(seq_len, d_v)矩阵Z_i，并将其拼接起来，得到一个
        # (seq_len, d_v*num_heads=embed_dim)的矩阵，再乘以(embed_dim, embed_dim)的矩阵W_O
        # 得到(seq_len, embed_dim)的矩阵Z
        self.q_weight = nn.Linear(param.embed_dim, param.d_q * param.num_heads, bias=False)
        self.k_weight = nn.Linear(param.embed_dim, param.d_k * param.num_heads, bias=False)
        self.v_weight = nn.Linear(param.embed_dim, param.d_v * param.num_heads, bias=False)
        self.o_weight = nn.Linear(param.d_v * param.num_heads, param.embed_dim, bias=False)

    def forward(self, q_inputs, k_inputs, v_inputs, attn_mask=None):
        """
        :param q_inputs:   (batch_size, q_len, embed_dim)的张量
        :param k_inputs:   (batch_size, k_len, embed_dim)的张量
        :param v_inputs:   (batch_size, v_len, embed_dim)的张量，其中v_len=k_len
        :param attn_mask:  (batch_size, q_len, k_len)的张量
        :return:           (batch_size, q_len, embed_dim)的张量
        """

        residual = q_inputs  # 用于残差连接，为(batch_size, q_len, embed_dim)的张量
        batch_size = q_inputs.size(0)

        # 生成用于attention函数的mask张量
        if attn_mask is not None:
            # (batch_size, q_len, k_len) -.unsqueeze-> (batch_size, 1, q_len, k_len)
            # -.repeat-> (batch_size, num_heads, q_len, k_len)
            attn_mask = attn_mask.unsqueeze(1).repeat(1, param.num_heads, 1, 1)

        # 利用线性变换计算Q, K, V矩阵，并把embed_dim平均分配给head_num个头
        # 以query为例，(batch_size, q_len, embed_dim) -.Linear-> (batch_size, q_len, embed_dim)
        # -.view-> (batch_size, q_len, num_heads, d_q) -.transpose-> (batch_size, num_heads, q_len, d_q)
        # 同理，k和v分别为(batch_size, num_heads, k_len, d_k)和(batch_size, num_heads, v_len, d_v)的张量
        q = self.q_weight(q_inputs).view(batch_size, -1, param.num_heads, param.d_q).transpose(1, 2)
        k = self.k_weight(k_inputs).view(batch_size, -1, param.num_heads, param.d_k).transpose(1, 2)
        v = self.v_weight(v_inputs).view(batch_size, -1, param.num_heads, param.d_v).transpose(1, 2)

        # 计算Attention
        attn = attention(q, k, v, attn_mask, self.dropout)

        # 利用线性变换将head_num个头的d_v维向量拼接成head_num*d_v维的向量
        # (batch_size, head_num, q_len, d_v) -.transpose-> (batch_size, q_len, head_num, d_v)
        # -.view-> (batch_size, q_len, embed_dim)
        z = self.o_weight(attn.transpose(1, 2).contiguous().view(batch_size, -1, param.d_v * param.num_heads))

        return nn.LayerNorm(param.d_v * param.num_heads)(z + residual)
        # return nn.LayerNorm(param.d_v * param.num_heads).cuda()(z + residual)
