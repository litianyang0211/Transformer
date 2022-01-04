from Model.utils import attention, clones
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        :param embed_dim: 词嵌入的维度
        :param num_heads: 多头的数量
        :param dropout:   在每次迭代训练时不参与训练的概率
        """

        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0
        self.d_k = embed_dim // num_heads  # d_k = d_q
        self.d_v = self.d_k  # 假设d_v = d_k
        self.num_heads = num_heads
        self.dropout = nn.Dropout(p=dropout)

        # 以query为例，W_Q为(embed_dim, d_q=d_k)的矩阵，一共有num_heads个头，
        # 对应num_heads个W_Q矩阵，拼接起来为(embed_dim, d_q*num_heads=embed_dim)的矩阵
        # 又例如，根据注意力公式得到的num_heads个(seq_len, d_v)矩阵Z_i，并将其拼接起来，得到一个
        # (seq_len, d_v*num_heads=embed_dim)的矩阵，再乘以(embed_dim, embed_dim)的矩阵W_O
        # 得到(seq_len, embed_dim)的矩阵Z
        self.linear_lst = clones(nn.Linear(embed_dim, embed_dim, bias=False), 4)

    def forward(self, q_inputs, k_inputs, v_inputs, attn_mask=None):
        """
        :param q_inputs:   (batch_size, q_len, embed_dim)的张量
        :param k_inputs:   (batch_size, k_len, embed_dim)的张量
        :param v_inputs:   (batch_size, v_len, embed_dim)的张量，其中v_len=k_len
        :param attn_mask:  (batch_size, q_len, k_len)的张量
        :return:           (batch_size, q_len, embed_dim)的张量
        """

        # 生成用于attention函数的mask张量
        if attn_mask is not None:
            # (batch_size, q_len, k_len) -.unsqueeze-> (batch_size, 1, q_len, k_len)
            # -.repeat-> (batch_size, num_heads, q_len, k_len)
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        residual = q_inputs  # 用于残差连接，为(batch_size, q_len, embed_dim)的张量
        batch_size = q_inputs.size(0)

        # 利用线性变换计算Q, K, V矩阵，并把embed_dim平均分配给head_num个头
        # 以query为例，(batch_size, q_len, embed_dim) -W(X)-> (batch_size, q_len, embed_dim)
        # -.view-> (batch_size, q_len, num_heads, d_q=d_k) -.transpose-> (batch_size, num_heads, q_len, d_q=d_k)
        # 同理，k和v分别为(batch_size, num_heads, k_len, d_k)和(batch_size, num_heads, v_len, d_v=d_k)的张量
        q, k, v = [W(X).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
                   for W, X in zip(self.linear_lst, (q_inputs, k_inputs, v_inputs))]

        # 计算Attention
        attn = attention(q, k, v, attn_mask, self.dropout)

        # 利用线性变换将head_num个头的d_v维向量拼接成head_num*d_v维的向量
        # (batch_size, head_num, q_len, d_v) -.transpose-> (batch_size, q_len, head_num, d_v)
        # -.view-> (batch_size, q_len, embed_dim)
        z = self.linear_lst[-1](attn.transpose(1, 2).contiguous().view(batch_size, -1, self.d_v*self.num_heads))

        return nn.LayerNorm(self.d_v*self.num_heads)(z + residual)
        # return nn.LayerNorm(self.d_v*self.num_heads).cuda()(z + residual)
