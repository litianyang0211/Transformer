import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        """
        :param embed_dim: 词嵌入的维度
        :param dropout:   在每次迭代训练时不参与训练的概率
        :param max_len:   提前准备好的序列的位置编码的长度
        """

        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, embed_dim)  # 得到(max_len, embed_dim)的矩阵来表示其位置编码
        position = torch.arange(0, max_len).unsqueeze(1)  # 得到(max_len, 1)的矩阵
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim))  # 为了防止数值过大溢出
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数下标的位置
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数下标的位置
        pe = pe.unsqueeze(0)  # (max_len, embed_dim) -> (batch_size=1, max_len, embed_dim)
        self.register_buffer("pe", pe)

    def forward(self, inputs):
        """
        :param inputs: Embeddings的词嵌入结果，为(batch_size, seq_len, embed_dim)的张量，seq_len为src_len或tgt_len
        :return:       词嵌入加上位置编码的结果，为(batch_size, seq_len, embed_dim)的张量，seq_len为src_len或tgt_len
        """

        return self.dropout(inputs + self.pe[:, :inputs.size(1), :])
