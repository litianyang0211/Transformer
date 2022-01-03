import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, embed_dim, d_ff, dropout=0.1):
        """
        :param embed_dim: 词嵌入的维度
        :param d_ff:      中间隐单元的个数
        :param dropout:   在每次迭代训练时不参与训练的概率
        """

        super(PositionwiseFeedForward, self).__init__()
        self.embed_dim = embed_dim
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, embed_dim)
        )

    def forward(self, x):
        """
        :param x: (batch_size, q_len, embed_dim)的张量
        :return:  (batch_size, q_len, embed_dim)的张量
        """

        return nn.LayerNorm(self.embed_dim)(self.fc(x) + x)
        # return nn.LayerNorm(self.embed_dim).cuda()(self.fc(x) + x)
