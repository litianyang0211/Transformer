import param
import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    def __init__(self):
        super(PositionwiseFeedForward, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(param.embed_dim, param.d_ff),
            nn.ReLU(),
            nn.Dropout(param.dropout),
            nn.Linear(param.d_ff, param.embed_dim)
        )

    def forward(self, inputs):
        """
        :param inputs: (batch_size, q_len, embed_dim)的张量
        :return:       (batch_size, q_len, embed_dim)的张量
        """

        return nn.LayerNorm(param.embed_dim)(self.ffn(inputs) + inputs)
        # return nn.LayerNorm(param.embed_dim).cuda()(self.ffn(inputs) + inputs)
