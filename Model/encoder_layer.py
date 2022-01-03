import torch.nn as nn
from Model.feed_forward import PositionwiseFeedForward
from Model.multi_head_attn import MultiHeadAttention


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_attn = MultiHeadAttention()
        self.pff = PositionwiseFeedForward()

    def forward(self, enc_input, enc_attn_mask):
        """
        :param enc_input:     (batch_size, src_len, embed_dim)的张量
        :param enc_attn_mask: (batch_size, src_len, src_len)的张量
        :return:              (batch_size, src_len, embed_dim)的张量
        """

        return self.pff(self.enc_attn(enc_input, enc_input, enc_input, enc_attn_mask))
