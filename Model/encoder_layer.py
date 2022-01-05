from Model.pos_feed_forward import PositionwiseFeedForward
from Model.multi_head_attn import MultiHeadAttention
import torch.nn as nn


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_attn = MultiHeadAttention()
        self.pff = PositionwiseFeedForward()

    def forward(self, enc_inputs, enc_mask):
        """
        :param enc_inputs:    (batch_size, src_len, embed_dim)的张量
        :param enc_mask:      (batch_size, src_len, src_len)的张量
        :return:              (batch_size, src_len, embed_dim)的张量
        """

        return self.pff(self.enc_attn(enc_inputs, enc_inputs, enc_inputs, enc_mask))
