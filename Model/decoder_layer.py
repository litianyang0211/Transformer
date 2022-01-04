from Model.pos_feed_forward import PositionwiseFeedForward
from Model.multi_head_attn import MultiHeadAttention
import Model.param as param
import torch.nn as nn


class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_attn = MultiHeadAttention(param.embed_dim, param.num_heads)
        self.dec_enc_attn = MultiHeadAttention(param.embed_dim, param.num_heads)
        self.pff = PositionwiseFeedForward(param.embed_dim, param.d_ff)

    def forward(self, dec_inputs, enc_outputs, dec_mask, dec_enc_mask):
        """
        :param dec_inputs:   (batch_size, tgt_len, embed_dim)的张量
        :param enc_outputs:  (batch_size, src_len, embed_dim)的张量
        :param dec_mask:     (batch_size, tgt_len, tgt_len)的张量
        :param dec_enc_mask: (batch_size, tgt_len, src_len)的张量
        :return:             (batch_size, tgt_len, embed_dim)的张量
        """

        dec_outputs = self.dec_attn(dec_inputs, dec_inputs, dec_inputs, dec_mask)
        dec_outputs = self.dec_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_mask)
        return self.pff(dec_outputs)
