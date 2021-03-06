from Model.decoder_layer import DecoderLayer
from Model.embeddings import Embeddings
from Model.positional_encoding import PositionalEncoding
from Model.utils import attention_mask, clones, padding_mask
import Model.param as param
import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = Embeddings(param.tgt_vocab_size, param.embed_dim)
        self.pos_enc = PositionalEncoding(param.embed_dim)
        self.layers = clones(DecoderLayer(), param.num_layers)

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        """
        :param dec_inputs:  形状为(batch_size, tgt_len)的torch.LongTensor
        :param enc_inputs:  (batch_size, src_len)的张量
        :param enc_outputs: (batch_size, src_len, embed_dim)的张量
        :return:            (batch_size, tgt_len, embed_dim)的张量
        """

        dec_outputs = self.pos_enc(self.tgt_emb(dec_inputs))
        dec_pad_mask = padding_mask(dec_inputs, dec_inputs)  # .cuda()
        dec_attn_mask = attention_mask(dec_inputs, dec_inputs)  # .cuda()
        dec_mask = torch.gt(dec_pad_mask + dec_attn_mask, 0)  # .cuda()

        dec_enc_pad_mask = padding_mask(dec_inputs, enc_inputs)  # .cuda()
        dec_enc_attn_mask = attention_mask(dec_inputs, enc_inputs)  # .cuda()
        dec_enc_mask = torch.gt(dec_enc_pad_mask + dec_enc_attn_mask, 0)  # .cuda()

        for layer in self.layers:
            dec_outputs = layer(dec_outputs, enc_outputs, dec_mask, dec_enc_mask)

        return dec_outputs
