from Model.embeddings import Embeddings
from Model.encoder_layer import EncoderLayer
from Model.positional_encoding import PositionalEncoding
from Model.utils import clones, padding_mask
import Model.param as param
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = Embeddings(param.src_vocab_size, param.embed_dim)
        self.pos_enc = PositionalEncoding(param.embed_dim)
        self.layers = clones(EncoderLayer(), param.num_layers)

    def forward(self, enc_inputs):
        """
        :param enc_inputs: 形状为(batch_size, src_len)的torch.LongTensor
        :return:           (batch_size, src_len, embed_dim)的张量
        """

        enc_outputs = self.pos_enc(self.src_emb(enc_inputs))
        enc_mask = padding_mask(enc_inputs, enc_inputs)

        for layer in self.layers:
            enc_outputs = layer(enc_outputs, enc_mask)

        return enc_outputs
