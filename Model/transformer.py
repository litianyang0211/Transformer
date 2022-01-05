from Model.decoder import Decoder
from Model.encoder import Encoder
import Model.param as param
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()  # .cuda()
        self.decoder = Decoder()  # .cuda()
        self.linear = nn.Linear(param.embed_dim, tgt_vocab_size, bias=False)  # .cuda() TODO: tgt_vocab_size未定义，来自数据集

    def forward(self, enc_inputs, dec_inputs):
        """
        :param enc_inputs: (batch_size, src_len)的张量
        :param dec_inputs: (batch_size, tgt_len)的张量
        :return:           (batch_size*tgt_len, tgt_vocab_size)的张量
        """

        enc_outputs = self.encoder(enc_inputs)  # (batch_size, src_len) -> (batch_size, src_len, embed_dim)
        dec_outputs = self.decoder(dec_inputs, enc_inputs, enc_outputs)  # -> (batch_size, tgt_len, embed_dim)

        # (batch_size, tgt_len, embed_dim) -.Linear-> (batch_size, tgt_len, tgt_vocab_size)
        dec_logits = self.linear(dec_outputs)

        # 利用.view将不同的句子合并为长句
        # (batch_size, tgt_len, tgt_vocab_size) -.view-> (batch_size*tgt_len, tgt_vocab_size)
        return dec_logits.view(-1, dec_logits.size(-1))
