import torch.nn as nn
import math


class Embeddings(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        """
        :param vocab_size: 当前语言的词典大小（单词个数）
        :param embed_dim:  词嵌入的维度
        """

        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab_size, embed_dim)  # lut: lookup table
        self.embed_dim = embed_dim

    def forward(self, inputs):
        """
        :param inputs: 形状为(batch_size, seq_len)的torch.LongTensor，seq_len为src_len或tgt_len
        :return:       (batch_size, seq_len, embed_dim)的张量，seq_len为src_len或tgt_len
        """

        # nn.Embedding在初始化时，用的是xavier_uniform，而乘法运算是为了让最后分布的方差为1，
        # 使网络在训练时的收敛速度更快
        return self.lut(inputs) * math.sqrt(self.embed_dim)
