import numpy as np
import torch.nn as nn


class Embeddings(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        """
        :param vocab_size: 当前语言的词典大小（单词个数），为src_vocab_size或tgt_vocab_size
        :param embed_dim:  词嵌入的维度
        """

        super(Embeddings, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, inputs):
        """
        :param inputs: 形状为(batch_size, seq_len)的torch.LongTensor，seq_len为src_len或tgt_len
        :return:       (batch_size, seq_len, embed_dim)的张量，seq_len为src_len或tgt_len
        """

        # nn.Embedding在初始化时，用的是xavier_uniform，而乘法运算是为了让最后分布的方差为1，
        # 使网络在训练时的收敛速度更快
        return self.embed(inputs) * np.sqrt(self.embed_dim)
