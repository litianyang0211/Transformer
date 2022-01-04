import copy
import math
import torch
import torch.nn as nn
import numpy as np


def attention(query, key, value, attn_mask=None, dropout=None):
    """
    :param query:     (batch_size, head_num, q_len, d_q)的张量
    :param key:       (batch_size, head_num, k_len, d_k)的张量
    :param value:     (batch_size, head_num, v_len, d_v)的张量，其中v_len=k_len
    :param attn_mask: (batch_size, head_num, q_len, k_len)的张量
    :param dropout:   nn.Dropout(p)
    :return:          注意力结果，为(batch_size, head_num, q_len, d_v)的张量
    """

    d_k = key.size(-1)  # d_k = d_q
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # -> (batch_size, head_num, q_len, k_len)

    # 对已经计算好的scores，按照mask张量填入-1e9，
    # 这样在下一步在计算softmax时，被设置成-1e9的数对应的值可以被忽略
    if attn_mask is not None:
        scores.masked_fill_(attn_mask, -1e9)  # attn_mask是一个由True和False构成的张量

    # 对scores的最后一个维度执行softmax
    attn = nn.Softmax(dim=-1)(scores)

    # 使用nn.Dropout防止过拟合
    if dropout is not None:
        attn = dropout(attn)

    return torch.matmul(attn, value)  # -> (batch_size, head_num, q_len, d_v)


def attn_subsequence_mask():
    """
    TODO: attn_subsequence_mask function
    """
    return np.zeros(1)


def clones(module, n):
    """
    :param module: 需要复制的结构
    :param n:      复制的个数
    :return:       复制的结构
    """

    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def padding_mask(q, k):
    """
    :param q: (batch_size, seq_len)的张量，seq_len为src_len或tgt_len
    :param k: (batch_size, seq_len)的张量，seq_len为src_len或tgt_len
    :return:  (batch_size, q_len, k_len)的张量，填充部分为True，非填充部分为False
    """

    batch_size, q_len = q.size()
    k_len = k.size(1)

    # .eq(0)判断某个位置的值是否等于0
    # (batch_size, k_len) -.unsqueeze(1)-> (batch_size, 1, k_len) -.expand-> (batch_size, q_len, k_len)
    return k.data.eq(0).unsqueeze(1).expand(batch_size, q_len, k_len)
