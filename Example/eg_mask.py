from Model.utils import attention_mask, padding_mask
import torch


if __name__ == "__main__":
    # 假设最后两列是padding mask, 则pad_mask输出一个最后两列全为True(=1)，而其余位置为False(=0)的矩阵
    dec_inputs = torch.tensor([[0.7, 0.2, 0., 0.],
                               [0.3, 0.5, 0., 0.],
                               [0.2, 1.4, 0., 0.],
                               [0.8, 0.1, 0., 0.]])
    pad_mask = padding_mask(dec_inputs, dec_inputs)

    # attn_mask是一个上三角矩阵，对角线及以下位置为0(=False)，其余位置为1(=True)
    attn_mask = attention_mask(dec_inputs)

    # 当pad_mask和attn_mask相加时，需要mask的位置必然大于等于1，而其它位置为0，利用.gt可以生成只有True和False的矩阵，
    # 其中，需要mask的位置为True，不需要mask的位置为False
    test = torch.gt(pad_mask+attn_mask, 0)
    print(test)
