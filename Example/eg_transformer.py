from Model.transformer import Transformer
import torch


if __name__ == '__main__':
    src = torch.randn(2, 5).long()
    tgt = torch.randn(2, 6).long()
    model = Transformer()
    out = model(src, tgt)
    print(out.shape)