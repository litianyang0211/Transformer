from embeddings import Embeddings
from positional_encoding import PositionalEncoding
import torch


if __name__ == "__main__":
    x = torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]], dtype=torch.long)
    embedding = Embeddings(vocab_size=11, embed_dim=512)
    x = embedding(x=x)
    pos_encoding = PositionalEncoding(embed_dim=512)
    x = pos_encoding(x=x)
    print(x.shape)  # torch.Size([2, 4, 512])
