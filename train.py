from Model.transformer import Transformer
import torch.nn as nn
import torch.optim as optim

model = Transformer()  # .cuda()
loss = nn.CrossEntropyLoss(ignore_index=0)  # 不计算填充位置的损失
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)
