from Model.transformer import Transformer
import easy_data as data
import torch
import torch.nn as nn


model = Transformer()  # .cuda()
# for p in model.parameters():
#     if p.dim() > 1:
#         nn.init.xavier_uniform_(p)

criterion = nn.CrossEntropyLoss(ignore_index=0)  # 不计算填充位置的损失
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)

for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(data.enc_inputs, data.dec_inputs)
    loss = criterion(outputs, data.dec_outputs.contiguous().view(-1))
    print("Epoch:", "%03d" % (epoch + 1), "cost =", "{:.5f}".format(loss))
    loss.backward()
    optimizer.step()

# Test
