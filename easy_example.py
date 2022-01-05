from Model.transformer import Transformer
import easy_data as data
import torch
import torch.nn as nn


model = Transformer()  # .cuda()
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

criterion = nn.CrossEntropyLoss(ignore_index=0)  # 不计算填充位置的损失
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


for epoch in range(20):
    optimizer.zero_grad()
    outputs = model(data.enc_inputs, data.dec_inputs)
    loss = criterion(outputs, data.target_batch.contiguous().view(-1))
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
    loss.backward()
    optimizer.step()

# Test
predict = model(data.enc_inputs, data.dec_inputs)
predict = predict.data.max(1, keepdim=True)[1]
print(data.sentences[0], '->', [data.number_dict[n.item()] for n in predict.squeeze()])
