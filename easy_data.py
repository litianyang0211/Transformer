import torch


# S: Symbol that shows starting of decoding input
# E: Symbol that shows starting of decoding output
# P: Symbol that will fill in blank sequence if current batch data size is shorter than time steps
sentences = ["ich mochte ein bier P", "S i want a beer", "i want a beer E"]

src_vocab = {"P": 0, "ich": 1, "mochte": 2, "ein": 3, "bier": 4}
src_vocab_size = len(src_vocab)

tgt_vocab = {"P": 0, "i": 1, "want": 2, "a": 3, "beer": 4, "S": 5, "E": 6}
number_dict = {i: w for i, w in enumerate(tgt_vocab)}
tgt_vocab_size = len(tgt_vocab)

# Make batch
enc_inputs = torch.LongTensor([[src_vocab[n] for n in sentences[0].split()]])
dec_inputs = torch.LongTensor([[tgt_vocab[n] for n in sentences[1].split()]])
dec_outputs = torch.LongTensor([[tgt_vocab[n] for n in sentences[2].split()]])
