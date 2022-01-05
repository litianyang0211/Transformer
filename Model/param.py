import easy_data


src_vocab_size = easy_data.src_vocab_size
tgt_vocab_size = easy_data.tgt_vocab_size
embed_dim = 512
num_heads = 8
dropout = 0.1
max_len = 5000
d_k = d_q = embed_dim // num_heads
d_v = d_k
d_ff = 2048
num_layers = 6
