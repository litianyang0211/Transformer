# Introduction

本项目使用**PyTorch**搭建Transformer模型，并给出尽可能详细的代码注释。在阅读代码之前，请先熟读经典论文[Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)以及[关于Transformer的解释](https://l-ty.com/Computer_Science/Natural_Language_Processing/Transformer)。

关于代码的细节，也可以参考[Transformer in PyTorch](https://l-ty.com/Computer_Science/PyTorch/Transformer_in_PyTorch)。

# Model

* `param.py`设置了相关参数。

* `utils.py`包括四个帮助函数：自注意力的计算、掩码的生成、克隆函数。

* `embeddings.py`实现了词嵌入。

* `positional_encoding.py`实现了添加位置编码。

* `multi_head_attn.py`是模型中最核心的部分，实现了多头注意力机制。

* `pos_feed_forward.py`实现了前馈网络。

* `encoder_layer.py`与`decoder_layer.py`分别实现了编码器和解码器，包括了多头注意力机制和前馈网络。

* `encoder.py`与`decoder.py`实现了编码器和解码器的堆栈，以及编码器和解码器之间的互动。

* `transformer.py`实现了整一个Transformer模型。

# Usage

* 运行`easy_example.py`即可训练一个简易的德英翻译模型。

# TODO

* 简易例子的测试

* 复杂例子