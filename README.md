# Transformer with Manual KV Cache

An educational implementation of a Decoder-only Transformer in PyTorch, based on Andrej Karpathy's [Let's Build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) tutorial.
This code extends the baseline implementation by adding a Stateful Key-Value (KV) Cache from scratch.

## Implementations
While the core architecture follows the tutorial, this repo adds:

* **Manual KV Caching:** Implements explicit storage for Keys and Values within the Attention blocks.
* **Chunked Prefill Masking:** Adds a Rectangular Causal Mask" to handle `dL > 1` (chunked inputs), ensuring causality is preserved even when processing multiple prompt tokens at once.
* **Stateful Layer:** Encapsulates cache state within the `MaskedSelfAttention`, 'MLP', 'MultiHead' module, managing the slicing and concatenation logic internally.


