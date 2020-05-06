---
toc: true
layout: post
description: Transformers for loooong documents
author: Shubham Gupta
categories: [nlp, transformer, review, longformer]
image: images/longformer/training.png
title: LongFormer
---

# Introduction

- The NLP world had it's ImageNet moment with the introduction of the Transformer in the paper **Attention is All you Need**. 
- The ability to be able to process multiple words/tokens in parallel and train models without labelled data(using self attention) led to creation of multiple models which gave us SOTA results on many interesting tasks such as Question Answering, Summarization, etc. -
- However, the biggest drawback is the Transformer architecture is the limitation it has on the number of tokens it can process at a once, due to exponentially increasing memory and compute requirements(typically about 512 tokens), causing the performance to deterioate over large documents. 
- [Longformer](https://arxiv.org/abs/2004.05150) by the team at Allen AI aims to address this problem and demonstrate it's application to do transfer learning for large documents.
- Other approaches to are described in recent work such as [Transformer XL](link), [Blockwise](link), [Reformer](link), etc. Their characteristics are mentioned below:

![Comparison[]{data-label="fig:overview"}]({{site.baseurl}}/images/longformer/comparison.png)

# Key Contributions

- Transformers are expensive because of the massive matrix operations involved in the self-attention step. Since each token can attend to every other token in the given input, we get a runtime of $O(n^2)$, where $n$ is the sequence length(typically 512 tokens).
- LongFormer aims to solve this using a form of sparse attention and reducing the operation complexity to $O(n)$. They acheive this using the concept of sliding window and dilated sliding window. 
- The authors also show how this attention pattern can be modified (using dilation and global attention) on a per-task basis, thereby allowing us to use a single model for all tasks rather than creating task-specfic architectures.

# Attention Patterns
- The attention patterns implemented are as follows:
![Attention[]{data-label="fig:overview"}]({{site.baseurl}}/images/longformer/attention.png)

## Sliding Window Attention

- **TLDR** : Similar to kernels for CNN which apply a matrix operation to a set of pixels and move onto the next set, apply attention to tokens in current window _only_.
- In this, we change the attention objective to only focus on the tokens that occur in a context window $w$. 
- Each token will be able to attend to $\frac{1}{2}w$ number of tokens to it's left and right.


# Notes 

- Longformer is a paper by Allen AI which was referenced in DAIR's nlp newsletter(available [here](https://dair.ai/NLP_Newsletter_10_en/))
- It aims to solve the limitation of the number of tokens that can be processed simultaneously in the transformer architecture.
- Use CNN principle. Use kernel and sliding technique to reduce the memory comsumption. Memory reduced to _d*k_
- Longformer does for transformer what CNN does for MLP.
- **Dilated Sliding Window**: Sliding window might take lot of layers to accomodate all information. TLDR: Dilated sliding window is a skip one toke type appraoch. Instead of attending all nearby tokens, it will attend to alternate tokens. LARGER window size => Faster information aggregation
- Use combo of both sliding window and dilated sliding window attention
  - In lower layers, use sliding window. Reasoning is that in the start local context is more important.
  - In higher layers, use dilated window. Reasoning is that in the end global context is more important.

- Global attention: Sparse. Special units. Basically, they can attend to anything, similar to classic self-attention in transformer. Reasoning: Sometimes needed. Engineering choice. Eg: In question answering task for yes/no, use the [CLS] token for classification. Hence, all [CLS] tokens will be special units with full self-attention

- They can copy output from roberta because they use window size of 512. Such a clever hack!! This helps them reduce training time as well.
