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

- The NLP world had its ImageNet moment with the introduction of the Transformer in the paper **Attention is All you Need**. 
- The ability to be able to process multiple words/tokens in parallel and train models without labeled data(using self-attention) led to the creation of multiple models which gave us SOTA results on many interesting tasks such as Question Answering, Summarization, etc. -
- However, the biggest drawback is the Transformer architecture is the limitation it has on the number of tokens it can process at a once, due to exponentially increasing memory and compute requirements(typically about 512 tokens), causing the performance to deteriorate over large documents. 
- [Longformer](https://arxiv.org/abs/2004.05150) by the team at Allen AI aims to address this problem and demonstrate it's application to do transfer learning for large documents.
- Other approaches to are described in recent work such as [Transformer XL](link), [Blockwise](link), [Reformer](link), etc. Their characteristics are mentioned below:

![Comparison[]{data-label="fig:overview"}]({{site.baseurl}}/images/longformer/comparison.png)

# Key Contributions

- Transformers are expensive because of the massive matrix operations involved in the self-attention step. Since each token can attend to every other token in the given input, we get a runtime of $O(n^2)$, where $n$ is the sequence length(typically 512 tokens).
- LongFormer aims to solve this using a form of sparse attention and reducing the operational complexity to $O(n)$. They achieve this using the concept of the sliding window and dilated sliding window. 
- The authors also show how this attention pattern can be modified (using dilation and global attention) on a per-task basis, thereby allowing us to use a single model for all tasks rather than creating task-specific architectures.

# Attention Patterns
- The attention patterns implemented are as follows:
![Attention[]{data-label="fig:overview"}]({{site.baseurl}}/images/longformer/attention.png)

## Sliding Window Attention

- **TLDR** : Similar to kernels for CNN which apply a matrix operation to a set of pixels and move onto the next set, apply attention to tokens in current window _only_.
- In this, we change the attention objective to only focus on the tokens that occur in a context window $w$. 
- Each token will be able to attend to $\frac{1}{2}w$ number of tokens to it's left and right.
- **Question**: But doesn't this limit the number of tokens being taken into account to only the tokens in the window?
  - Yes, it does. This is why we stack multiple layers of self-attention. As shown in the image below, the green neuron learns from the first 3 tokens(Lionel, Messi, is). However, the brown neuron learns from the green, yellow, and red neuron, who together learn from the first 5 tokens. This way, we can apply attention to long sequences(Lionel, Messi, is, the, true).
- As with the CNN, we will have $l$ layers to this sliding window attention(multi-head attention) implemented to learn low level and high-level features. A balance should be found between the number of layers $l$(efficiency) and the window size $w$(model representation capacity). 

![Sliding Window Attention[]{data-label="fig:overview"}]({{site.baseurl}}/images/longformer/sliding_window.png)

- **Pros**: Reduces computation from $O(n^2)$ to $O(n*w)$ i.e the computation complexity will only scale linearly now.

- **Cons**: To learn dependencies for a large sequence, we would either have to increase the window size $w$ or increase the number of layers $l$, both of which will cause an increase in the amount of memory and processing power required to train and test the model.

## Dilated Sliding Window
- **TLDR**: Use dilation instead of window attention i.e for some particular window size, take alternate elements while performing self-attention.
- To solve the problem for long sequences, the authors propose that instead of considering all tokens in window $w$, consider alternate(or any number $d$)tokens instead. The range of tokens will now be $l * d * w$, which will be large for even a small value of $d$.

- **Pros**: This small change will allow us to cover a wider range of tokens without significant changes to the architecture.
- **Cons**: Skipping tokens might lead to loss of information in the lower layers which will get propagated to the higher layers. This will lead to unstable training and poor model performance.

## Global Attention
- **TLDR**: Use full attention for certain tokens depending on the task. This is an engineering choice.
- In BERT style models, optimal representation for input sequence varies by task.
  - For MLM, local context is used to predict the masked word
  - For classification, [CLS] token is used.
  - For QnA, question is concated with document to help model learn through self attention.
- The windowed and dilated attention are not flexible enough to learn task specific representations.
- Hence, for some tokens enable global tokens i.e at these tokens, all tokens in the sequence can attend to it. For classifcation, enable global attention on the [CLS] token.
- **Pros**: 
  - Adding global attention improves performance for specific tasks. Since these tokens are limited in number, the complexity still stays at $O(n)$. 
  - It also increases representational power of the model.

### Linear Projections

- **TLDR**: Use two sets of Q,K and V matrices, one for sliding window attention, one for global attention.
- Attention is defined as:

  $$
  \begin{aligned}
  Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
  \end{aligned}
  $$
- We will use two different sets of Q,K and V matrices for sliding window and global attention. 
- $Q_g$, $K_g$, $V_g$ are initialized with $Q_s$, $K_s$, $V_s$

![Banded Matrix]({{site.baseurl}}/images/longformer/old_matrix.svg)
<center><b>Banded Matrix(<a href="https://en.wikipedia.org/wiki/Band_matrix">Source</a>)</b></center>
![Compressed Banded Matrix]({{site.baseurl}}/images/longformer/band_matrix.svg)
<center><b>Compressed Banded Matrix(<a href="https://en.wikipedia.org/wiki/Band_matrix">Source</a>)</b></center>

### CUDA Kernels
- One of the important and interesting contributions of this paper is the implementation of matrix multiplication via CUDA kernels.
- In dilated sliding window, the matrix formed is called a **band matrix** i.e there are diagonal bands of indices that have values and the other values are 0.
- Implementing matrix operations for band matrices using native for loops and via frameworks is not easy and optimized.
- The authors have provided custom CUDA kernels implemented using [TVM](https://github.com/apache/incubator-tvm) for this banded matrix operations.
- As demonstrated in the image below, the custom CUDA kernels have a significant impact on the time and memory consumption of the model. The kernels and implementation for the longformer is available [here](https://github.com/allenai/longformer).
![Performance]({{site.baseurl}}/images/longformer/performance.png)
<center><b>LongFormer Performance</b></center>

# Autoregressive Language Modelling

- Estimate the probability of a token given it's previous tokens/characters in a input sequence.
- It is a fundamental task in natural language and all prevous work use this task as their primary evaluation measure.

## Attention Pattern
- In multi-head attention, each head computes a different score.
- To get a good representation of all tokens, the authors propose that normal sliding window attention can be used for the lower layers, and dilated sliding window attention can be used the higher layers(top 1-2 layers).
- The reasoning for this approach is that in the lower layers, the local context is more important, and in the upper layers, the global context is more important. Hence, it is acceptable to skip over a few tokens in the upper layers.

# Experimental Setup

## Task and Datasets
- The authors focus on character level modelling because the sequences are naturally longer than those of word level language modelling.
- Datasets that were used are _text8_ and _enwik8_.

## Training and Evaluation
- The model was trained in multiple phases.
  - The window and sequence length was increased in each phase. This is to allow local context from tokens to be learnt efficiently.
  - Overall five training phases used, starting from token length of 2048 to 23040 (45x more than vanilla BERT).
  - Two models were created for evaluation:
    - Small model: 12 layers, 512 hidden size 
    - Large model: 30 layers, 512 hidden size (2.5x larger)
  - During model evaluation, the model is able to run on a sequence length of 32256(63x more than vanilla BERT).

## Results
![Results]({{site.baseurl}}/images/longformer/results.png)
- Longformer acheives SOTA using the small models with BPC of 1.10 and 1.00 for text8 and enwik8.
- The large model was only tested on enwik8 due to the computational cost of training.
- It's also important to note that, while the large model did not acheive SOTA, it performs much better that it's counterparts who have almost 2x more parameters.

# Pretraining and Finetuning
- The LongFormer is trained to solve the tasks of classification, QA and coreference resolution.
- It is trained with MLM objective.

## Copy trick
- Since the MLM objective pretraining objective is expensive, the authors continue to train from the checkpoints of the [RoBERTA](https://arxiv.org/abs/1907.11692) model.
- The attention mechanism is replaced with the new attention module.
- 

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

