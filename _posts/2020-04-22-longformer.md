---
toc: true
layout: post
description: Make transformers more efficient
author: Shubham Gupta
categories: [nlp, transformer, review, longformer]
image: images/longformer/training.png
title: 'LongFormer'
---

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
