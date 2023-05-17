---
layout: default
title: Reading 6
id: reading6
---


# Reading Assignment 6: Autoregressive Models

[Chapter 21 from Probabilistic Machine Learning: Advanced 
Methods](https://probml.github.io/pml-book/book2.html) gives a good overview of
autoregressive models. Since these models are conceptually very straightforward,
there is not too much deep theory here.

Next, [The unreasonable effectiveness of RNNs](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
by Andrej Karpathy covers the basics of language modeling. Although outdated,
state-of-the-art approaches like GPT work in essentially the same way.

## Classic Success Stories

Consider these optional reading in case you want to get some more concrete examples.
Some of these may be treated in more detail later in the class.

- [PixelRNN](https://arxiv.org/pdf/1601.06759.pdf) models images directly on the 
pixel level. As the CNN variant was used a lot more, you can skip the details of
the RNNs. There is also a [follow-up](https://arxiv.org/pdf/1606.05328.pdf), and
[another one](https://arxiv.org/pdf/1701.05517.pdf).
- [Wavenet](https://www.deepmind.com/blog/wavenet-a-generative-model-for-raw-audio)
was a revolutionary model that generated audio in an autoregressive manner. It's
basically PixelCNN in 1D.
- [VQ-VAE](https://arxiv.org/pdf/1711.00937.pdf) encodes data in a lower-dimensional
space and then uses an autoregressive model on that space. There is also 
[a follow-up](https://arxiv.org/pdf/1906.00446.pdf) which uses a multi-level
approach to generate very high-quality images.
