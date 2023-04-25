---
layout: default
title: Reading 3
id: reading3
---


# Reading Assignment 3: Approximate Inference & Variational Autoencders

## Preliminaries

You can start by reading [chapter 19, intro and 19.1](https://www.deeplearningbook.org/contents/inference.html)
for some context of what we
understand under "inference", why this is difficult as well as connecting to
the previous topics (at least somewhat). You may want to
check [the definiton of the Kullback-Leibler Divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)
first.

You can also read the remainder of the chapter if you want -- however this is 
**very** hard to follow. Another, somewhat simpler explanation can be found 
[in this short blog post](https://mbernste.github.io/posts/variational_inference/).
However, at the end of the day, variational inference is a highly mathematical topic,
so formulas are hard to avoid. 

**The main point is for you to understand that we will be using a lower bound on
the likelihood, namely the evidence lower bound (ELBO), as a training objective.**

Finally, read [section 20.9](https://www.deeplearningbook.org/contents/generative_models.html)
in the Deep Learning Book on backpropagating through random operations. This is
an important "trick" to successfully use VAEs. You can skip section 20.9.1 on
discrete operations.

## Variational Autoencoders

We offer you a variety of readings on VAEs. These explain the concept in different
levels of detail and from different perspectives. See which ones work for you!
- [Section 20.10.3 in the DL book](https://www.deeplearningbook.org/contents/generative_models.html)
is written in the familiar style and links back to many concepts of chapter 19 (which
you may not have read).
- [This blog post](https://jaan.io/what-is-variational-autoencoder-vae-tutorial/)
first introduces VAEs from the neural network perspective, which should be more familiar
to you, and then introduces the graphical models perspective. **If you only read
one of these links, make it this one.**
- [This blog](https://kvfrans.com/variational-autoencoders-explained/) gives a rather
simplified explanation. This can be good as a first piece of context, but you
should aim for a deeper understanding in this class.
- A longer, more in-depth treatment can be found [here](https://arxiv.org/pdf/1606.05908.pdf).
- Finally, you can read the [original paper](https://arxiv.org/pdf/1312.6114.pdf),
if you want.
  - [This paper](https://arxiv.org/pdf/1401.4082.pdf) is often credited as essentially
  an independent "parallel discovery".
