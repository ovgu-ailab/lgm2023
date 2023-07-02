---
layout: default
title: Assignment 11
id: ass11
---


# OPTIONAL Assignment 11: VQ-VAE
**Discussion: July 6th**  
**Deadline: July 6th, 9am**

While not as simple as diffusion models, and a bit cumbersome to set up, VQ-VAEs
and related discrete AE variants are still components of several state-of-the-art
generative models. As such, seeing how they are implemented can be useful.

It is essentially a two-step process:
1. Set up a VAE and implement a codebook lookup/vector quantization operation. 
Train the VAE.
2. Create a new dataset of discrete codes and train an autoregressive model.

Generation proceeds by generating a "code image" using the autoregressive model,
and then decoding back to the data space using the trained VAE.

The following describes the components in more detail. A starter notebook can be
found on Gitlab.


## Discrete VAE

Starting off with an autoencoder should be no issue for you at this point. It
should be a fully-convolutional VAE, with no flatten/dense layers. You likely want
to not downsample too much, e.g. not below 8x8 or something like that (depends on
the input size).

Next, add vector quantization at the end of the encoder. Your model needs to have
a trainable `codebook` variable, of dimensions `K x d` where `d` is the dimensionality
of the encodings (number of channels), and `K` is the codebook size, a hyperparameter.
Perhaps try something on the order of a few hundred.

The starter notebook has code to do the vector quantization: In the model call,
the encoder outputs should be replaced by the closest codebook entries. The
quantized encodings then go into the decoder for reconstruction.

For training, the _straight-through estimator_ is used: As the quantization operation
is not differentiable, the decoder gradients are passed directly to the encoder.
Training the codebook can be done via a codebook loss that moves codebook entries
towards the corresponding encoder outputs. Conversely, a _commitment loss_ moves
the encoder outputs towards the codebook entries. See the notebook for both.
Note that, despite being a VAE, the KL-divergence is actually a constant, so it
drops out of the loss (there are VQ-VAE variants where this is not the case).


## Encoding the Dataset

The above should be enough to train the VAE itself. Use the trained encoder +
codebook to encode and discretize the entire training dataset. You should now
have a dataset of low-resolution "images", where each "pixel" is the index of a
codebook vector.


## Autoregressive Model

Once again, building blocks can be found in the starter notebook. An autoregressive
sequence model can be used to generate images by bringing pixels into an arbitrary
order (often top-left to bottom-right) and trying to predict each entry (index)
given the previous ones. It's just a model with a softmax output layer, with as
many classes as there are codebook vectors (`K` further above). It is actually
very reminiscent of the language modeling task, just that you are predicting
codebook indices instead of word/character indices.

Some model options are:
- Flatten the "code images" into 1D sequences and train an RNN. This ignores
image structure and is generally a weak model.
- Flatten, but use a Transformer. This also ignores the image structure, but is
actually used in most recent models of this kind (e.g. the first DALL-E).
- Train a PixelCNN-style model using masked convolutions -- implementation is in
the notebook. Note, this was tested in Tensorflow 2.11 and may not work in other
versions due to changes in the internal code base.

Generating autoregressively is generally quite slow. There is a speed-quality trade-off
here: Heavier downsampling with the VAE makes the code images smaller, significantly
speeding up autoregressive generation. However, more downsampling makes it harder
to maintain high image quality for the decoder. In general, don't expect crazy good
results here; large Transformer architectures or rather complex PixelCNN designs
are often needed for that.
