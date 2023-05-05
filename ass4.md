---
layout: default
title: Assignment 4
id: ass4
---


# Assignment 4: Variational Autoencoders
**Discussion: May 11th**  
**Deadline: May 11th, 9am**

In this assignment, we will implement (and experiment with) 
a variational autoencoder (VAE).


## Implementing a VAE

From an implementation standpoint, a VAE is pretty much just an autoencoder with
a stochastic encoder and a regularized latent space. As such, you might want to
proceed as follows:
- Build an autoencoder. Use any dataset/model of your choice.
  - Note: If you get bored of using the same datasets all the time, the 
  [Tensorflow Datasets module](https://www.tensorflow.org/datasets) allows easy
  access to many more. A straightforward example could be using SVHN instead of MNIST
    (also digits of sorts, but in color!).
- Add stochasticity in the last encoder layer. With the common choice of a
Gaussian distribution, this just means splitting the layer into two parts, one
of which generates means, the other variances. Then you use these values to take
Gaussian samples. You can use `tf.random.normal`
for this -- take samples from a standard normal distribution, multiply
with the standard deviation and add the mean (this implements the "reparameterization trick").
Be careful with the layer that
generates standard deviations/variances; think about what value range these can be in and what
value range your layer returns. That is, choose a sensible activation function!
  - You will commonly find `tf.exp` as a choice here. This is appropriate as it always
  returns values > 0. However, it is often unstable (values/gradients tend to explode).
  If you are struggling with `nan` losses, try the following:
    - Use another function like `tf.nn.softplus`. This does not explode like `exp`.
    However, it seems to lead to worse results empirically.
    - Initialize the weights of your variance layer to 0. Then all outputs will
    be 0 initially, and `exp(0) = 1`. Empirically, this seems to prevent `nan`
    issues due to unstable gradients.
    - Reduce the learning rate.
- Add a regularizer term to the reconstruction loss, corresponding to the KL-divergence.
The exact form of this for the Gaussian case can be found in many available
  tutorials (like [this one](https://kvfrans.com/variational-autoencoders-explained/)),
implementations as well as 
[the original paper](https://arxiv.org/pdf/1312.6114.pdf).

If you want, you can use the autoencoder code from Assignment 0 (uploaded to gitlab)
as a base to start with.
Also, you will likely find many VAE implementations around the web. Feel free to use
these for "inspiration", but make sure you understand what you are doing! In particular,
here are some technicalities to pay special attention to:

- Choose your reconstruction loss carefully. Recall the discussion from the first
exercise (second part of Assignment 0). The loss needs to correspond to the negative
log-likelihood of the data conditioned on the latents, which will depend on how you choose to parameterize it.
If you just pick "some" loss, you might not have an actual variational autoencoder.
  - One particularly devious issue comes up when using Keras losses like `BinaryCrossentropy`
    or `MeanSquaredError`.
 By default, this will compute a per-pixel loss and then _average_ over all pixels.
 However, in this case we should _sum_ over pixels since:
    - We make the assumption that pixels are independent.
    - With independence, the image probability becomes the product of pixel probabilities.
    - Since we use log probabilities, this becomes the sum of log probabilities.
  - Normally it doesn't matter much if we sum or average over pixels, since if the number of
   pixels is always the same, this is a constant factor. However in this case it matters _a lot_,
    since averaging over pixels would make the loss much smaller relative to the KL
    divergence term, and this will screw up the learning.
- In your KL loss (and throughout the rest of your program), pay special 
  attention to where you need the variance, the log variance,
standard deviation, log standard deviation etc. Depending on how you parameterize
  it with your model, you will need different kinds of conversions here.


## Experiments

Train your VAE and generate some samples, perhaps trying out multiple
architectures and datasets. As usual, you can try any experiments that interest you.
Here is one proposal:

In the [beta-VAE](https://openreview.net/pdf?id=Sy2fzU9gl), the KL-term is
multiplied with a hand-picked hyperparameter `beta`, where usually `beta > 1`.
Implementing this model on top of your VAE should be trivial. Now, run several
trials with the same dataset/architecture, but varying `beta`. You can both let
`beta` go to 0, as well as increase it to larger numbers.
Some aspects you could investigate:
- How does reconstruction quality change with `beta`? What about sample quality?
- Can you find a "golden zone" for `beta` where you get the best samples?
Is this close to 1, smaller, bigger?
How large is this zone, i.e. how sensitive is performance to small changes in 
`beta`?


Alternatively, you could also try to introspect the latent space. For example:
- Encode an image, then start changing single latent dimensions and decode. Can
you find dimensions that correspond to interpretable features for a range of inputs?
- Find feature representations through averaging. For example, in MNIST you might
encode a bunch of left-rotated digits and average the codes, and do the same for
right-rotated digits. Presumably, these codes approximately represent their respective
rotation "feature". Now you could take a left-rotated digit, subtract the feature,
and add the right-rotated feature. Do you get the same digit just rotated differently?
- Interpolate between inputs in the latent space: Choose two inputs, encode both.
Then, move from one code to the other in small steps, decoding multiple times along
the way. The result should be a "morphing" between the two inputs.

All of the above may also interact with the `beta` term. According to theory,
higher `beta` should lead to better disentanglement in the latent space, and thus
more interpretable dimensions in the code.
