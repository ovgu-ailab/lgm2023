---
layout: default
title: Assignment 8
id: ass8
---


# Assignment 8: Score-based Generative Models
**Discussion: June 15th**  
**Deadline: June 15th, 9am**


We continue our mission to implement every single kind of generative model ever
invented by tackling score-based generative models. Arguably the simplest approach
is to use _denoising score matching_ with multiple noise scales.
The ingredients are:
- A _noise-conditional_ score model that takes a noisy input and returns the score.
- The loss function. which applies and sums up the score model at all noise scales.
- A sampling function based on annealed Langevin dynamics.

Let's look at these components in turn. We will be referring back to these papers:
- [Original paper](https://arxiv.org/pdf/1907.05600.pdf)
- [Improved follow-up](https://arxiv.org/pdf/2006.09011.pdf)


## Noise-conditional Score Model

In general, a score model should return the gradient of `log(p(x))`, given `x`.
This means the model should have a structure where it takes in a data sample, and
returns an output with _the same shape_. 
- In principle, you could apply a CNN with
no downsampling, and possibly dilated convolutions for larger receptive fields.
- Another option is to have an encoder-decoder network with downsampling and
upsampling. This can be more efficient, as parts of the network work on a lower
resolution. You can add _skip-connections_ from the encoder to the decoder, like in
a U-net, to improve performance.

We cannot expect the score model to work at all noise scales, but training one model
per scale is not feasible. Instead, the model should take a second input, which is
the scalar noise value `sigma`. Then the model, can return different scores depending
on the noise level. Recall that you can creat
e multi-input models like this:

`model = tf.keras.Model([x_input, noise_input], score_output)`

The bigger question is how this conditioning should look like, exactly. The original
paper proses _conditional Instance normalization_, which normalizes the hidden
layers to different means and standard deviations depending on `sigma`. The
follow-up proposes something much simpler: Simply run the model on `x`, and then
divide the output by `sigma`. See "Technique 3" in section 3.3 of the improved paper.

A note on the model architecture: **Do not use batch normalization!** See if you
can figure out why this is a bad idea for these models. Recommended replacements
are `InstanceNormalization` or `GroupNormalization` from the `tensorflow_addons`
package.


## Loss function

The loss is relatively simple, for example, you can find it in equation 2 of the
improved paper (section 2.2). There, `x` is a data sample, and "x tilde" is a
noisy version of that sample. Noisy versions are attained simply by adding
random normal noise with mean 0 and  the given standard deviation.

The problem is that we need to sum the loss over all noise scales, and the score
network needs to be applied separately to each noisy sample. This is _very_ slow
for many noise scales. If you run into this issue, you can also _sample_ a nosie
scale randomly at each training step, and only do the training for that noise scale.
- The sampling needs to happen inside the training step. Make sure use Tensorflow
functions, _not_ numpy or native Python random functions!! These will appear to work,
but actually only sample a single noise when compiling, which will be used throughout.
Tensorflow random functions, on the other hand, are part of the graph, and will
create a new random sample each time they are called.
  - When writing a custom training loop, things like adding noise can also be
  handled outside the training step, since this needs to be backpropagated through.
- You could either sample a single noise scale per training step, and use that for
the entire batch, or sample one noise scale per batch element. The latter seems
like the more sensible and unbiased version. However, it seems to be very unstable.
Thus, you might want to go with option 1 (single noise scale per batch), which
is also simpler to implement.

There is also a noise-dependent weighting function that the papers propose to set
to `sigma**2`. In the aforementioned equation 2 in the improved paper, this has
already been "integrated" into the loss itself.

Finally, note that the sampling approach makes the noise somewhat unreliable.
It may fluctuate significantly, depending on what noise is sampled at each step.


## Annealed Langevin Dynamics

Again, this can be found in the papers, e.g. Algorithm 1 of the improved version.
- Initialize samples to random values
- Iterate over noise scales
  - For each noise scale, compute the step size alpha
  - Run some number of steps of the MCMC update
    - For each step, apply the score model to the previous sample
    - The new sample is the previous, plus the score update, plus random noise,
    all scaled by the step size

For efficiency, it's a good idea to use `tf.function`. However, wrapping loops
in a function often doesn't work so well. Instead, you can offload the per-step
update into a separate function, only wrap that with `tf.function`, and then loop
over that.

It is typical to choose the overall number of steps on the order of 1000 or so --
so you would use `1000 / number_of_noise_scales` per scale. More may be better.

**Note** the algorithms in the first and second paper are slightly different with
respect to `alpha`; one uses two times that of the other. If you are using the
recommendations from the improved version, you should also implement the algorithm
of that paper!

Finally, pay attention that `alpha` is scaled using the _variances_, not standard
deviations. Things like this are easy to get wrong and may screw up your entire
implementation!


## Choosing Noise Scales

So far, we have not discussed how to choose the noise scales. The original paper
does this in an ad-hoc fashion:
- The smallest noise should be such that it cannot (or barely) be seen. The authors
choose `sigma_1 = 0.01`. This is still a bit high IMHO, maybe try 0.001.
- For the largest noise, they choose `sigma_L = 1` for image data scaled to [0, 1].
- They choose `L = 10` noise scales, with a gemoetric progression. Check out `np.geomspace`
to generate this.

These values may work for small datasets. In the follow-up, the authors conduct
analyses to arrive at more principled choices. These are techniques 1 and 2 in
section 3 of the paper. These result in much higher `sigma_L` and `L`. For example,
for MNIST I arrive at `sigma_L ~ 20` and `L > 100`. **There is a starter notebook
on Gitlab that does these computations for you**.

In principle, your code should not get any more complex whether you have 10 noise
scales or 10000! It's all just loops anyway.


## Other Techniques

The improved paper also offers guidance on how to choose the step size `epsilon`
in Langevin sampling. The aforementioned notebook also implements this.

Finally, "Technique 5" states that we should use exponential averages of the
learned parameters in inference, instead of using the parameters directly. This
functionality is built into TF optimizers in recent versions:
- Pass `use_ema=True` to the optimizer constructor.
- Set `ema_momentum` in the constructor to an appropriate value. ;)
- Using `model.fit()`, the variables are automatically overwritten by the averages.
- Using custom training, use `optimizer.finalize_variable_values(model.variables)`.
- This may not make much of a difference/be worth it except for very large,
overfitting, unstable networks.


## Fun Variations on Generation

Score-based models are quite flexible in their applications. A few things you
could try are:

### Inpainting
The original paper has a variation of Langevin dynamics in the appendix B,
algorithm 2. Here, we mask out certain parts of an input and have the model
generate the rest. That is, the model has to "fill in the gaps".

### Interpolation
Interpolating between images is possible, but a bit more complicated than e.g.
in VAEs or GANs. One method is named in appendix B.2 of the improved paper.

### Creating Variations
It's possible to partially diffuse an image and then reconstruct it from there
to receive a similar, but different picture. You can modify Langevin dynamics as
such:
- Start not from the largest noise scale, but some intermediate one.
- Start not from a random sample, but a noisy data sample, with the noise level
as chosen above.
- Run Langevin dynamics down to the smallest noise scale.

This will "reconstruct" the noisy data sample, but since the process is random,
the result will be different. How different will depend on what noise scale you
started from. The larger, the more different it will be.
