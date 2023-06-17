---
layout: default
title: Assignment 9
id: ass9
---


# Assignment 9: Denoising Diffusion Models
**Discussion: June 22nd**  
**Deadline: June 22nd, 9am**


We made it! Diffusion models are heavy on theory, but surprisingly simple to
implement. They are also very similar to score-based models. As such, we will
mainly treat the _differences_ to a functioning score-based model implementation.

You are welcome to use the score implementation on Gitlab as a starting point if
you want; there is also a starter notebook for diffusion-specific components.
Finally, [the 2020 paper by Ho et al.](https://arxiv.org/pdf/2006.11239.pdf)
serves as a reference.


## Forward Process

The "noisification" is a bit different in diffusion models. First, you need to
decide on the number of steps `T` and a noise schedule `beta_t`. The paper proposes
(beginning of section 4)
`T = 1000`, and `beta_t` to increase linearly from 0.0001 to 0.02. You can use
`np.linspace` for this. In many places, you will also need `alpha = 1 - beta`
and `alpha_bar = np.cumprod(alpha)`. `alpha_bar` can be seen as how much data
is still left at a time step `t`. You can plot `alpha_bar` and make sure this
goes to approximately 0. In the paper, these terms are defined right before
equation 4, in section 2.

You can experiment with smaller `T`, but will likely need to increase the `beta`
terms to make up for this -- and make sure `alpha_bar` is still going to 0.

Finally: Note that the paper scales data to [-1, 1] instead of the usual [0, 1].
They do this so that the scale at different noise levels is more consistent (as
the noisy data goes to mean 0, standard deviation 1). Another option could be to
scale the data to mean 0, standard deviation 1. Simply leaving it in [0, 1] may
also work. Whatever you do, note that you may need to reverse this scaling (scale
back to [0, 1]) before plotting samples from your model.


## Time-conditional Model

Like score-based models, diffusion models need to be conditioned on the noise
level. They do not propose the simple "divide by noise" conditioning, though.
The paper proposes to condition on the _time step_ `t`, although conditioning on
the noise level would also be okay.

To be precise, the paper uses a _positional encoding_ of `t`, like a transformer.
This replaces `t` by `(sin(f*t), cos(f*t))` for different frequencies `f`. An
example is shown in the starter notebook. The resulting vectors can be broadcast
over width/height and be concatenated to the input and/or hidden feature maps of
the model.

Finally, as in score-based models, avoid batch normalization in your architecture!


## Loss function

The loss function can be found in equation 14 of the paper, or algorithm 1. It
involves acquiring a noisy data sample using `alpha_bar`, applying the model, and
computing the squared difference between the result and the "target noise". Like
in score-based models, you can/should sample a time step `t` for this. Once again,
you have a choice between sampling a single `t` for the whole batch, or one `t`
per batch element. Unlike for score-based models, the latter version actually
works this time, but the first one is still fine.


## Sampling

Check algorithm 2 of the paper. Sampling uses Langevin dynamics like score-based
models, but you do not have to look for a suitable `epsilon`.


## Aside From That...

... there is not much difference to score-based models. Diffusion models tend
to be a bit easier to train and perform better. You can do all the modified
generation types (inpainting, interpolation, variations...) proposed in the
previous assignments as well.
