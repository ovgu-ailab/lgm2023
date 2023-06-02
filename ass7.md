---
layout: default
title: Assignment 7
id: ass7
---


# Assignment 7: The NICEst Assignment
**Discussion: June 8th**  
**Deadline: June 8th, 9am**

In this assignment, we want to implement some simple flow models on toy 
datasets, as well as attempt to fit simple "real" datasets like MNIST.


## NICE

The OG flow model is the NICE model from 
[this paper](https://arxiv.org/pdf/1410.8516.pdf). It is also a relatively simple,
making it a good candidate for first experiences with these models. Recall that
in [one of the readings from the lecture](https://blog.evjang.com/2018/01/nf1.html),
code examples for how to implement flows in Tensorflow Probability are given.
However,
- they are outdated,
- they did not work when we tried to adapt them to TF 2.0 (no gradients),
- to understand every detail of training flow models, we want to implement
everything ourselves.

But first, a note on terminology: In principle, it doesn't matter which direction
of the flow you call forward or backward, in which direction a function `f` is 
applied and in which its inverse, etc. However, it's easy to get confused here
because people use different conventions. I will strictly stick to the convention
from the NICE paper, which is:
- `forward` goes from the data (complex distribution) to the simple distribution.
This is the function `f` ("encoder" if you will).
- `backward` goes from simple distribution to complex (data). This is `f^-1`,
the inverse of `f` ("decoder").

You might want to proceed as follows:

### NICE Coupling layer

Implement a single NICE coupling layer. Note that this is _not_ a single neural network
layer (but inheriting  from `tf.keras.layers.Layer` is still useful)!
  - NICE uses a very simple additive coupling layer. Check the paper, equations
  3 and 4 and/or section 3.2. Your layer needs a network to implement the 
  transformation function `m`. You can use a Keras model for this. This should
  take a _d_-dimensional input and also produce a _d_-dimensional input (what
  _d_ should be is explained further below). The network itself can be arbitrarily
  complex (this also means hidden layers can be much larger or smaller than _d_).
    - Take care that each coupling layer should get _their own_ network. No weight sharing!
  - Implement forward and backward operations for the layer. These are very
  simple: Split the input into two parts, and take `y1 = x1` and `y2 = x2 + m(x1)`,
  where `m` is the network defined above. For the backward layer, subtract
  the shift instead. Afterwards, put the two parts together again and return the result.
    - How to split the input? In principle, there are many ways of doing it. Most
      likely, you want to split it into two parts of the same size.
      In this case, `d` above would be half the input dimensionality.
      A very simple way would be to simply split it in the middle (e.g. `tf.split`).  
    You could also try
    other schemes, e.g. all even dimensions go into part 1 and all odd dimensions
    into part 2, or...

### NICE Full Model
The full NICE model simply stacks an arbitrary number of such coupling layers.
  - The forward call chains the forward calls of the individual layers. However, note
  that the layers pass `x1` through unchanged! The usual way to solve this issue
  is to alternate layers, such that one modifies the first half, the next one the
  second half, the next one the first half again, etc. There are different ways
  to implement this.
    - You could implement the layers such that they can either change the first
    or second half, and use these in alternation. This is probably the best way.
    - You could implement the layers such that they always change the same half,
    but inbetween layer calls, "swap" the data halves. E.g., `tf.split` followed
    by `tf.concat`, but the other way around. This may be more "elegant", but is
      error-prone. Don't do this.
  - The backward call simply chains the backward calls of the individual layers (once 
  again take care to swap the data halves appropriately). Of course, the layers
need to be applied in reverse order.
  - The NICE paper also proposes a diagonal scaling layer at the end. This is
  because otherwise the model has no way of contracting or expanding the space.
  You can implement this via a `tf.Variable`, one number per data dimension,
  and simply multiply this with the output at the very end of `forward`.
  Accordingly, you need to invert this at the _start_ of `backward` (via 
  division).
    - You can parameterize this using `tf.exp`, which makes many computations
      (such as taking logarithms) easy -- check the paper!
  - The NICE paper most certainly has an error in section 5: All the equations
  show applying `m` to `x`, but it should be the respective `h`
    output of the previous layer.

As a first sanity check, you should set up a very simple model on some toy data
and check that the `forward` and `backward` functions are actually inverses of
each other. That is, check that the difference between `data` and 
`backward(forward(data))` (and/or the other way around) is near 0 (small 
differences due to numerical reasons are okay).

### Training
That takes care of the model itself. Once this works, setting up training is
very simple!
- The training criterion for flow models is maximum likelihood. As such, we
have to add functions to compute this. The general equation for flow models is:
`log(p(data)) = log(p(forward(data))) + log(det(forward_jacobian(data)))`. The
ingredients are:
  - Passing data through `forward` is simple, if you sanity-checked that your
  implementation works! Now we need a "simple" probability distribution for the
  "latent" space. In principle, this can be anything that is easy to evaluate.
  As is often the case, the Normal distribution is a popular choice, although
    the paper proposes using a logistic distribution instead. You should
  use `tensorflow_probability` to create a distribution and use the `log_prob`
  function to evaluate probabilities.
  - NICE with only additive coupling layers actually has a determinant of `1`
  for the Jacobian, meaning that the term disappears completely. If you use a
  scaling layer at the end (recommended) the determinant is simply 
  `prod(scale_factors)` (or `sum(log(scale_factors))` for the log deteminant). See section 3.3 in the paper!
  
With all this taken care of, your model is ready to train. First, try it on
simple toy data. See the notebook for a sample dataset (parabola). Training
proceeds as usual, by gradient descent. You can use the negative log likelihood
as a loss function and use standard TF optimizers. Feel free to try other toy
datasets as well. Make sure you can successfully fit such datasets before moving
on! If training fails, there are likely problems with your implementation.
- The paper advocates using four coupling layers. You can also try more; less
is a bad idea because it makes it difficult to model dependencies between the
two "splits".
- Give it some time -- flows seem to require more training steps to start
producing results (could be several thousands even on toy data).
- The paper uses rather large models -- five hidden layers with 1000 hiden units,
_per coupling layer_, for MNIST. I have found this overfit _horribly_ when using
"modern" optimization methods like Batchnorm. Be sure to keep track not only of
the training loss, but also a validation/test loss to check for overfitting. You
may want to make the models much smaller than proposed in the paper.
  - LayerNormalization seems to work much better than BatchNormalization 
  _for some reason_.

Do not expect great results on datasets like MNIST. At the end of the day, this
is not such a nice model. Haha.

## Applications
Here are a few things you can do with your trained model.

### Inpainting
See Section 5.2 of the NICE paper. You can fix a part of the input, and have the
model generate the rest. This is also possible with e.g. autoregressive models, but
they are limited due to their fixed generation order. With flows, any dimensions can
be inpainted given any others.

The method simply uses gradient ascent to create an input that maximizes the
likelihood, but only optimizing the "unknown" pixels (fixing the known ones).

### Density Estimation/Outlier detection
A trained flow should be able to detect "atypical" inputs. A simple experiment
could go like this:

1. Define a "corruption process" to turn data points into "atypical" ones. For
example:
   - Adding increasing amounts of random noise
   - Rotating images by increasing angles
   - Other transformations such as shearing, contrast, etc.
   - Slowly morphing/interpolating inputs into ones from a different dataset (e.g. MNIST -> FashionMNIST)
   - ...
   - 
    
2. Compute likelihoods, using your flow model, on the original data and on increasingly
"corrupted" transformations of the data. Can you see a difference? Ideally, the
corrupted data should receive a lower likelihood, and this should become more visible
as you increase the degree of corruption.
3. Can you find a threshold for likelihood below which data is likely to be an outlier?
What kind of precision/recall can you achieve with such a simple method?
