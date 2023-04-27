---
layout: default
title: Assignment 3
id: ass3
---


# Assignment 3: Restricted Boltzmann Machines
**Discussion: May 4th**  
**Deadline: May 4th, 9am**

In this assignment, we will be implementing a Binary RBM. This requires some
low-level programming rather than just sticking a bunch of layers together.

Start off by implementing the RBM model and algorithm 18.1 from the deep learning book.
- The main thing here is actually implementing the `gibbs_update` step. Recall
that RBMs allow for efficient Gibbs sampling by first updating all hidden 
units at once, and then all visible ones. One Gibbs update means updating _both_
  hidden and visible units!
  - As such, you will need to implement both conditional distributions (`h` given `v`
  as well as vice versa). Use the
  conditional distributions from chapter 20.2. 
  - For sampling, `tfp.distributions.Bernoulli` should be helpful. Docs can be found 
  [here](https://www.tensorflow.org/probability/overview). Do not use other
  submodules such as MCMC-related functions!
- You will need to implement the RBM energy function to compute the log
unnormalized probabilities. This is required for both the positive and negative phase.
  - The book gives the formula for a single sample
  only (equation 20.5). You will want to implement a batched version of this for
  efficiency. This is not trivial! In particular, batching the third part of the
  formula (`v^T W h`) requires some thought.
  - The positive phase requires data samples. However, we do not have data for
    the hidden variables `h`. The most straightforward way to get these is to
    take a sample from `p(h|v)` where `v` is the real data.
- You can use `GradientTape` to compute the gradient update by treating 
"negative phase minus positive phase" as a loss function to be minimized.
However, the "ideal" value for this loss will actually be 0 (data and model
distributions are identical) despite the function in principle being able to
take on any value. If your loss decreases to arbitrarily large negative values
there is likely something wrong with your training (although it's normal for
negative values to appear in the beginning). 
  - Make sure you compute the gradients correctly! Although the Gibbs sampling
  procedure makes use of the model parameters, this should not influence the 
  gradients, i.e. the samples are treated as constants just like the data. In Tensorflow 2,
  as long as you only wrap the _necessary parts_ (i.e. the energy function computation)
  in `GradientTape`, this should not be an issue.
- Alternatively, the gradients are relatively simple to work out "by hand", which
would allow you to implement the gradient steps directly, without a tape.

Once you have the basic algorithm going, you might want to test it first. Since
we are working with binary RBMs, MNIST seems like the best option here. You may
"binarize" the data by rounding all values to 0 or 1, however since MNIST is
already almost binary this will likely not make a large difference 
(still, it is "more correct" to do so). Experiment
with different numbers of hidden units and burn-in steps, and generate some
samples of the trained models for subjective evaluation.

Next, you should improve on the basic procedure.
- Experiment with ways to get the sampling going, i.e. "initialize a set of m 
samples to random values". Likely the most basic way is to sample each pixel
independently with probability 0.5. Implement and test at least one other way
that you think is more appropriate.
- Algorithm 18.1 is rather slow due to the long burn-in time required at each
gradient step. Implement contrastive divergence and/or persistent contrastive
divergence (18.2/18.3) to alleviate this. This requires relatively minor changes.

  - For PCD, the training just needs to have access to the final samples of the
previous step as a starting point. 
  -  For CD, use the actual batch of data as a
starting point. The only difficulty here is that you don't have a starting point
for the hidden units. You can create one, for example, via a single Gibbs
sampling step from the visible units (just like for the positive phase).

Test your algorithms once again and compare the results (as well as the speed at
which you achieve them) to the basic algorithm.

## Bonus

Feel free to try other training methods such as pseudolikelihood or score matching
(from optional reading).
