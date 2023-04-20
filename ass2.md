---
layout: default
title: Assignment 2
id: ass2
---


# Assignment 2: Let's go to Monte Carlo
**Discussion: April 27th**  
**Deadline: April 27th, 9am**

In this assignment, we will be investigating Monte Carlo Methods with a few
simple examples. There is a lot of text for explanation, but the actual tasks
are rather compact.


## Markov Chain Monte Carlo

To understand the basics of MCMC methods, we consider the simple example in 
section 17.3 of the deep learning book, where we are interested in sampling 
single integers _x_ from 0, ..., _n_ according to some distribution. Try the 
following:
- Define a transition matrix _A_ as in the book. This should be a stochastic matrix,
i.e. each column must sum to 1. You can choose _n_ as you wish; it's okay to stick
with a small number (say, _n_=5). You can create the matrix from random values,
or write it down by hand. For theoretical reasons, it's best to have no 0s in the matrix.
- Create an arbitrary initial distribution _v(0)_, i.e. an _n_-element vector with elements
summing to 1.
- Repeatedly multiply the matrix _A_ with the vector. The result should converge
very quickly (just a handful of steps) to a vector _v'_. This vector represents 
the probability distribution that this Markov chain (represented by matrix _A_) 
will converge to!
- Try different starting values of _v_. How does this influence _v'_ ?

Now, we run a Markov chain:
- Start off with an arbitrary state _x(0)_.
- Repeatedly take the column from _A_ that corresponds to the current state _x(i)_
 (i.e. if the current state is 3, you take the 3rd column). Sample a new state
 _x(i+1)_ from this probability distribution. Do this many times and collect all
 samples in a list or something similar.
- At the end, plot a histogram of your samples. This gives you an empirical
distribution over the samples. Does this distribution match the vector _v'_
computed above? If not, you might need to run the chain for more steps. If yes,
congratulations! By running your chain you are sampling from the distribution
represented by _A_.
- Note that you can use `tf.random.categorical` for sampling; however this only
takes _logits_ of a distribution. In this case, you might want to define _A_ in
terms of logits in the first place, and apply softmax (per column!!) to get
a regular probability matrix. Alternatively, `tensorflow-probability` supports
categorical distributions based on probabilities as well as logits.


## Gibbs Sampling & Mixing

In most of our use cases, the situation is not as in the example above: We don't
have a transition distribution given and just start running a Markov chain on it.
Instead, we have a _desired target distribution_ and need to figure out how to
get there.

Let's try to sample from a mixture of Gaussians via Gibbs sampling.
- Set up a distribution. In order to be able to do any conditioning, this needs
to be multivariate. For simplicity (and easy plotting), stick to a 2D distribution.
Also, we want at least two components; these can be simple independent Gaussians.
- Start with an arbitrary initial sample (e.g. a vector of 0s). Now, repeatedly
do the following: Sample a new value for _x_ given the value for _y_. Then, sample
a new value for _y_ given the (new!) value for _x_. Since we now have sampled new
values for both dimensions, we have essentially taken a new sample of our 2D
distribution.
- You can use `tensorflow-probability` (in particular, the `distributions` module)
to build the distributions. There are tutorials on 
[the Tensorflow website](https://www.tensorflow.org/probability/overview). You will
likely want to use `MixtureSameFamily` of `Normal` distributions.
- The hardest part is to figure out how to take a conditional sample. It turns
out that in this case, where _p(x,y)_ is a mixture of Gaussians, the conditional
distribution _p(x|y)_ (as well as the other way around) is a mixture of Gaussians
as well! You can find a derivation 
[here](https://stats.stackexchange.com/questions/348941/general-conditional-distributions-for-multivariate-gaussian-mixtures).
Being able to translate such formulae into code is important! By using independent
Gaussians as components, the only thing that actually changes in the conditional
distribution are the mixture coefficients!
- You can use `tfp` to sample from the conditional, one-dimensional, mixtures of
Gaussians as mentioned above. You might say: Why not just sample from a two-dimensional
  mixture of Gaussians in the first place? The reason we don't do this is that
  this is a contrived example to show the concept of Gibbs sampling. ;)
- Please note the file `assignment02_starter.ipynb` uploaded to the course Gitlab --
this can help with some of the above issues.

You should collect a reasonable number of samples (1000 or more) and plot both the
target distribution (mixture of Gaussians) as well as your samples. Do the samples
reflect the distribution well? In particular are both modes of the Gaussian mixture
covered equally? You can do this visually and/or using statistics. 
Also, experiment with different locations/scales for the Gaussians.
That is, move the components further apart or closer together and repeat the
sampling process each time. The quality of the samples should vary dramatically
based on the distance between components!


## (Optional) Importance Sampling

Finally, let's give importance sampling a shot. Say we are interested in estimating
the expected norm of a Gaussian mixture distribution via Monte Carlo sampling,
but cannot sample from the distribution directly.
Instead, we can only sample from a simple Gaussian distribution with mean 0 and
standard deviation 1. Let's try to estimate the average norm via
importance sampling. In the language of section 17.2:
- _p_ is the mixture of Gaussians
- _q_ is the simple Gaussian
- _f_ is the (Euclidean) norm

Take a bunch of samples from _q_ and compute the Monte Carlo estimate (i.e. sample
average) as in equation 17.10. You may want to plot how this evolves over time
(i.e. with the number of samples). Does it converge? Next, also take samples from
the mixture of Gaussians directly (the thing we pretended we couldn't do) and
compute the average norm on those samples. Compare with the importance sampling
estimate. Do they converge to the same point? Does this depend on the locations/scales
of the mixture components? Experiment with different values!

