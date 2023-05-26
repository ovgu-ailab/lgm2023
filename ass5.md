---
layout: default
title: Assignment 5
id: ass5
---


# Assignment 5: GANs
**Discussion: May 25th**  
**Deadline: May 25th, 9am**  

Guess what we're doing this time.

**Note:** For this assignment and in general, you can upload more than one notebook
if you find this better, e.g. to avoid overly long notebooks with a loooong string
of pretty much independent experiments.


## Basic Setup

Implementing the basic GAN logic isn't too difficult. You will likely want to
use low-level training loops (i.e. `GradientTape`) because of the non-trvial
control flow. There are many examples around the web that can help you get
started (but note that many are outdated, e.g. using TF 1.x).
For example, there is a DCGAN guide
[on the TF website](https://www.tensorflow.org/tutorials/generative/dcgan).
There is also an `assignment05_starter` notebook on Gitlab with some more tips.
- Define two models: One that maps from a noise space to data (generator) and
one that maps from data to a single number (discriminator). Use whichever dataset
you like!
- Train the two models. The simplest setup is to alternate one step of generator
training and one step of discriminator training.
- To train D, get a batch of generated samples from G and a batch of real samples
from the dataset and minimize D's classification error. Be sure to only
update the parameters of D!
- To train G, generate a batch of samples 
and maximize D's classification error. Be sure to only update the parameters of
G!

This should be enough for a functional training procedure. Train some models and
generate samples for evaluation. They will most likely be terrible.

Take note: Evaluating whether GAN training is progressing/"working" is 
difficult. The loss values are not very informative. You will want
to take some samples and plot them every so often while training is progressing
to get an impression of the current state. However, even this can be misleading:
You might run into mode collapse problems early on, which you can take as
evidence that training is not working, and stop the process early. However, it
_could_ actually happen that the mode collapse "magically" gets fixed over the
course of a few training iterations, and diverse samples are produced. For this
reason, consider always training for a large number of steps (larger than e.g.
VAEs) and just see what happens.


## Improving GAN training

GANs are notoriously difficult to train. In the rest of this assignment, you are
asked to try out various ways to improve the basic procedure. There are countless
advanced GAN variants, but for now you may focus on "tricks" to make the original
formulation more stable. Here are some leads:
- [The original GAN paper](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)
proposes (at the end of section 3) to "flip" the generator loss, which should
lead to better gradients early in training.
- [This follow-up](https://arxiv.org/pdf/1606.03498.pdf) discusses some techniques
for improved training. Some of these are probably a bit much to implement, but
you could try things like one-sided label smoothing, minibatch discrimination or
 making use of class labels (semi-supervised learning).
  - In particular, _feature matching_ can stabilize and speed up GAN training
    significantly, as well as mitigating mode collapse. Here, you train D as usual,
    but G is trained with a loss that tries to match the features in D for generated
    data with those for real data. The paper proposes using "an intermediate layer"
    of D to get the features, without specifying further. In practice, you can
    get good results by summing this loss over all or at least a number of layers
    of D (e.g. after every activation function, or after every second, or...).
- [DCGAN](https://arxiv.org/pdf/1511.06434.pdf) is a reference architecture for
how to implement GANs with CNNs (though it is outdated by now)
  and includes many "tricks" that seem to work well
in practice. In particular, if using Adam as optimizer, have a look at their
hyperparameters (very non-standard).
- [ganhacks](https://github.com/soumith/ganhacks) is a repository of yet more
tricks that might help, although this is quite outdated.
- Batchnorm can be problematic in GANs -- see the text in `assignment05_starter.ipynb`.
- Don't be afraid to go big! These models can really profit from increased capacity
in terms of number of layers and layer sizes.
- It can take literally _thousands_ or even tens of thousands of training steps 
on more complex datasets until results really start "cleaning up". Be prepared to
  wait some time for your final models.

Include as many of these methods as you want/need into your model and try to
achieve some nice samples!


## Optional: Advanced GAN Architectures

GANs have come a long way since their original inception in 2014, and discussing
and/or implementing all significant improvements is not feasible. Below you can
find some more leads for advanced architectures; if you have the time, you may implement some of these, 
experiment with the associated parameters and compare with your previous implementation.
Some of these methods may be discussed in class in future sessions.
In principle you can also mix and match these methods with each other or the
improvements from part 1. Some of these mixes might work well, others not at all...

- [Conditional GAN](https://arxiv.org/pdf/1411.1784.pdf)
  - Works for datasets that have class information or any other kind of metadata
  to condition on.
  - Simply add this information (encoded somehow, e.g. as a one-hot class vector
    or an embedding) to the generator and discriminator. Now the discriminator can
    learn to reject/accept data based on the class, and the generator can generate
    data for specific labels.
- [Wasserstein GAN](https://arxiv.org/pdf/1701.07875.pdf)
  - Use a linear output activation in the discriminator.
  - The loss is just the difference between real/fake outputs, 
    flipped for the generator loss.
  - The discriminator includes weight clipping. 
    You can do this easily in Keras using a layer’s `kernel_constraint` and 
    `bias_constraint` arguments and the `clip_by_value` function. 
    _All_ layers with weights need to have such constraints! 
    Be careful with normalization layers such as Batch Normalization, as these 
    could easily “break” the discriminator by rescaling the outputs to arbitrary ranges.
  - Usually the discriminator is trained for several steps each time (e.g. 5-10 
    discriminator steps per generator step).  
- [Improved Wasserstein GAN](https://arxiv.org/pdf/1704.00028.pdf)
  - Remove the weight clipping from the WGAN.
  - Add the gradient penalty as described in the paper:
    - Create an “interpolation batch” between the real and generated batches.
    - Run this batch through the critic, and compute gradients of the critic’s 
      output with respect to the input batch. 
      Note that this means having a `GradientTape` (to compute input gradients) 
      inside another one (to compute parameter gradients), but this is fine.
    - Compute the deviation of these gradients from the desired value of 1, and 
      add this term to the original loss function, scaled by some factor.  
- [Least Squares GAN](https://arxiv.org/pdf/1611.04076.pdf)
  - Simple: Use a linear output activation in the discriminator, and a squared
  error loss function. However, note in the paper that there are different choices
    for what numbers you use as labels for real/fake data.
- [Spectral Normalization](https://arxiv.org/pdf/1802.05957.pdf)
  - Heavy on theory, but to implement this you basically just need to wrap all
  layers in D that have weights in a `SpectralNormalization` object you can find
    in `tensorflow_addons.layers`. You should also remove all other normalization
    layers such as batchnorm from D, and only use "contractive" activation
    functions (ReLU and LeakyReLU are fine).
  - In principle, you can view this as _enforcing_ a 1-Lipschitz constraint,
  instead of only regularizing for it as in the WGAN. However, you can use spectral
    normalization with any kind of GAN loss (cross-entropy, least-squares, Wasserstein...).
- [Progressive Growing](https://arxiv.org/abs/1710.10196),
  [StyleGAN](https://arxiv.org/abs/1812.04948) or 
  [StyleGAN2](https://arxiv.org/abs/1912.04958) (these will all require 
  significantly more effort to implement).
