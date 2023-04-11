---
layout: default
title: Assignment 0
id: ass0
---


# Assignment 0: Why (And How) Generative Models
**Discussion: April 13th**  
**Deadline (optional): April 13th, 9am**

This first assignment is intended as a refresher on DL models as well as a first
look at why we might need generative models by
considering how other models fail at modeling data distributions properly.
Use this as an opportunity to set up
training/experimentation workflows that you can reuse in later assignments.
**Please read the entire text carefully for information on submissions etc.**

**This assignment does not need to be handed in. However, if you struggle with
the contents of part 1, you likely do not fulfill the prerequisites
for this class.**


## General Assignment Notes

- Assignments will generally
consist of writing code, usually to implement some kind of generative model and
to apply it/play around with it a little. If something is unclear, or you run
into trouble, **ask**. Do not spend all week doing nothing! Ask questions on
Mattermost (or via Gitlab issue) in a _public_ channel since other people might
have the same question (or be able to provide an answer)!

- Programming must be done in Python 3.x and Tensorflow 2.x.

- Code should be your own. In the event that you take parts of code from elsewhere,
clearly indicate this and provide a source. Do not copy-paste complete tutorials
  from elsewhere as your submission. Plagiarism is not permitted!

- Tasks are often deliberately open-ended in order to encourage exploration. Don't
concern yourself with achieving X% of points. These assignments are for you to
gain practical experience and to get into the habit of asking (and answering)
interesting questions.

- Unfortunately, we currently cannot supply GPU servers for the class.
You will have to work on your own machine or with Google Colab. If you need a
refresher, see the general notes of last semester's IDL class 
[here](https://ovgu-ailab.github.io/idl2022/ass1.html). 

- For each assignment, your final results should be in a Jupyter notebook. Use 
images/plots liberally. Also, feel free to use markdown cells to write text to 
explain or discuss your code/results. **Make sure that your notebooks include
outputs**. We should not have to run your notebooks to confirm that your code
works correctly.

- Submissions are handled via 
[Moodle](https://elearning.ovgu.de/course/view.php?id=14287). Upload your file(s)
before the deadline. **Regular submission is obligatory to be admitted to the exam.
You may miss at most two submissions (excluding the usual
things like medical reasons, emergencies etc).**
  
- Assignments can be worked on in groups (up to ~3 people). However, **each group
member must upload the solution separately**. Submissions should include
  information on who the group members are. All members needs to be able to
  present the solution (no freeloading!).

- The deadline is generally on Thursday morning.


## Part 1: Revisiting Autoencoders

Since autoencoders consist of an encoder and a decoder, they could in principle
be used to generate data through use of the decoder on its own. Try this:
- Build and train an autoencoder on a dataset of your choice. You may start with
a simple fully-connected network and MNIST (...).
- Confirm that your autoencoder has appropriate "generative capacity" by 
checking that reconstructions look reasonable (ideally on the test set).
- Use the decoder to construct data from randomly generated codes. Note that you
will first need a way to generate "reasonable" codes. One example could be
uniform random numbers within the bounds set by the encodings of the training
set. This is not necessarily a good method, just an example!
- Plot some of the results.

Are you happy with the outputs? You most likely aren't. Offer some hypotheses as
to why this might be. Think about and propose (and, if you want to, implement)
ways to produce better codes/images.


### Ideas for Further Exploration

- Use a different dataset. You should be able to easily switch to other small
image datasets such as CIFAR. The results will likely be more 
"dramatic"(ally bad).
- Try to use CNNs for the encoder/decoder instead.
- Look for other ways to explore the code space and its relation to the image
space. For example: Encode two images A and B. Interpolate between the two
codes, decoding some number of codes (maybe 10 or so) you encounter "along the 
way". This can give you an impression of how the code transitions between
different kinds of data (e.g. different MNIST digits). Arguably the simplest
interpolation method is linear interpolation. Can you think of "better" ones?


## Part 2 (Advanced)

When training a model in part 1, you likely used an "off-the-shelf" loss function
such as binary cross-entropy or mean squared error. In this section, we will take
a step back and think about _deriving_ loss functions based on prior considerations.
- Consider the decoder as a _probabilistic model_ `p(x|z)` where `x` is the data and
`z` the latent code. We will ignore the encoder for now.
  - First we need to decide on a probability distribution that fits our data well.
  Common choices for image data in [0, 1] are Gaussian distributions or Bernoulli
  distributions. Do either of these provide a good fit in your opinion?
  - Probability distributions generally have parameters `theta`. Reframe the
  decoder neural network as outputting those parameters. Now, we have expressed
  `p(x|z)` via `p(x|theta)` with `theta = decoder(z)`.
  - Another problem to consider is that we generally have high-dimensional data,
  not single numbers. Thus, you need to choose 
    high-dimensional probability distributions as well. A usual assumption is 
    that the different data dimensions are independent, which simplifies things
    considerably. For example, when using a Gaussian distribution, you could
    choose an _isotropic_ Gaussian (diagonal covariance matrix with all dimensions
    having the same variance), which makes things a lot simpler than the general
    case.
- In the commonly used _maximum likelihood framework_, the idea is to choose
parameters such that the probability of the observed data given these parameters
is maximized. Equivalently, we may minimize the negative logarithm of the probability.
  - The logarithm usually results in better numerical stability without changing
  the solution (why?).
  - Minimizing the negative probability is the same as maximizing the probability
  and allows us to formulate the problem as minimizing a loss function.
- Derive the negative log likelihood for the distribution you chose earlier and
use this as a loss function. Simplify as much as possible.
 Report on problems you run into, points where you
felt stuck etc. Note that in practice we often make simplifying assumptions, such
as setting the variance to a constant and only optimizing with regards to the mean 
  in case of a Gaussian distribution.
- Train your network. With `theta` as the output of the decoder, backpropagation
automatically trains the _network parameters_ such that the _distribution parameters_
`theta` are optimized given our loss.

To set some expectations, using a Gaussian distribution with fixed variance should
give you the squared error as a loss function; using a Bernoulli distribution should
result in the binary cross-entropy.
