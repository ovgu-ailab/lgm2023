# LGM 2023 Recap

This document is intended to serve as a _rough_ overview of this year's material.
Obviously, this cannot be exhaustive. It should be taken as a rough guide of what
parts of the material are relevant, what kind of questions you should be able to
answer, etc., as well as _excluding_ certain parts of the material that you don't
need to know in detail (all these points especially with reference to the exam).


## In General

You should (be able to)...
- Explain how specific classes of generative models work
- Give the _full story_ on your own without being prompted for every step
- Have a deep _conceptual_ understanding of the different models
- Explain what certain kinds of generative models can/cannot do, what their
respective advantages and disadvantages are, and how they relate to each other

You do not need to
- Provide detailed formulas etc.
- Give mathematical proofs
  - Still, you should be okay with basic ideas of probability theory, e.g.
  conditional probabilities, Bayes' rule, log probabilities, expected values, etc.
- Know specific architectural details, except where necessary (e.g. StyleGAN is all
about the architecture)


## Generative Models

A generative model is a model of a data distribution, `p(x)`. The main differentiation
is from _discriminative models_ `p(y|x)`. An example of a discriminative model is
a classifier. These models usually learn to ignore large parts of the input and
focus on the relevant patterns. They often map from a high-dimensional space (e.g.
images) to a lower-dimensional one (e.g. classes).

In contrast, generative models need to learn "all" aspects of the data to model
`p(x)` faithfully. This can be a more difficult task. Outputs (model samples or
probabilities, scores etc.) are often high-dimensional.

You should know what a generative model is and how it relates to other kinds of
models (e.g. most of the IDL content is discriminative models).  That is, you should have a rough idea what
this class was about. ;)


## Graphical Models

Many generative models are formulated as _graphical models_. Aspects here are
- What are nodes and edges in a graphical model? -> Random variables and dependencies
between them
- What types of models are there? -> Directed and undirected (technically also hybrid)
- Recall the concept of _separation_, i.e. certain variables can become _conditonally
independent_ when other variables are observed. This can be different in undirected
or directed models. In particular, we have the "collider case" in directed models
which makes variables _conditionally dependent_! This is what makes inference hard
and leads to techniques like variational inference/VAEs!
- You should know what class certain models belong to, e.g. Boltzmann machines
are undirected, VAEs are directed, etc., as well as be able to sketch their
respective graphical models.
- Do not mix up _graphical models_ and _neural networks_. A graphical model
does not make any reference to neural networks. Those are generally used to
implement the conditional distributions in the graphical model. Sometimes the
roles of the neural network is a bit more complicated, e.g. in the VAE the
"encoder" implements the approximate posterior `q`. **There is no encoder in the
graphical model**!
  - As a rough rule of thumb, the graphical model encodes the "generative process"
  and nothing else. E.g. the VAE and GAN graphical models look the same!


## Undirected Models/RBMs

Undirected graphical models are those with undirected edges (duh). Their probability
distributions are defined in terms of _clique potentials_ which, taken together,
give an _unnormalized_ probability distribution. This needs to be normalized,
and the normalization constant is called the _partition function_, often denoted
as `Z`.

The partition function is often intractable, as it is the sum/integral of the
unnormalized probability for _all_ possible data points, the number of which
grows exponentially with the data dimensionality. Thus, to train undirected
models, we usually need to "side-step" computation of `Z` somehow.

The only example we looked at explicitly was the Restricted Boltzmann Machine
(RBM). Going top to bottom,
- _Energy-based models_ are a subclass of undirected graphical models.
- _Boltzmann Machines_ are a subclass of energy-based models (with a specific
energy function).
- _Restricted Boltzmann Machines_ are Boltzmann machines with a bipartite graph
structure.

### Training
Training RBMs is done using maximum likelihood. Again, the partition function
is in the way. Luckily, we don't actually need to compute the likelihood itself --
for gradient-based optimization, we only need the gradient! It turns out that
the _gradient_ of the partition function can be re-written in a way that makes
it feasible to approximate it via model samples. This leads to the training
algorithm using a _positive_ and _negative phase_.
- In the positive phase, the unnormalized probability of data points is _increased_.
- In the negative phase, the unnormalized probability of model samples is _decreased_.
- In the limit, if the model recovers the data distribution successfully, both
phases cancel out and training converges.

Despite the terminology, both "phases" are usually done in parallel, i.e. each
gradient step uses both positive and negative phase updates. Recall the
algorithms in the deep learning book.

### Sampling
Sampling is not only relevant to generate new data points: We also need to sample
from the model for the negative phase in training.

Unfortunately, sampling is not so easy in undirected models: Since there is no
clear direction of dependencies between variables, it's not clear "where to start".
In Boltzmann Machines, sampling usually proceeds via _Gibbs sampling_: 
- Start by some arbitrary initial sample.
- One variable is sampled conditioned on all others, while the others are held fixed.
These conditional distributions are tractable, other than the full joint distribution
  (remember the partition function).
- Then, another variable is re-sampled, with the rest held fixed (the previously
updated first variable keeps the updated value). 
- This is done in turn, until all variables have been updated once, given the others.

This is _one step_ of Gibbs sampling. It can be shown that performing _many steps_
of this procedure will lead to samples from the model distribution, no matter what
the initial sample was like (under some mild conditions). Gibbs sampling is one
example of a _Markov Chain_ method.

In RBMs, Gibbs sampling is particularly efficient: As the `v` variables are
conditionally independent given the `h` variables, and vice versa, we can update
all `v` in parallel, and then all `h`. This means that one step of Gibbs sampling
only takes two conditional updates, instead of `n` (number of variables in total).

### Further Notes
- There are some "advanced" training algorithms:
  - _Contrastive Divergence_ initializes the Markov Chains in training not from
  random samples, but from data. This way, we can use shorter chains, as we start
  closer to the model distribution (if the model is already close to the data).
  The disadvantage is that the chains will likely not move too far from the data,
  and may fail to suppress "spurious modes", which can lead to bad model samples
  later.
  - _Persistent Contrastive Divergence_ starts the chains at the beginning of training,
  from random samples, but _persists_ these chains throughout training, whereas
  the "naive" method restarts the chains at each training step. Again, this means
  we can usually get away with shorter chains. A possible issue is that the model
  may update faster than the chain can catch up, in which case the samples are not
  accurate representations of the model distribution.

- The training algorithm is a so-called _Markov Chain Monte Carlo_ method (MCMC).
A Monte Carlo method is simply one that estimates an expected value via samples.
Here, we estimate the partition function gradient by _samples_ from the model
distribution. An MCMC method uses Markov Chains to draw these samples.


All following methods are generally _directed_ models.


## VAEs

In directed graphical models, _inference_ (`p(z|x)`) is intractable (collider case).
Also, computing the likelihood `p(x)` is intractable as it needs an integral over
the latent variables.
- Fun fact: These two issues are actually one and the same. Being able to compute
one of `p(x)` or `p(z|x)` allows for computing the other (in a directed model with
a structure like a VAE). This can be seen by applying Bayes' rule to `p(z|x)`.

Variational inference replaces `p(z|x)` by an approximation `q(z|x)`, which is
designed to be tractable (e.g. Gaussian with diagonal covariance). This also allows
for computation of the evidence (or variational) lower bound (ELBO), a lower bouhd
on `p(x)`.

- VAEs are likelihood-based models, at least approximately (we optimize ELBO)
- `ELBO = reconstruction_loss + KL-Divergence(q(z|x) || p(z))`
- In the deep learning view, this translates to an encoder-decoder architecture
where the encoder is the "inference network" that outputs `q(z|x)` and the decoder
is the "reconstruction network" that outputs `p(x|z`). The encoder is _stochastic_, i.e. it outputs
parameters of a probability distribution, and the latent codes are then sampled
from that distribution. The KL-D term can be seen as a regularizer that encourages
the encoder distributions to become close to the prior.
- Gaussians are a common choice for simplicity, but **not required**! If you think
a VAE always uses Gaussians, you did not fully understand the framework!
  - In particular, choosing Gaussian prior and posterior allows for computation of the
  KL-divergence in closed form. In general, KL-divergence is an expectation and would
  need to be estimated via samples, which generally leads to more variance and
  slower/worse training.
  - Another factor is that the chosen distribution needs to be amenable to the
  _reparameterization trick_: During training we need to sample from `q`, and
  sampling is not differentiable. Thus we need to somehow "go around" the sampling
  operation in backpropagation.
- Common problems include
  - Training issues such as posterior collapse (latent code `z` becomes uninformative
  about input, KL-divergence goes to 0)
  - Generations are often blurry and generic-looking, lacking detail
    - This issue likely also related to the commonly used element-wise reconstruction
    losses (e.g. squared error), which correspond to the assumption that elements
    in the data `x` (e.g. pixels) are conditionally independent given `z`.

Some advanced variants:
- beta-VAE (multiply KL-divergence by hyperparameter `beta > 1`) is supposed to give
better disentanglement in latent space
- Normalizing Flows (e.g. inverse autoregressive) in the latent space allow for more powerful (but still 
tractable) inference distributions `q`

### VQ-VAE
From the later "advanced techniques" session, this is an autoencoder with a
_discrete_ code space. Encodings are discretized to the closest vector in a fixed-size
_codebook_. This discretization is not differentiable, so learning proceeds via
the _straight-through estimator_, which simply funnels the gradients from the
decoder into the encoder, as if the discretization had never happened 
(it is treated as an identity operation).  
The codebook is learned in a very simple fashion; it is usually updated such that
the codebook vectors move closer to the encoder outputs which "belong" to each
respective codebook entry.

Some problems include:
- Straight-through estimator leads to inaccurate encoder updates
- "Codebook collapse" may occur, where parts of the codebook fall out of use
 and are no longer updated (can be tackled by resetting those vectors)
- In general, reconstructions will be worse due to the limited representational power of the
discrete latent space
- Using VQ-VAEs for generation requires another generative model (often autoregressive)
on the code space

VQ-VAEs are very powerful compression models, as we do not need to store the
encoding vectors, only the indices of the codebook entries, along with the codebook
itself. This had lead to the development of _neural codes_, compression algorithms
using neural networks, that can outperform established hand-crafted codecs like
JPEG, MP3, OPUS etc. in terms of quality and/or amount of compression.

## GANs

GANs implement a two-player game where one player (G) generates data, and the
other (D) classifies data as real or generated. D tries to do this successfully,
while G tries to "fool" D. This leads to a _minimax_ loss formulation where D
tries to maximize some loss term, while G tries to minimize it (or the other way
around, this is not critical). The goal is to end up in an _equilibrium_ where  G
generates the data distribution and D cannot tell apart real and generated data.

There is also a more theoretical approach to GANs as "implicit generative models".
These kinds of models do _no_ explicit modeling of `p(x)`; they only generate
samples directly. In this framework, GANs can be understood as optimizing certain
divergences or probability metrics, under strong assumptions such as a discriminator
with infinite capacity. Recall the chapter on GANs in the Murpy book. Popular variants include
- The original GAN with D being a binary classifier trained with cross-entropy loss
- _Least Squares GAN_ optimizing a simple squared error loss
- Hinge Loss
- Wasserstein GAN

**You should have a rough idea of this**. E.g. that different loss functions lead
to different metrics, such as the Jensen-Shannon-Divergence for the default GAN
loss, and that, in practice, the theory often doesn't hold as assumptions are not
met. No details are necessary.

Some problems:
- Training tends to be very unstable and difficult to scale
- Theory (D computes certain divergences etc.) often breaks down in practice
- Generations often contain unrealistic artifacts
- Other than VAEs, GANs cannot perform inference (getting `z` from `x`), unless
we introduce additional inference networks or use ad-hoc procedures such as
gradient-based optimization to find the "best" `z`

### StyleGAN
StyleGAN is an important architecture in modern GAN variants. You should have a rough
grasp of this. The main thing is that the latent code `z` gets pulled into a "side path"
where a "style network" computes a style vector `w`, which is then used to _modulate_
the activations in the "main branch" of convolutions. Modulation can be done via
adaptive normalization techniques. You should also be aware of the previous paper
on Progressive Growing (what's the main idea here).


## Autoregressive Models

Autoregressive models are very simple in terms of theory. They are graphical models
with no latent variables, and are represented by a _complete directed acyclic graph_.
Nodes can be brought in an order, and each node receives all previous nodes as input.
The same idea can also be derived via the _chain rule of probability_. It should
be noted that, although many types of data have a "natural" ordering (sequences),
the order in an AR model is, in principle, arbitrary. Any permutation of variables
is a valid ordering.

AR models allow for direct (and efficient) computation (and thus optimization) of the log-likelihood.
Their main disadvantage is that they are very slow in terms of generation. Samples
need to be generated _sequentially_, variable by variable, as each prediction
relies on the previous variables being available. Furthermore, their inductive bias
of sequential generation is questionable in many domains (e.g. images).

A common use of AR models is to use another model (autoencoder, often VQ-VAE)
to reduce the dimensionality of the data, and train an AR model on this smaller
space. AR models also form the basis of current large language models.


## Flows

Flows are another type of model that optimizes likelihood directly. The key here
is the _change of variables_ theorem.
- Define a "base distribution" that is easy to evaluate, such as a standard Gaussian
- Define an _invertible_ mapping from data space to this "simple space"
- The (log-)likelihood of the data can be computed in terms of the simple distribution
and the determinant of the Jacobian of the transformation

Generation proceeds by drawing a sample from the simple distribution, and applying
the inverse transformation to map to the data space.

Practical considerations:
- All transformations need to be efficiently (!) invertible
- The Jacobian determinant needs to be efficient to compute
  - In general, this is achieved through ensuring the Jacobian is a triangular matrix

You should know some of the ways to achieve this, e.g. through affine coupling
layers (e.g. NICE), (inverse) autoregressive flows, etc.


## Score-Based Models

_Score matching_ relies on the fact that, for two probability distributions,
matching their scores (gradient of the log probability) also results in matching
distributions. Thus we have a principled approach that does not rely on severe
approximations (VAEs), strong architectural or structural assumptions (Flows, AR)
or unstable adversarial losses (GANs).

Given a score model, sampling can be done by _Langevin Dynamics_, which follows
the score (like gradient ascent) plus an additional noise term for stochasticity.
It can be shown that this leads to samples from the distribution that the score
belongs to.

The big problem is that we do not have access to the true data score, and re-writing
the objective to get rid of it results in losses containing expensive terms such
as the trace of the Jacobian. In practice, we most often use _denoising score matching_.
Very roughly speaking, we replace the data distribution by a noisy version, which
extends the distribution on the entire space and allows for computation of data scores.

- In practice, small noise is not sufficient, as noisy samples will be close to the real
data and not fill out the space properly
  - This means the model will not learn proper score estimates in "empty" regions
  where there is no data
  - With inaccurate score estimates, sampling from the model is unlikely to succeed
- Large noise is also a no-go, since we are now matching the noisy data distribution,
which is not useful if the noise is too large

The solution is to do score matching at _multiple noise levels_. If successive noise levels
are close enough, the noisy distributions will overlap significantly. 
Then, sampling from the target distribution can
be done by successively sampling from decreasing noise levels: Each level gets us
into an area where the gradients of the next-lower level are accurate.

This process can be generalized to _infinitely many_ noise scales, leading to
_stochastic differential equations_. This framework also encompasses and further
generalizes diffusion models. No details are necessary here.


## Diffusion

While diffusion models are trained and sampled very similarly to score-based models,
they have a very different theoretical setup. In essence, they are _deep VAEs_
with many layers of latent variables (do not confuse this with the depth of a 
neural network). The variational inference distributions `q` (i.e. the encoders) are _fixed_ to Gaussians
that add a slight bit of noise to the previous layer. The number of steps and the
noise schedule are chosen such that the "deepest" latent variable is essentially
standard normal. This step-wise transformation of data into noise is often called
the _forward process_.

It can be shown that, under these conditions, the _reverse process_ is also approximately
Gaussian. What we are really doing in diffusion models is training this reverse
process, i.e. find the parameters of the Gaussian distributions going, step by step,
from noise to data. Only through various simplifications and reparameterizations
do we end up with the simple setup that is _equivalent_ to score matching:
- Fixing the variance of the reverse process Gaussians means we only need to predict
the _mean_ of the distribution.
- _Reparameterizing_ the model to predict the _noise_ instead of the mean
- simplifying the variational lower bound (typical VAE loss function) to receive
a re-weighted variational lower bound

Similarly, sampling looks like the Langevin sampler in score-based models, but we
are actually just sampling from the Gaussians at each step of the reverse process.


## Text-to-Image Models

At their core, text-to-image models are "just" conditional generative models, i.e.
models of `p(x|c)` where `x` is an image and `c` is text. It shouldn't hurt to have
an idea of how this can be achieved in practice, i.e. know the rough architecture
of 1-2 such models (DALL-E, GLIDE, Imagen, Latent Diffusion...). 

Currently, most
such models use a transformer to process the text into a sequence of latents, and
diffusion models conditioned on the text to generate the images. Conditioning is
usually implemented via attention. Many models make user of _cascaded diffusion_,
where a diffusion model first outputs a low-resolution image (e.g. 64x64), and then
one or more diffusion models trained on super-resolution output higher-res versions
(e.g. 256x256, 1024x1024). The success of these models seems to mostly depend on
scale (hundreds of millions of text-image pairs, billions of parameters).

One other thing you should be familiar with is the _guided diffusion_ method, e.g.
_classifier-free guidance_. Here, instead of using `e(x|c)` as an update, we use
`(1+w) * e(x|c) - w * e(x)` (`e` denotes the diffusion model noise prediction). Note that this makes use of the equivalence between
the diffusion sampling process and score-based models.


## Large Language Models

LLMs are not particularly "exciting" from a model perspective. All current LLMs
are just _very_ large autoregressive models using Transformers. You should have a
rough idea of the scale at play here (up to 100s of billions of parameters currently).
Aside from that, there are some other interesting aspects:

### Scaling Laws
Research shows that, empirically, autoregressive modeling performance improves
_predictably_ as a function of model size, dataset size, and amount of compute.
Many architectural details, such as depth vs width, number of attention heads etc.,
have a relatively minor impact on performance.
This is a key result that justified the "just go big" method of LLMs. 

There are more takeaways, e.g. that large models can actually be more efficient than
small ones (they reach better losses with the same compute), and so training
large models without converging can be preferred to training small models to
convergence. It was also shown that Transformers outperform LSTMs as they can
make better use of larger context sizes (LSTMs quickly plateau and seem unable
to use large contexts).

You do not need to know details about the power law formulas etc. -- focus on the
main takeaways.

### Fine-tuning vs Few-shot Learning
The first GPT system served as a base system that could be fine-tuned on specific
tasks by training it further. Later systems showed that LLMs can already perform
well on many tasks _without_ training; we just need to provide good prompts
(e.g. zero-shot vs one-shot vs few-shot).

However, it should be noted that fine-tuned models still outperform the "generalist"
few-shot approach.

### RLHF
_Reinforcement Learning with Human Feedback_ is a key technique in systems like 
ChatGPT. Recall that autoregressive models generally just predict the next token in
a sequence, i.e. they are just "autocomplete" methods. To make them usable for dialog,
they need to be re-trained specifically for that purpose. We would also like to take
measures to make them more "aligned" with human expectations and values, i.e. refusing to answer
inappropriate prompts, being as unbiased as possible etc. RLHF proceeds as follows:
- Recruit a number of "human chatbots" that design prompts and provide answers as well.
- Fine-tune a language model on these question-answer style prompts. This already
helps with moving away from autocomplete to a question-answer format.
- Recruit people to _rank_ several model answers for a given prompt. This gives an idea
of which answers people prefer.
- Train a _reward model_ that mimics human preferences.
- Further fine-tune the language model using the reward model. This is done via
reinforcement learning.

### LoRA

_Low-rank adaptation_ is a technique to more efficiently fine-tune LLMs. In fine-tuning,
we have a base model given with weights `W0`, and adapt them to new weights `W`.
We may write `W = W0 +dW` where `dW` is the change done during fine-tuning.

Literature suggests that, in large neural networks, weight matrices are often _rank-defficient_,
i.e. their actual rank is much lower than the maximum possible. The hypothesis is that
the weight updates `dW` may also be low-rank.

Say we have an `m x n` weight matrix. In LLMs, these dimensions may be on the order
of thousands, possibly tens of thousands. We can write `dW = A * B`, where `A` is
`m x r` and `B` is `r x n`, with `r` much smaller than `m` or `n`.
The matrix product is again of shape `m x n`. However, the rank will be at most `r`.
Also the total number of parameters is `m*r + n*r`, which for small `r` is much smaller
than `m*n` in the original matrix.

By learning `A` and `B` instead of `dW`, we thus save many, many, many parameters.
In practice, even single-digit ranks can be sufficient to reach decent fine-tuning
performance.

Note that we still need to be able to run the model itself, which can require huge
amounts of computational resources. However, reducing the amount of trainable parameters
in fine-tuning still helps: Modern optimizers like Adam store several momentum
terms for all weights, making the number of parameters stored during training
3x the model parameters. With LoRA, these momentum terms need only be stored for
the `A, B` matrices, possibly reducing parameter storage by a factor of almost 3
(as only the actual model parameters need to be stored).


## Evaluating Generative Models

Evaluating generative models is difficult. 

- Likelihood is often expensive or impossible to compute
- Estimates of likelihood (e.g. Parzen windows) are very inaccurate in practice
- Subjective evaluation of sample quality is expensive (need "test subjects")
- Different measures can be essentially independent
  - e.g. models with great likelihood but terrible samples, or the other way around
- Thus, evaluation always needs to be done with specificity for the factor of interest
  (need good samples? Evaluate the samples. Etc.)

### Inception Score & FID

IS and FID are two measures that are still commonly being used. They (supposedly)
correlate well with human judgement of sample quality, and so can be used as a
cheap automated replacement.

You should understand how both measures are defined. Both need an external pre-trained
classifier.

- Inception score uses classifier probabilities only.
  - The conditional distributions `p(y|x)` should be as "peaked" as possible, i.e.
  high probability for a single class.
  - The overall distribution `p(y)` should be as uniform as possible, i.e. different
  generations are varied.
- FID uses hidden layer outputs.
  - The generated feature distribution should be as close as possible to the true
  data distribution.
  - A Gaussian distribution is assumed (i.e. mean and covariance are fit for bot
  real and generated distributions).
  - The _Fischer divergence_ is used to compute the distance.

In practice, both IS and FID have issues. They can be "gamed" to achieve high scores,
and it has been demonstrated that they do not correlate with human judgements in
some cases. Finding better measures for evaluation is an active research problem.
