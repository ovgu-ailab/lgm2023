---
layout: default
title: Assignment 10
id: ass10
---


# Assignment 10: Conditional Generation & Guided Diffusion
**Discussion: June 29th**  
**Deadline: June 29th, 9am**

For this assignment, you will need a working diffusion model implementation. Either
- Use your own one (if it works ;))
- Find one on "the internet"
- Use the one uploaded on Gitlab [here](https://code.ovgu.de/ai-lab/teaching/lgm/learning-generative-models-2023/-/blob/main/assignment09_small.ipynb)

The  latter is recommended as it is self-contained and extendable.

The Gitlab code should work as-is! If it doesn't, please ask as it is most likely
not your fault!


## Conditional Generative Models

A conditional model is a model of `p(x|c)` instead of just `p(x)`, where `c` is
some conditional information. A very straightforward kind is a class-conditional
model, where you can supply a class as "input" and get a generation of that class.
Let's start by implementing one of those.

- Choose a dataset with conditioning information. Straightforward choices are MNIST,
FashionMNIST or CIFAR, in ascending order of difficulty.
- Adapt your model to be class-conditional. You need to "somehow" inject the class
information into the model. Recall the discussion in the exercise. Furthermore,
the provided diffusion model is already time-conditional. You can add class conditioning
in pretty much the same way, i.e. project the class index to a vector (e.g. `Embedding`)
and then insert it into the hidden layers through operations such as adaptive normalization.
- Change the generation code (Langevin sampler) to take a vector of class indices
as input, which are provided to the model at each sampling step.

Train a conditional model and make sure that it's generating appropriate samples
given class inputs!


## Guided Diffusion

Guided Diffusion has been shown to provide a quality-variety tradeoff that can
be tuned as desired. There are two kinds of guidance: Classifier guidance and
classifier-free guidance. The latter is more popular and easier to implement. The
score changes from `gradient(log(p(x|c)))` to `(1+w) * gradient(log(p(x|c))) - w * gradient(log(p(x)))`.
Diffusion models don't implement a score directly, but the common implementation is
equivalent. So in the Langevin sampler, just change `model(x,c)` to `(1 + w) * model(x,c) - w * model(x)`.
`w` is a hyperparameter. Some special cases are:
- `w = -1` results in unconditional sampling.
- `w = 0` results in a standard conditional model.
- `w > 0` enables guidance. Normally `w >= 1`.

We just have one problem: We trained a conditional model, but now we need an _unconditional_
model as well. Luckily, we can get both at the same time.

- During training, randomly _drop out_ the conditioning information. This effectively
trains an unconditional model. Common choices are somewhere between 5-20% dropout rate.
Maybe try 10%.
You can either randomly drop out an entire batch (easier to implement) or randomly
drop elements within each batch (probably lower variance).
- There are different ways to achieve this dropping out:
  - If you are providing class information as a one-hot vector, turn it into an all-0
  vector instead.
  - If you are providing class indices, you can introduce an additional class that
  acts as "no class". E.g. you could shift all classes up by 1 (0->1, 1->2 etc)
  and use class index 0 as "no class". This seems counter-intuitive, but because
  this 0 class is introduced randomly instead of the actual class, for any input,
  this really acts as a "no-class class".


A simple way do achieve this dropout could be like this (it's basically a binary mask):
```python
drop_prob = 0.1  # for example
# you may need to choose dtype=tf.float32 or dtype=tf.int32 depending on condiitoning
drop_dist = tfp.distributions.Bernoulli(probs=1-drop_prob, dtype=??)

conditioning = ... # e.g. class labels as one_hot or indices
drop_sample = drop_dist.sample(tf.shape(conditioning))
dropped_conditioning = drop_sample * conditioning
```

Otherwise, training doesn't change from the normal conditional model.

Finally, as mentioned further above, adapt the Langevin sampler to provide both
conditional _and_ unconditional model outputs (just run the model twice, once
with conditioning, once with no/dropped conditioning). Experiment with different values
of `w`. Common values for good samples are in the region of 3-10. Do you find that
quality increases as opposed to `w = 0` (no guidance)? What happens as you increase
`w` further?
