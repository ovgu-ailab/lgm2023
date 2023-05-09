---
layout: default
title: Reading 5
id: reading5
---


# Reading Assignment 5: Evaluating Generative Models

There is no one "correct" method to evaluate generative models. First off, you
can get an overview over different approaches from the following articles:
- The method of [Parzen windows](https://www.milania.de/blog/Introduction_to_kernel_density_estimation_%28Parzen_window_method%29)
is relatively straightforward, though outdated.
- The [inception score](https://medium.com/octavian-ai/a-simple-explanation-of-the-inception-score-372dff6a8c7a)
evaluates the class distribution of generated images and is thus of limited applicability.
- The [Fréchet Inception Distance](https://en.wikipedia.org/wiki/Fr%C3%A9chet_inception_distance)
is currently the most used metric. You don't need to understand the theory in full,
but try to grasp the formula for the Gaussian case. Another article, including
code, can be found [here](https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/).

Aside from that, there is [a very influential paper](https://arxiv.org/pdf/1511.01844.pdf)
which shows how problematic evaluation really is, e.g. in that different metrics
can be effectively independent.

Finally, [here is an interesting article](https://sander.ai/2020/09/01/typicality.html)
on the concept of _typicality_ and how it relates to likelihood. All in all, the
latter two articles show that maximum likelihood may not be the ideal goal for 
training generative models.
