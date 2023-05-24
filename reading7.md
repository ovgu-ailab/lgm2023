---
layout: default
title: Reading 7
id: reading7
---


# Reading Assignment 7: Normalizing Flows

## Overview

The below are multiple options for getting the general idea of normalizing flows,
along with an overview over different methods.

- An approachable introduction to normalizing flows can be found in 
[this blog post](https://lilianweng.github.io/posts/2018-10-13-flow-models/) by
Lilian Weng. You can skip the parts on autoregressive models (PixelRNN, Wavenet).
- Next, [this two-part blog post](https://blog.evjang.com/2018/01/nf1.html) by
Eric Jang provides another view; especially the discussion on the meaning
of the Jacobian determinant in the first part can be a nice addition. A lot of
information will be a repetition from the first blog. You can/should skip the
code sections; they are heavily outdated.
- Finally, [Murphy's book](https://probml.github.io/pml-book/book2.html) has an
in-depth chapter devoted to normalizing flows.
- If you need a refresher, on the substitution rule for integrals,
[the Wikipedia article](https://en.wikipedia.org/wiki/Integration_by_substitution)
should be sufficient. The _change of variable theorem_ used for normalizing flows
is the multi-dimensional version of this.

## Specific Models

This can be considered optional reading, but try to read at least one of these to
get a detailed view on a specific flow model.

- [NICE](https://arxiv.org/pdf/1410.8516.pdf): A relatively simple model and not very
powerful, but it arguably lead the groundwork for all deep flow models.
- [RealNVP](https://arxiv.org/pdf/1605.08803.pdf): A massive step up from NICE.
- [Glow](https://arxiv.org/pdf/1807.03039.pdf): Yet another iteration on RealNVP,
using invertible 1x1 convolutions.
- [Parallel Wavenet](https://arxiv.org/pdf/1711.10433.pdf): A neat application of
IAFs for efficient Wavenet sampling.
- [A whole other usage](https://arxiv.org/pdf/1606.04934.pdf) 
of normalizing flows is to improve the variational posteriors in VAEs.
