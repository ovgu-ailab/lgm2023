---
layout: default
title: Reading 8
id: reading8
---


# Reading Assignment 8: Score-based Models

Read [this blog by Yang Song](https://yang-song.net/blog/2021/score/),
who first proposed deep generative models based on score matching. You can skip
the section on Stochastic Differential Equations, but note that towards the bottom,
there is a subsection "Controllable generation for inverse problem solving" which can
be interesting.

Unfortunately, the blog skips details on how to actually implement tractable score
matching objectives, which are needed for practical implementations. These details
can be found in [Murphy's book](https://probml.github.io/pml-book/book2.html):
Section 24.3 introduces denoising and sliced score matching (skip the relation
to contrastive divergence section), while section 25.3 gives the (surprisingly
simple) objectives used in practice.

Finally, if you want to get the full picture, this is the "classic" series of
papers by Yang Song:
- [Original paper](https://arxiv.org/pdf/1907.05600.pdf) introducing the method
- [Follow-up](https://arxiv.org/pdf/2006.09011.pdf) significantly improving it
- [Paper on SDEs](https://arxiv.org/pdf/2011.13456.pdf) generalizing the method
to continuous time scales