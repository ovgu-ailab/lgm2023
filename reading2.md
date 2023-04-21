---
layout: default
title: Reading Assignment 2
id: reading2
---


# Reading Assignment 2: Partition Function & RBMs

From the [Deep Learning Book - Chapter 18: Confronting the Partition Function](http://www.deeplearningbook.org/contents/partition.html), please read these sections:

* Sections 18 (intro), 18.1, 18.2 with focus on:
  * What are positive and negative phase of learning?
  * Which role does the partition function play in learning generative models?
  * How do we avoid to compute the partition function directly and why is it possible to do this?
  * How does Contrastive Divergence (CD) work and what (dis)advantages does this method have?
  * How does Persistent Contrastive Divergence (PCD) work and what (dis)advantages does this method have?
  * What is the relation between PCD and Stochastic Maximum Likelihood?
  * Optional, Section 18.3: What is the idea behind Pseudolikelihood?

* Sections 18.7 & 18.7.1 (optional, low priority):
  * What are possible ways of estimating the partition function?
  * What is the idea of Annealed Importance Sampling (AIS)?
  * Note: You will probably need to read the section on Importance
  Sampling (Chapter 17.2) to understand this.

For the actual application of these techniques, please read these sections from the [Deep Learning Book - Chapter 20: Deep Generative Models](http://www.deeplearningbook.org/contents/generative_models.html):

* Sections 20 (intro), 20.1, 20.2 with focus on:
  * What are the Energy functions of a Boltzmann Machine (BM) and a Restricted Boltzmann Machine (RBM)?
  * How would the partition function of an RBM be computed?
  * What makes sampling in an RBM efficient?
  * What is the sampling procedure and how does it relate to MCMC?
  * How can BMs and RBMs be trained?
  * Where are the connections between training an RBM and the concepts of chapter 18 (partition function, CD/PCD, AIS)

## Optional: DBMs and DBNs

If you are interested, you can also read sections 20.3 on Deep Belief Networks,
which incorporate _both_ directed and undirected connections, and section 20.4
on Deep Boltzmann Machines, a deep undirected model. Beyond that, sections 20.4-20.8
discuss other Boltzmann Machine variants (such as Gaussian BMs for real-valued data).
These topics will not be treated in this class.
