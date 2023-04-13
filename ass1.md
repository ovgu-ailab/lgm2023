---
layout: default
title: Assignment 1
id: ass1
---


# Assignment 1: Probability Review
**Discussion: April 20th**  
**Deadline: April 20th, 9am**


## Testing Illnesses
Consider a dangerous and/or common illness that people are being tested for to
recognize it early (e.g. cancer) and/or prevent its spread (e.g. COVID). The test
is either positive or negative. We make the following assumptions:

- About 1% of the population has the illness. That is, any given person has a
1% "a priori" probability of being sick.
- If a sick person is tested, the test returns a positive result 99.9% of the 
  time.
- If a healthy person is tested, the test still returns a positive result 1% of the 
  time.

You take part in a study where a random, representative sample of the population
is tested for the illness. Your test result is positive. What is the probability
that you have the illness?
1. **(Submission)** Solve this via simulation. 
   1. Take a "population sample" of a specific size
   (experiment with different sizes!) where every "generated person" has a 1% 
    chance of turning out sick.  
   2. Test your "people" -- if they are sick, the test should have a 99.9% chance
    of returning a positive result; if they are healthy, it should be 1%.
   3. Out of all people that have been _tested sick_, get the proportion of people
    that _are actually sick_.
2. **(Submission)** Solve this via mathematics. This requires a basic grasp of marginal and conditional 
   probabilities as well as Bayes' theorem. These are fundamental concepts without which
   you will be lost in this class! [The corresponding wiki article](https://en.wikipedia.org/wiki/Bayes%27_theorem)
   should be sufficient.
   
Next (mathematical solution is sufficient, no need for more simulation):
1. **(Submission)** Conversely, assume the test result is negative. What is the probability that
you have the illness anyway?
2. **(Submission)** To bullet-proof the results of their study, the researchers decide to 
   administer _two_ tests to each participant. The second test has the following
   properties:
   - If a sick person is tested, the test returns a positive result 96% of the
    time.
   - If a healthy person is tested, the test still returns a positive result 2%
    of the time.  
     
   As we can see, the second test is much more prone to errors than the first.
   However, assume that the results of the second test are _independent_ of the
   first. That is, whether the second test makes an error does not depend
   on whether there is an error on the first test and vice versa.  
   Now, _both_ of your tests come back positive. Given this information,
   what is the probability that you are indeed sick?
   

## Modeling Waiting Times

The purpose of this part is for you to walk through a basic probabilistic 
modeling task yourself.

Say you are at the doctor or maybe a government office and are waiting for your
appointment. You would like to get an estimate of how long you will have to wait. The
best way seems to be to base this on other people's waiting times.


First, we need to decide how to model the data. This essentially boils down to
finding a sensible probability distribution. Consider questions such as
- Can waiting times be negative? Can they be zero?
- Can they be whole numbers only, or real numbers (there might not be one correct
response here)?
- Should the distribution be unimodal, or something else?
- ...

An internet search should help you here; it's a good idea to choose a well-known
"standard" distribution, as these are often relatively easy to work with.

Derive the maximum likelihood solution for the
parameter(s) of your model/distribution given the data. To do this, compute the
log-likelihood of the dataset to receive a term that can be maximized. Next, you
can use basic calculus to derive a solution for the optimal parameters analytically.


To try out our model, we need some data. You have two options:
1. Make up some data yourself by hand, i.e. just write a list with some numbers.
2. Randomly generate waiting times. This way, you can generate a lot more data.


If you decide to generate data (option 2 above), choose some parameters for your
distribution and draw random samples -- if your distribution is not too exotic, there
should be functions e.g. in `numpy` or `scipy`.

Finally, compute your model solution for the data. You can use your analytical solution
and/or use gradient ascent to iteratively
arrive at a solution. Does the solution make sense to you? In particular, you might
compare measures such as the expected value of the distribution with quantities such as
the mean or median of the dataset. In case you generated data from a distribution,
you can also check if your derived result matches the actual distribution parameters.

**Submission**: You should include a write-up of your modeling decisions as well
as the derivation of the maximum likelihood solution. Also include any experiments
you conduct with the toy data you created (e.g. finding the optimal parameter values).  
You can do the mathematical derivations (also for part 1) on paper
and upload a scan/photograph, typeset using markdown or latex, or just sketch
them in a Python notebook.
