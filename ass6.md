---
layout: default
title: Assignment 6
id: ass6
---


# Assignment 6: Autoregressive Language Modeling
**Discussion: June 1st**  
**Deadline: June 1st, 9am**  

By far the most "hyped" AI application right now is generating natural language
(ChatGPT). The good news is that state-of-the-art systems are conceptually very
simple, relying on sequential autoregressive generation. The bad news is that they
require _massive_ scale.

In this assignment, we will build such a system on a small scale to understand the
underlying principles, and have some fun with the terrible generations we will probably
get.

Tutorials for this are plentiful:
- If you haven't done so, read [this blog post](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
for an illustrative walkthrough on how to train a character-based RNN model.
- [Tensorflow Text](https://www.tensorflow.org/text/tutorials/text_generation) has
effectively provides a Tensorflow version of the above.
- There are further code examples [on the Keras website](https://keras.io/examples/generative/text_generation_with_miniature_gpt/) --
you can find more than the one linked above in the navigation on the left.

As usual, please do not just copy a tutorial and call it a day; you will not learn
anything. If you want to roll your own version, below is a rough summary of the
necessary steps:


## Preparing Data

### Finding a Dataset
You have LOTS of options here. Some of the tutorials linked above also provide
data. A common tutorial choice are things like the collected works of Shakespeare
or the IMDB dataset. However, these are tiny by modern standards, and you will
likely not get a good model from such datasets. Another option could be
[Tensorflow Datasets](https://www.tensorflow.org/datasets/catalog/overview) --
check the "Text generation" tab in the catalogue, and the guide for how to use
the library.

### Tokenization
Independently of the dataset, you will have to settle for a level of representation.
Do you want to model characters, words, or something inbetween? You will have to 
process the data, which is usually provided as raw strings, into sequences of tokens.
- Characters are easiest to set up: Every character is one token. In Python, strings
are already sequences of characters, so this is easy to deal with. You will also
have a very small token vocabulary, which makes running the model a bit easier.
- Words can be a bit tricky to tokenize. Recall the [Machine Translation tutorial](https://www.tensorflow.org/text/tutorials/nmt_with_attention)
we used in IDL -- this includes code to do word-level tokenization. However, this
also does stuff like lowercasing everything, which you might not want. Also words
lead to a very large vocabulary, which can make training slow -- only keeping
the most common words and replacing all else with "UNKNOWN" (as we did in the
translation example) is not really desirable for text generation.
- Byte-pair encoding may be a good compromise, but you will need code for this.
Maybe one of those tutorials can help...

You should also convert your tokens to indices by a vocabulary (token-to-index mapping)
that you will also need to create.

### Making Sequences
Obviously, you can't put the entire dataset into your model in one go -- you will
have to divide into smaller subsequences that can be treated as single training examples.
There are basically two choices:
- Some "principled" division, like one sequence per sentence, or paragraph, or
movie review, or... This can be more satisfying on theoretical grounds, but it
also _massively_ complicates things: You will have to deal with uneven sequence
lengths (padding, masking), potentially very long sequences, etc.
- Simply dividing into sequences of a fixed length. This is recommended, as it makes
training so much simpler. You can choose a length yourself; usually a few hundred
tokens. Longer sequences make training harder, but let your model learn longer-term
dependencies. Note: Just because you are _training_ on a certain length, doesn't
mean your model can't _generate_ longer sequences later!

### Building a Model
This part is actually quite simple! You essentially need three components:
1. Embedding layer for your tokens.
2. A sequence model. You will probably want to start with an RNN, e.g. a GRU or
LSTM. Later, if you want, you could also attempt a Transformer.
3. A (softmax) output layer with one class per vocabulary entry.

The model needs to be trained to predict the next token given the previous ones.
One step of training proceeds as follows:
1. Get a batch of sequences.
2. Put the batch in your model to get a batch of output sequences.
3. Compute the cross-entropy between the outputs and the targets. This can be done
on the entire sequence at once; it will compute the cross-entropy per-timestep
and average over time.

What are the targets? For any time step, it should always be the bext token. This
can be achieved as follows:

```python
inputs = sequences[:-1]
targets = sequences[1:]
```

Inputs exclude the final token (doesn't predict anything); targets exclude the
first token (not predicted by anything). This aligns inputs and targets in such
a way that at each time step, the target is the next token.

### Generating Text

Having trained your model, generation proceeds as follows:
- Provide a _prompt_ to your model, i.e. an initial part of a sequence. If you
are using an RNN, you will need to run the model over the prompt to get the hidden
state after seeing it. A Transformer doesn't need to do this.
- Get the probability for the next token from the output layer. Sample from this
distribution (e.g. `tfp.distributions.Categorical`). Put the sampled token back into
your network, get the probability for the next token etc. Continue for some number
of steps. What happens if you use the _highest_ probability (argmax) instead
of sampling?
- You can also generate unprompted, but you will need _something_ to get the probability
over the first token. For example, you could insert special `START` tokens into
your data (e.g. at the beginning of every movie review, paragraph, etc.) and
provide that as input -- the model can then learn what should come after a `START`
token and generate a reasonable beginning itself.

**Note:** If using an RNN, you will need to "conserve" the state after inputting each
new token, as you obviously can't just apply it to the sequence (it doesn't exist yet).
As such, you should set up your model such that it returns the current state along
with the outputs, as well as take a state as an argument alongside the input.
The Tensorflow RNNs provide functionality for this:
- `return_state=True` will return the state at the end.
- `call` an takes `initial_state` argument.
- You can also use a `stateful` RNN, but this is a bit awkward and not recommended.
