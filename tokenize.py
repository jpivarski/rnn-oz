#!/usr/bin/env python

import glob
import re

import numpy

wordPattern = re.compile(r"^([A-Za-z]+'[A-Za-z]+|[A-Za-z]+|[0-9]+)")
paraPattern = re.compile(r"^ *\n\n+ *")
whspPattern = re.compile(r"^( *\n *| +)")
dashPattern = re.compile(r"^\-{2,}")

def match(pattern, subtext, result):
    m = re.match(pattern, subtext)
    if m is None:
        return False
    else:
        result[0] = m.group()
        return True

def tokenize():
    for fileName in glob.glob("inputs/*.txt"):
        text = open(fileName).read()
        position = 0
        while position < len(text):
            subtext = text[position:]
            result = [None]
            if match(wordPattern, subtext, result):
                yield result[0]
            elif match(paraPattern, subtext, result):
                yield "\n\n"
            elif match(whspPattern, subtext, result):
                yield " "
            elif match(dashPattern, subtext, result):
                yield "---"
            else:
                result[0] = subtext[0]
                yield result[0]
            position += len(result[0])

dictionary = set()

for token in tokenize():
    dictionary.add(token)

dictionary = sorted(dictionary)
dictionaryLookup = {x: i for i, x in enumerate(dictionary)}

hiddenNeurons = len(dictionary)
Wxh = numpy.random.randn(hiddenNeurons, len(dictionary)) * 0.01   # input to hidden
Whh = numpy.random.randn(hiddenNeurons, hiddenNeurons) * 0.01     # hidden to hidden
Why = numpy.random.randn(len(dictionary), hiddenNeurons) * 0.01   # hidden to output
bh = numpy.zeros((hiddenNeurons, 1))                              # hidden bias
by = numpy.zeros((len(dictionary), 1))                            # output bias

testSequence = []
for token in tokenize():
    if token != " " and token != "\n\n":
        testSequence.append(token)
    if len(testSequence) == 100:
        break

# def lossFunction(inputs, targets, hprev):
if True:
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = numpy.copy(hprev)
    loss = 0.0
    # forward pass
    for t, (input, target) in enumerate(zip(inputs, targets)):
        print "forward", t, input, target,
        # input vector (indicator)
        xs[t] = numpy.zeros((len(dictionary), 1))
        xs[t][dictionaryLookup[input]] = 1.0
        # hidden state
        hs[t] = numpy.tanh(numpy.dot(Wxh, xs[t]) + numpy.dot(Whh, hs[t - 1]) + bh)
        # output state (unnormalized log probabilities for the next token)
        ys[t] = numpy.dot(Why, hs[t]) + by
        # probabilities for the next token
        ps[t] = numpy.exp(ys[t]) / numpy.sum(numpy.exp(ys[t]))
        # how close was the prediction to the truth?
        loss += -numpy.log(ps[t][dictionaryLookup[target], 0])
        print "most probable", numpy.max(ps[t]), dictionary[numpy.argmax(ps[t])]
    print "total loss", loss
    # derivatives for backpropagation
    dWxh = numpy.zeros_like(Wxh)
    dWhh = numpy.zeros_like(Whh)
    dWhy = numpy.zeros_like(Why)
    dbh = numpy.zeros_like(bh)
    dby = numpy.zeros_like(by)
    dhnext = numpy.zeros_like(hs)
    for t, target in reversed(list(enumerate(targets))):
        print "backward", t, target
        # difference between predicted probabilities (unit vector) and true probabilities (indicator)
        dy = numpy.copy(ps[t])
        dy[dictionaryLookup[target]] -= 1.0
        # derivatives
        dWhy += numpy.dot(dy, hs[t].T)
        dby += dy
        dh = numpy.dot(Why.T, dy) + dhnext  # backpropagation onto h
        dhraw = (1.0 - hs[t] * hs[t]) * dh  # through tanh nonlinearity
        dbh += dhraw
        dWxh += numpy.dot(dhraw, xs[t].T)
        dWhh += numpy.dot(dhraw, hs[t - 1].T)
        dhnext = numpy.dot(Whh.T, dhraw)
    for dparam in dWxh, dWhh, dWhy, dbh, dby:
        numpy.clip(dparam, -5, 5, out=dparam)
    print loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs) - 1]

hprev = numpy.zeros((hiddenNeurons, 1))

loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFunction(testSequence[:-1], testSequence[1:], hprev)
