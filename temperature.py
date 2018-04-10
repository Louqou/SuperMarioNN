import sys
import numpy

preds = [0.1, 0.1, 0.1, 0.1, 0.6]

def sample(preds, temperature):
  # helper function to sample an index from a probability array
  preds = numpy.asarray(preds).astype('float64')
  preds = numpy.log(preds) / temperature
  exp_preds = numpy.exp(preds)
  return exp_preds / numpy.sum(exp_preds)
  # probas = numpy.random.multinomial(1, preds, 1)
  # return numpy.argmax(probas)

new_preds = sample(preds, 1.0)
print new_preds
print sum(new_preds)