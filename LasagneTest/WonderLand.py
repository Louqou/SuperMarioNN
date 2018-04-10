# The following code was written with reference to:
# http://colinraffel.com/talks/hammer2015recurrent.pdf [Accessed: 6 Feb 2017]
# http://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/ [Accessed: 6 Feb 2017]
# https://github.com/Lasagne/Recipes/blob/master/examples/lstm_text_generation.py [Accessed: 6 Feb 2017]
import numpy
import lasagne
import theano
import time

SEQ_LEN = 40
HIDDEN_UNITS = 512
BATCH_SIZE = 450
EPOCHS = 100
training = False

# get the training text and remove any unwanted characters
text = open("alice.txt", "r").read()
text = text.lower()
unwanted = ['\x80', '\x98', '\x99', '\x9c', '\x9d', '\xe2', '[', '_']
for c in unwanted:
  text = text.replace(c, '')

# get all the unique characters
chars = sorted(list(set(text)))
vocabSize = len(chars)
textSize = len(text)

charToInt = {c:i for i,c in enumerate(chars)}
intToChar = {i:c for i,c in enumerate(chars)}

# Created with referece to gen_data in:
# https://github.com/Lasagne/Recipes/blob/master/examples/lstm_text_generation.py [Accessed: 6 Feb 2017]
def generateData(startPoint, data, batchSize, getExpected):
  # for each batch will store the sequence of one hot arrays that resembles each character
  inputSeq = numpy.zeros((batchSize, SEQ_LEN, vocabSize), dtype='int32')
  # what character should be predicted next as an index in the one hot array
  expectedOutput = numpy.zeros(batchSize, dtype='int32')
  
  # for each batch in each sequence set the value in the one hot array
  # at the end of the sequence get the next character
  for batch in range(batchSize):
    for seq in range(SEQ_LEN):
      inputSeq[batch, seq, charToInt[data[startPoint+batch+seq]]] = 1

    if getExpected:
      expectedOutput[batch] = charToInt[data[startPoint + batch + SEQ_LEN]]

  if getExpected:
    return inputSeq, expectedOutput
  else:
    return inputSeq 

# Created with reference to:
# http://colinraffel.com/talks/hammer2015recurrent.pdf [Accessed: 6 Feb 2017]
# https://github.com/Lasagne/Recipes/blob/master/examples/lstm_text_generation.py [Accessed: 6 Feb 2017]
def main():

  # input layer of the network
  inputLayer = lasagne.layers.InputLayer(shape=(None, SEQ_LEN, vocabSize))

  # how the gate parameters should be initialised  
  gateParameters = lasagne.layers.recurrent.Gate(
    W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(),
    b=lasagne.init.Constant(0.))

  # how the cell parameters should be initialised
  # cell to cell connection doesn't use weights
  cellParameters = lasagne.layers.recurrent.Gate(
    W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(),
    W_cell=None, b=lasagne.init.Constant(0.),
    nonlinearity=lasagne.nonlinearities.tanh)

  # an lstm layer, clips gradients to avoid exploding gradient problem
  # will also learn its initial values
  lstmLayer1 = lasagne.layers.recurrent.LSTMLayer(
    inputLayer, HIDDEN_UNITS,
    ingate=gateParameters, forgetgate=gateParameters,
    cell=cellParameters, outgate=gateParameters,
    learn_init=True, grad_clipping=100.)

  # another layer, same as the previous
  # only returns final results so a slice/reshape layer is not needed
  # only interested in the final prediction of the sequence
  lstmLayer2 = lasagne.layers.recurrent.LSTMLayer(
    lasagne.layers.DropoutLayer(lstmLayer1, 0.5), HIDDEN_UNITS,
    ingate=gateParameters, forgetgate=gateParameters,
    cell=cellParameters, outgate=gateParameters,
    learn_init=True, grad_clipping=100.,
    only_return_final=True)

  # takes the output from the last layer, of size BATCH_SIZE, HIDDEN_UNITS
  # will output BATCH_SIZE, VOCAB_SIZE
  # softmax used to create a probability distribution
  outputLayer = lasagne.layers.DenseLayer(
    incoming=lasagne.layers.DropoutLayer(lstmLayer2, 0.5), 
    num_units=vocabSize, 
    nonlinearity=lasagne.nonlinearities.softmax)

  targetOutput = theano.tensor.ivector('target_output')

  if training:
    networkOutput = lasagne.layers.get_output(outputLayer)
  else:
    networkOutput = lasagne.layers.get_output(outputLayer, deterministic=True)

  # using cross entropy for the loss and take the mean across the batch
  cost = lasagne.objectives.categorical_crossentropy(networkOutput, targetOutput).mean()
  # get all the parameters from the network for training
  allParams = lasagne.layers.get_all_params(outputLayer)

  # how to update the gradients 
  updates = lasagne.updates.adam(cost, allParams)

  # trains the network
  train = theano.function(
    [inputLayer.input_var, targetOutput],
    cost,
    updates=updates,
    allow_input_downcast=True)

  # compute the cost
  computeCost = theano.function(
    [inputLayer.input_var, targetOutput],
    cost,
    allow_input_downcast=True)

  # will be used to get the most likely character
  probDist = theano.function([inputLayer.input_var], networkOutput, allow_input_downcast=True)

  # Created with reference to:
  # https://github.com/Lasagne/Recipes/blob/master/examples/lstm_text_generation.py [Accessed: 6 Feb 2017]
  def predictText(phrase, numPredict):
    assert(len(phrase) >= SEQ_LEN)
  
    netInput = generateData(len(phrase) - SEQ_LEN, phrase, 1, False)
    predictedCharsIndex = []
  
    for c in range(numPredict):
      probablities = probDist(netInput).ravel()
      #charIndexChoices = numpy.arange(vocabSize)
      
      # find index of highest probability
      #charIndex = numpy.argmax(probablities)

      # sample next char from the probability distribution
      #charIndex = numpy.random.choice(a=charIndexChoices, p=probablities)

      # sample using temperature
      charIndex = sample(probablities, 0.1)
      
      predictedCharsIndex.append(charIndex)
      # shift the sequence
      netInput[:, :-1, :] = netInput[:,1:,:]
      # set last char in the sequence to zero
      netInput[:, -1, :] = 0
      # input the character at the end of the sequence by setting the one hot value
      netInput[0, -1, charIndex] = 1.

    generatedText = phrase + ''.join(intToChar[charIndex] for charIndex in predictedCharsIndex)
  
    print "----------\n" + generatedText + "\n----------"

  # Following method was copied from:
  # https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py [Accessed: 12 Feb 2017]
  def sample(preds, temperature=1.0):
    preds = numpy.asarray(preds).astype('float64')
    preds = numpy.log(preds) / temperature
    exp_preds = numpy.exp(preds)
    preds = exp_preds / numpy.sum(exp_preds)
    probas = numpy.random.multinomial(1, preds, 1)
    return numpy.argmax(probas)

  startingPhrase = "Intelligence is the ability to adapt to change "

  if training:
    for epoch in range(EPOCHS):
      cost = 0
      print "Epoch: " + str(epoch)
      for point in range(0, textSize - (SEQ_LEN + BATCH_SIZE), BATCH_SIZE):
        print("Completed: %.3f" % (point/float(textSize) * 100))
        netInput, expectedOut = generateData(point, text, BATCH_SIZE, True)
        train(netInput, expectedOut)
        cost += computeCost(netInput, expectedOut)
  
      print("Average cost: %.5f" % (cost / ((textSize  - (SEQ_LEN + BATCH_SIZE))/(SEQ_LEN + BATCH_SIZE))))
      numpy.savez("thirdParameters" + str(epoch), lasagne.layers.get_all_param_values(outputLayer))
  else:
    data = numpy.load("secondParameters19.npz")
    parameters = data['arr_0']
    lasagne.layers.set_all_param_values(outputLayer, parameters)

    while True:
      predictText(startingPhrase, 1000)
      time.sleep(3)

if __name__ == "__main__":
  main()