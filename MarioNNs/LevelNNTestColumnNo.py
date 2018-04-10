import numpy
import lasagne
import theano
import time
import sys
import math

SEQ_LEN = 220
HIDDEN_UNITS = 2
BATCH_SIZE = 264
EPOCHS = 10000

vocabSize = 0
textSize = 0
text = ""

charToInt = {}
intToChar = {}

CHAR_CNT_DIV = 110
COL_CNT_MAX = 20
colInput = 0

testFile = open("test.txt", "w")
indexes = []

def initData():
  global vocabSize
  global textSize
  global charToInt
  global intToChar
  global text

  file = open("LevelData.txt", "r")
  text = file.read()

  # get all the unique characters
  chars = sorted(list(set(text)))
  vocabSize = len(chars)
  textSize = len(text)
  charToInt = {c:i for i,c in enumerate(chars)}
  intToChar = {i:c for i,c in enumerate(chars)}

  file.close()

def generateData(startPoint, data, batchSize, getExpected):
  global columnCounter
  global colInput
  global indexes
  # for each character will store the sequence of one hot arrays that resembles each character
  inputSeq = numpy.zeros((batchSize, SEQ_LEN, vocabSize + 1), dtype='int32')
  # what character should be predicted next as an index in the one hot array
  expectedOutput = numpy.zeros(batchSize, dtype='int32')
  
  # for each batch in each sequence set the value in the one hot array
  # at the end of the sequence get the next character
  for batch in range(batchSize):
    for seq in range(SEQ_LEN):
      inputSeq[batch, seq, charToInt[data[startPoint+batch+seq]]] = 1

      colInput = int(math.floor(math.floor((startPoint+batch+seq)/11)/10)) % 20

      inputSeq[batch, seq, -1] = colInput

      if not ((startPoint+batch+seq) in indexes):
        indexes.append(startPoint+batch+seq)
        testFile.write(str(colInput))

        if data[startPoint+batch+seq] == '\n':
          testFile.write('\n')

    if getExpected:
      expectedOutput[batch] = charToInt[data[startPoint + batch + SEQ_LEN]]

  if getExpected:
    return inputSeq, expectedOutput
  else:
    return inputSeq 

def initNet(training):
  # input layer of the network
  inputLayer = lasagne.layers.InputLayer(shape=(None, SEQ_LEN, vocabSize + 1))

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

  lstmLayer2 = lasagne.layers.recurrent.LSTMLayer(
    lasagne.layers.DropoutLayer(lstmLayer1, 0.5), HIDDEN_UNITS,
    ingate=gateParameters, forgetgate=gateParameters,
    cell=cellParameters, outgate=gateParameters,
    learn_init=True, grad_clipping=100.)

  # another layer, same as the previous
  # only returns final results so a slice/reshape layer is not needed
  # only interested in the final prediction of the sequence
  lstmLayer3 = lasagne.layers.recurrent.LSTMLayer(
    lasagne.layers.DropoutLayer(lstmLayer2, 0.5), HIDDEN_UNITS,
    ingate=gateParameters, forgetgate=gateParameters,
    cell=cellParameters, outgate=gateParameters,
    learn_init=True, grad_clipping=100.,
    only_return_final=True)

  # takes the output from the last layer, of size BATCH_SIZE, HIDDEN_UNITS
  # will output BATCH_SIZE, VOCAB_SIZE
  # softmax used to create a probability distribution
  outputLayer = lasagne.layers.DenseLayer(
    incoming=lasagne.layers.DropoutLayer(lstmLayer3, 0.5), 
    num_units=vocabSize + 1, 
    nonlinearity=lasagne.nonlinearities.softmax)

  # tensor variables
  targetOutput = theano.tensor.ivector('target_output')

  if training:
    networkOutput = lasagne.layers.get_output(outputLayer)
  else:
    networkOutput = lasagne.layers.get_output(outputLayer, deterministic=True)

  # using cross entropy for the loss and take the mean across the batch
  cost = lasagne.objectives.categorical_crossentropy(networkOutput, targetOutput).mean()
  # get all the parameters from the network for training
  allParams = lasagne.layers.get_all_params(outputLayer)

  # how the update the gradients 
  updates = lasagne.updates.adam(cost, allParams)

  # trains the neural network
  train = theano.function(
    [inputLayer.input_var, targetOutput],
    cost,
    updates=updates,
    allow_input_downcast=True)

  # computer the cost
  computeCost = theano.function(
    [inputLayer.input_var, targetOutput],
    cost,
    allow_input_downcast=True)

  if not training:
    # used to select the character
    probDist = theano.function([inputLayer.input_var], networkOutput, allow_input_downcast=True)
    return train, computeCost, outputLayer, probDist
  else:
    return train, computeCost, outputLayer

#https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
def sample(preds, temperature):
  # helper function to sample an index from a probability array
  preds = numpy.asarray(preds).astype('float64')
  preds = numpy.log(preds) / temperature
  exp_preds = numpy.exp(preds)
  preds = exp_preds / numpy.sum(exp_preds)
  probas = numpy.random.multinomial(1, preds, 1)
  return numpy.argmax(probas)

def trainNN(train, computeCost, outputLayer):
  global colInput
  file = open("testLoss.txt", "w") 
  for epoch in range(EPOCHS):
    cost = 0
    print "Epoch: " + str(epoch)
    for point in range(0, textSize - (SEQ_LEN + BATCH_SIZE), BATCH_SIZE):
      print("Completed: %.1f" % (point/float(textSize) * 100))
      netInput, expectedOut = generateData(point, text, BATCH_SIZE, True)
      train(netInput, expectedOut)
      cost += computeCost(netInput, expectedOut)
    colInput = 0
    avgLoss = cost / ((textSize  - (SEQ_LEN + BATCH_SIZE))/(SEQ_LEN + BATCH_SIZE))
    print("Average cost: %.3f" % avgLoss)
    file.write(str(epoch) + ": " + str(avgLoss) + "\n")
    numpy.savez("testParameters" + str(epoch), lasagne.layers.get_all_param_values(outputLayer))
  file.close()

def predictText(phrase, numPredict, probDist):
  assert(len(phrase) >= SEQ_LEN)

  netInput = generateData(len(phrase) - SEQ_LEN, phrase, 1, False)
  predictedCharsIndex = []

  for c in range(numPredict):
    probablities = probDist(netInput).ravel()

    charIndex = sample(probablities, 1.1)
    
    predictedCharsIndex.append(charIndex)
    # shift the sequence
    netInput[:, :-1, :] = netInput[:,1:,:]
    # set last char in the sequence to zero
    netInput[:, -1, :] = 0
    # input the character at the end of the sequence by setting the one hot value
    netInput[0, -1, charIndex] = 1
  generatedText = phrase + ''.join(intToChar[charIndex] for charIndex in predictedCharsIndex)

  file = open("GenLevel.txt", "w")
  file.write(generatedText)
  file.close()

def main():
  initData()
  training = sys.argv[1] == "train"
  if training:
    print "Training"
    train, computeCost, outputLayer = initNet(training)
  else:
    train, computeCost, outputLayer, probDist = initNet(training)
    print "Generating"

  if training:
    trainNN(train, computeCost, outputLayer)
  else:
    data = numpy.load("firstParameters20.npz")
    parameters = data['arr_0']
    lasagne.layers.set_all_param_values(outputLayer, parameters)
    predictText("NT        \n         N\nN         \n         N\nN         \n         N\nN         \n         N\nN         \n         N\nN         \n         N\nN         \n         N\nN         \n         N\nN         \n         N\nN         \n         N\n", 4400, probDist)

if __name__ == "__main__":
  main()