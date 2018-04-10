# The following code was written with reference to:
# http://colinraffel.com/talks/hammer2015recurrent.pdf [Accessed: 6 Feb 2017]
# http://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/ [Accessed: 6 Feb 2017]
# https://github.com/Lasagne/Recipes/blob/master/examples/lstm_text_generation.py [Accessed: 6 Feb 2017]
import numpy
import lasagne
import theano
import time
import sys
import math
import PlayableLevelString
sys.path.append('../PathThroughLevel')
import GenerateLevelPaths
import socket
import threading
from decimal import Decimal

SEQ_LEN = 110
HIDDEN_UNITS = 512
BATCH_SIZE = 330
LEARN_RATE = 0.0005
W_DECAY = 0.000001

EPOCHS = 100

vocabSize = 0
textSize = 0
crossVSize = 0
crossV = ""
text = ""

charToInt = {}
intToChar = {}

COL_CNT_DIV = 10
COL_CNT_MAX = 20

netCompilied = False

#chage for demo
# levelReady = False
# stringLevelFinal = ""
levelReady = True
demofile = open("DemoLevel.txt", "r")
stringLevelFinal = demofile.read()
demofile.close()

if sys.argv[1] == "server":
  servSocket = socket.socket()
  servSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
  servSocket.bind(('127.0.0.1', 12346))
  servSocket.listen(0)

def initData():
  global vocabSize
  global textSize
  global charToInt
  global intToChar
  global text
  global crossVSize
  global crossV

  file = open("LevelData.txt", "r")
  text = file.read()
  crossVFile = open("CVData.txt", "r")
  crossV = crossVFile.read()

  # get all the unique characters
  chars = sorted(list(set(text)))
  vocabSize = len(chars)
  textSize = len(text)
  crossVSize = len(crossV)
  charToInt = {c:i for i,c in enumerate(chars)}
  intToChar = {i:c for i,c in enumerate(chars)}

  file.close()
  crossVFile.close()

# Created with referece to gen_data in:
# https://github.com/Lasagne/Recipes/blob/master/examples/lstm_text_generation.py [Accessed: 6 Feb 2017]
def generateData(startPoint, data, batchSize, getExpected):
  # for each character will store the sequence of one hot arrays that resembles each character
  inputSeq = numpy.zeros((batchSize, SEQ_LEN, vocabSize + 1), dtype='int32')
  # what character should be predicted next as an index in the one hot array
  expectedOutput = numpy.zeros(batchSize, dtype='int32')
  
  # for each batch in each sequence set the value in the one hot array
  # at the end of the sequence get the next character
  for batch in range(batchSize):
    for seq in range(SEQ_LEN):
      inputSeq[batch, seq, charToInt[data[startPoint+batch+seq]]] = 1

      colInput = int(((startPoint+batch+seq)/11)/COL_CNT_DIV) % COL_CNT_MAX

      inputSeq[batch, seq, -1] = colInput

    if getExpected:
      expectedOutput[batch] = charToInt[data[startPoint + batch + SEQ_LEN]]

  if getExpected:
    return inputSeq, expectedOutput
  else:
    return inputSeq

# Created with referece to gen_data in:
# https://github.com/Lasagne/Recipes/blob/master/examples/lstm_text_generation.py [Accessed: 6 Feb 2017]
def generateDataRandom(startPoint, points):
  inputSeq = numpy.zeros((BATCH_SIZE, SEQ_LEN, vocabSize + 1), dtype='int32')

  expectedOutput = numpy.zeros(BATCH_SIZE, dtype='int32')
  
  for batch in range(BATCH_SIZE):
    for seq in range(SEQ_LEN):
      inputSeq[batch, seq, charToInt[text[points[startPoint+batch]+seq]]] = 1

      colInput = int(((points[startPoint+batch]+seq)/11)/COL_CNT_DIV) % COL_CNT_MAX

      inputSeq[batch, seq, -1] = colInput

    expectedOutput[batch] = charToInt[text[points[startPoint + batch] + SEQ_LEN]]

  return inputSeq, expectedOutput

# Created with reference to:
# http://colinraffel.com/talks/hammer2015recurrent.pdf [Accessed: 6 Feb 2017]
# https://github.com/Lasagne/Recipes/blob/master/examples/lstm_text_generation.py [Accessed: 6 Feb 2017]
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
    learn_init=True, grad_clipping=1.0)

  # lstmLayer2 = lasagne.layers.recurrent.LSTMLayer(
  #   lasagne.layers.DropoutLayer(lstmLayer1, 0.5), HIDDEN_UNITS,
  #   ingate=gateParameters, forgetgate=gateParameters,
  #   cell=cellParameters, outgate=gateParameters,
  #   learn_init=True, grad_clipping=1.0)

  # another layer, same as the previous
  # only returns final results so a slice/reshape layer is not needed
  # only interested in the final prediction of the sequence
  lstmLayer3 = lasagne.layers.recurrent.LSTMLayer(
    lasagne.layers.DropoutLayer(lstmLayer1, 0.5), HIDDEN_UNITS,
    ingate=gateParameters, forgetgate=gateParameters,
    cell=cellParameters, outgate=gateParameters,
    learn_init=True,
    only_return_final=True, grad_clipping=1.0)

  # takes the output from the last layer, of size BATCH_SIZE, HIDDEN_UNITS
  # will output BATCH_SIZE, VOCAB_SIZE
  # softmax used to create a probability distribution
  outputLayer = lasagne.layers.DenseLayer(
    incoming=lasagne.layers.DropoutLayer(lstmLayer3, 0.5), 
    num_units=vocabSize, 
    nonlinearity=lasagne.nonlinearities.softmax)

  targetOutput = theano.tensor.ivector('target_output')

  if training:
    networkOutput = lasagne.layers.get_output(outputLayer)
  else:
    networkOutput = lasagne.layers.get_output(outputLayer, deterministic=True)

  # using cross entropy for the loss and take the mean across the batch
  cost = lasagne.objectives.categorical_crossentropy(networkOutput, targetOutput).mean()

  # regularisation
  l2Reg = lasagne.regularization.regularize_network_params(outputLayer, lasagne.regularization.l2)
  cost += l2Reg * W_DECAY

  # get all the parameters from the network for training
  allParams = lasagne.layers.get_all_params(outputLayer)

  # how to update the gradients 
  updates = lasagne.updates.adam(cost, allParams, learning_rate=LEARN_RATE)

  # trains the network
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

# Following method was copied from:
# https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py [Accessed: 12 Feb 2017]
def sample(preds, temperature):
  # helper function to sample an index from a probability array
  preds = numpy.asarray(preds).astype('float64')
  preds = numpy.log(preds) / temperature
  exp_preds = numpy.exp(preds)
  preds = exp_preds / numpy.sum(exp_preds)
  probas = numpy.random.multinomial(1, preds, 1)
  return numpy.argmax(probas)

def trainNN(train, computeCost, outputLayer):
  previousLoss1 = Decimal(5.000)
  previousLoss2 = Decimal(5.000)
  for epoch in range(EPOCHS):
    cost = 0
    passes = 0
    randomPoints = numpy.random.permutation(textSize - SEQ_LEN)
    randomPointsSize = len(randomPoints)
    print "Epoch: " + str(epoch)
    point = 0
    while point + BATCH_SIZE < randomPointsSize:
    #for point in range(0, textSize - (SEQ_LEN + BATCH_SIZE), BATCH_SIZE):
      print("Completed: %.1f" % (point/float(textSize) * 100))
      netInput, expectedOut = generateDataRandom(point, randomPoints)
      train(netInput, expectedOut)
      cost += computeCost(netInput, expectedOut)
      passes += 1
      point += BATCH_SIZE
    avgLoss = cost / passes
    print("Average cost: %.5f" % avgLoss)
    file = open("lossTrain.txt", "a")
    file.write(str(epoch) + ": " + str(avgLoss) + "\n")
    file.close()

    avgLoss = round(Decimal(avgLoss), 3)
    currLoss = crossValidComp(epoch, computeCost)
    numpy.savez("params/p" + str(epoch), lasagne.layers.get_all_param_values(outputLayer))

    if avgLoss == previousLoss1 and avgLoss == previousLoss2:
      break

    previousLoss2 = previousLoss1
    previousLoss1 = avgLoss

def crossValidComp(epoch, computeCost):
  print "Testing with cross validation set"
  cost = 0
  passes = 0
  for point in range(0, crossVSize - (SEQ_LEN + BATCH_SIZE), BATCH_SIZE):
    netInput, expectedOut = generateData(point, crossV, BATCH_SIZE, True)
    cost += computeCost(netInput, expectedOut)
    passes += 1
  avgLoss = cost / passes
  print("Average cost for CV: %.5f" % avgLoss)
  file = open("lossCV.txt", "a")
  file.write(str(epoch) + ": " + str(avgLoss) + "\n")
  file.close()
  avgLoss = round(Decimal(avgLoss), 3)
  return avgLoss

# Created with reference to:
# https://github.com/Lasagne/Recipes/blob/master/examples/lstm_text_generation.py [Accessed: 6 Feb 2017]
def predictText(phrase, numPredict, probDist, temp):
  assert(len(phrase) >= SEQ_LEN)

  netInput = generateData(len(phrase) - SEQ_LEN, phrase, 1, False)
  predictedCharsIndex = []

  endCharFound = False
  for c in range(numPredict):
    probablities = probDist(netInput).ravel()

    charIndex = sample(probablities, temp)

    colInput = int(((SEQ_LEN+c)/11)/COL_CNT_DIV) % COL_CNT_MAX

    # shift the sequence
    netInput[:, :-1, :] = netInput[:,1:,:]
    # set last char in the sequence to zero
    netInput[:, -1, :] = 0
    # input the character at the end of the sequence by setting the one hot value
    netInput[0, -1, charIndex] = 1
    netInput[0, -1, -1] = colInput

    # end on an E character, but only if at least 150 lines have been generated
    if intToChar[charIndex] == 'E':
      if c > 1419:
        endCharFound = True
      else:
        charIndex = charToInt[' ']
    # my seed will have the start character, any more will confuse the pathfinder
    elif intToChar[charIndex] == 'T':
      charIndex = charToInt[' ']

    predictedCharsIndex.append(charIndex)

    # finish the line if end char is found
    if endCharFound and intToChar[charIndex] == "\n":
      break;

  generatedText = phrase + ''.join(intToChar[charIndex] for charIndex in predictedCharsIndex)

  file = open("GenLevel.txt", "w")
  file.write(generatedText)
  file.close()

  if not endCharFound:
    print "append end column"
    return generatedText + "NE        \n"
  else:
    return generatedText

def serverListen():
  global levelReady
  conn, addr = servSocket.accept()
  print "Server listening"
  while True:
    data = conn.recv(1024)
    if not data:
      break
    print "got message: " + data
    if data == "RDY?\n":
      if netCompilied:
        conn.send("RDY\n")
      else:
        conn.send("NRDY\n")
    elif data == "LVL?\n":
      if levelReady:
        conn.sendall(stringLevelFinal)
        # levelReady = False
      else:
        conn.send("NLVL\n")
    else:
      conn.send("Unknown command: " + data + "\n")
  conn.close()
  print "Server no longer listening"

def main():
  global netCompilied
  global levelReady
  global stringLevelFinal
  initData()

  if sys.argv[1] == "train":
    print "Training"
    train, computeCost, outputLayer = initNet(True)
  elif sys.argv[1] == "gen":
    print "Generating"
    train, computeCost, outputLayer, probDist = initNet(False)
  elif sys.argv[1] == "server":
    print "Server"
    serverThread = threading.Thread(target=serverListen)
    serverThread.setDaemon(True)
    serverThread.start()
    train, computeCost, outputLayer, probDist = initNet(False)
  else:
    sys.exit("Unknown command!")

  if sys.argv[1] == "train":
    trainNN(train, computeCost, outputLayer)
  elif sys.argv[1] == "server":
    # data = numpy.load("p20.npz")
    # parameters = data['arr_0']
    # lasagne.layers.set_all_param_values(outputLayer, parameters)
    netCompilied = True
    while True:
      # generatedLevel = predictText("NT        \n         N\nN         \n         N\nN         \n         N\nN         \n         N\nN         \n         N\n", 2090, probDist, 1.0)
      # levelString = PlayableLevelString.generatedToPlayable(generatedLevel)
      # if levelString == "":
        # print "Level couldn't be converted to playable"
        # continue
      # if GenerateLevelPaths.findPathThroughLevel(levelString) == -1:
        # print "Could not find path through level"
        # continue
      stringLevelFinal = PlayableLevelString.addPadding(stringLevelFinal)
      stringLevel = open("GeneratedLevel/FinalGenLevel.txt", "w")
      stringLevel.write(stringLevelFinal)
      stringLevel.close()
      levelReady = True
      print "Level generated"
      while levelReady:
        time.sleep(2)
    servSocket.close()
  else:
    data = numpy.load("p20.npz")
    parameters = data['arr_0']
    lasagne.layers.set_all_param_values(outputLayer, parameters)
    for levelNum in range(1):
      generatedLevel = predictText("NT        \n         N\nN         \n         N\nN         \n         N\nN         \n         N\nN         \n         N\n", 2090, probDist, 1.0)
      levelString = PlayableLevelString.generatedToPlayable(generatedLevel)
      if levelString == "":
        print "Level couldn't be converted to playable"
        continue
      with open("GeneratedLevels/Level" + str(levelNum) +".txt", "w") as file:
        file.write(levelString)
if __name__ == "__main__":
  main()
