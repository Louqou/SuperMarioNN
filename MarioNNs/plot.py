# used to plot my losses
import matplotlib.pyplot as plt

cvFile = open("lossCV.txt","r")
trainFile = open("lossTrain.txt","r")

cvData = cvFile.read()
trainData = trainFile.read()

trainFile.close()
cvFile.close()

cvData = cvData.split('\n')[:-1]
trainData = trainData.split('\n')[:-1]

figCV = plt.figure()
figTrain = plt.figure()

axCV = figCV.add_subplot(111)
axTrain = figTrain.add_subplot(111)

axCV.set_title('CV set loss')
axTrain.set_title('Training set loss')

axCV.set_xlabel('Epoch')
axCV.set_ylabel('Loss')
axTrain.set_xlabel('Epoch')
axTrain.set_ylabel('Loss')

axCV.plot(cvData)
axTrain.plot(trainData)

figCV.savefig('CVCurve.png')
figTrain.savefig('TrainCurve.png')