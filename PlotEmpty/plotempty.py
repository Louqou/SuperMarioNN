# used to plot my losses
import matplotlib.pyplot as plt
import numpy as np

cvFile = open("empty.txt","r")

cvData = cvFile.read()

cvFile.close()

y = [88.48, 88.28, 86.46, 86.52, 85.34, 82.56, 81.68, 83.01, 80.46]


figCV = plt.figure()

axCV = figCV.add_subplot(111)

axCV.set_xlabel('Temperature')
axCV.set_ylabel('Empty Space (%)')

x=[0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.3,1.4]
axCV.scatter(x, y)

axCV.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))

figCV.savefig('CVCurve.png')