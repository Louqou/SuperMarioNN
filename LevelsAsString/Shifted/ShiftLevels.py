import os

for fileName in os.listdir("."):
  if fileName.endswith(".txt"):  
    file = open(fileName, 'r')
    newFile = open(fileName + "_S", 'w')
    for line in file:
      newLine = line[:2] + line[1:-3] + line[-2:]
      newFile.write(newLine)
    file.close()
    newFile.close()