import os

# rotates the string levels and joins them together to create the training data
def rotateAndAppend(directory, data):
  for fileName in os.listdir(directory):
    if fileName.endswith(".txt") or fileName.endswith(".txt_S"):
      print fileName 
      file = open(directory + "/" + fileName, 'r')
      text = file.read()
      lines = text.split("\n")
      
      for i in range(len(lines[0])):
        lineToAdd = ""
        for l in range(len(lines) - 2, -1, -1):
          lineToAdd += lines[l][i]
        if i % 2 != 0:
          #reverses the string
          lineToAdd = lineToAdd[::-1]
        data += (lineToAdd + "\n")

  return data
    
data = ""

data = rotateAndAppend("../LevelsAsString", data)
data = rotateAndAppend("../LevelsAsString/Shifted", data)

file = open("./LevelData.txt", "w")
file.write(data)      
file.close()