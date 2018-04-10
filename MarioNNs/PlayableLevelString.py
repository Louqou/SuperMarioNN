# transforms the levels generated my the network into a format the mario engine can use
import os

def generatedToPlayable(genLevel):
  playLevelFile = open("GeneratedLevel/PlayableLevel.txt", "w")
  lines = genLevel.split("\n")[:-1]
  playLevel = ""
 
  for c in range(10):
    lineToAdd = ""
    for l in range(len(lines)):
      try: 
        if l % 2 == 0:
          lineToAdd += lines[l][-1*c-1]
        else:
          lineToAdd += lines[l][c]
      except IndexError:
        return ""
    lineToAdd += "\n"
    playLevel += lineToAdd

  playLevelFile.write(playLevel)
  playLevelFile.close()

  return playLevel

def addPadding(genLevel):
  lines = genLevel.split("\n")
  paddedLevel = ""
  for j in range(7):
    for i in range(len(lines[0])):
      paddedLevel += " "
    paddedLevel += "\n"
        
  paddedLevel += genLevel
  paddedLevel += lines[-2]
  paddedLevel += "\n"

  return paddedLevel