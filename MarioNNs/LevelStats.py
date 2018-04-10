# calculate some statics on the levels to be used in evaluation
import sys
import os
import numpy
sys.path.append('../PathThroughLevel')
import GenerateLevelPaths

def getNumberOfGaps(level):
  intLevel = GenerateLevelPaths.stringLevelToIntLevel(level)
  numRow, numCol = intLevel.shape
  gaps = 0
  
  previousEmpty = False
  for col in range(numCol):
    rowEmpty = True
    for row in range(numRow):
      if intLevel[row, col] == 1:
        rowEmpty = False
        continue
    if rowEmpty and not previousEmpty:
      gaps += 1
    previousEmpty = rowEmpty
    
  return gaps

def getLevelStats(level):
  numOfBlocks = len(level)
  emptySpace = 0
  coins = 0
  enemies = 0
  powerUps = 0
  towers = 0
  
  for char in level:
    if char == '?':
      powerUps += 1
    elif char == '+':
      coins += 1
    elif char == '0':
      coins += 1
    elif char == 'r':
      enemies += 1
    elif char == 'g':
      enemies += 1
    elif char == 'b':
      enemies += 1
    elif char == 's':
      enemies += 1
    elif char == 'y':
      enemies += 1
    elif char == 'G':
      enemies += 1
    elif char == 'R':
      enemies += 1
    elif char == 'C':
      towers += 1
    elif char == '-':
      towers += 1
    elif char == 'T':
      emptySpace += 1
    elif char == 'E':
      emptySpace += 1
    elif char == 'a':
      emptySpace += 1
    elif char == ' ':
      emptySpace += 1
    elif char == '\n':
      numOfBlocks -= 1
      
  percEmptySpace = (emptySpace / float(numOfBlocks)) * 100
  
  jumps = GenerateLevelPaths.findPathThroughLevel(level)
  
  return percEmptySpace, coins, enemies, powerUps, jumps, towers, getNumberOfGaps(level)

def main():
  level = ""
  form = "{0:4.2f}"
  temps = ["0.5", "0.6", "0.7", "0.8", "0.9", "1.0", "1.1", "1.2", "1.3"]
  for temp in temps:
    percEmptySpaceList = []
    coinsList = []
    leniencyList = []
    jumpsList =[]
    towersList = []
    gapsList = []
    statsFile = open("GeneratedLevels/"+temp+"/Stats", "w")
    for file in os.listdir("GeneratedLevels/"+temp):
      if file.endswith(".txt"):
        print "Finding stats for: " + file
        with open("GeneratedLevels/"+temp+"/" + file, "r") as file:
          level = file.read()
        percEmptySpace, coins, enemies, powerUps, jumps, towers, gaps = getLevelStats(level)
        percEmptySpaceList.append(percEmptySpace)
        coinsList.append(coins)
        leniencyList.append(enemies - powerUps)
        if jumps != -1:
          jumpsList.append(jumps)
        towersList.append(towers)
        gapsList.append(gaps)
        statsFile.write(form.format(percEmptySpace)+" "+form.format(coins)+" "+form.format(enemies-powerUps)+" "+form.format(jumps)+" "+form.format(towers)+" "+form.format(gaps)+"\n") 

    statsFile.write(form.format(numpy.mean(percEmptySpaceList))+" "+form.format(numpy.mean(coinsList))+" "+form.format(numpy.mean(leniencyList))+" "+form.format(numpy.mean(jumpsList))+" "+form.format(numpy.mean(towersList))+" "+form.format(numpy.mean(gapsList))+"\n") 
    statsFile.write(form.format(numpy.std(percEmptySpaceList))+" "+form.format(numpy.std(coinsList))+" "+form.format(numpy.std(leniencyList))+" "+form.format(numpy.std(jumpsList))+" "+form.format(numpy.std(towersList))+" "+form.format(numpy.std(gapsList))+"\n") 
    statsFile.close()

if __name__ == "__main__":
  main()
