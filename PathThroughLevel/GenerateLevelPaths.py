import os
import numpy
import FindPath
import numpy

# converts the level into representation for a* algorithm
def stringLevelToIntLevel(levelAsString):
  lines = levelAsString.count('\n')
  length = (len(levelAsString) / lines) - 1
  levelAsInt = numpy.zeros((lines,length), int)
  line = 0
  x  = 0
  for i in range(0, len(levelAsString)):
    # Impassible blocks
    if levelAsString[i] == '=' or levelAsString[i] == '?' or levelAsString[i] == '+' or levelAsString[i] == 'N' or \
       levelAsString[i] == 'X' or levelAsString[i] == '-' or levelAsString[i] == '~' or levelAsString[i] == 'F' or \
       levelAsString[i] == 'L' or levelAsString[i] == 'C' or levelAsString[i] == 'H' or levelAsString[i] == 'M':
      levelAsInt[line][x] = 1
    elif levelAsString[i] == 'T':
      levelAsInt[line][x] = 2
    elif levelAsString[i] == 'E':
      levelAsInt[line][x] = 3
    x += 1

    if levelAsString[i] == '\n':
      line += 1
      x = 0

  return levelAsInt

# draws the jump path onto the level string
def drawJump(mask, levelMap, start):
  print mask
  for j in range(0, mask.shape[0]):
    for i in range(0, mask.shape[1]):
      if mask[j][i] == 1:
        maskStart = [j,i]

  xMove = start[1] - maskStart[1]
  yMove = start[0] - maskStart[0]

  for j in range(0, mask.shape[0]):
    for i in range(0, mask.shape[1]):
      if mask[j,i] == 3 and j + yMove >= 0:
        levelMap[j + yMove, i + xMove] = 4

# These coords were generated using JumpMask.allReachablePoints()
neighbors = [[0, -1], [-4, 1], [-3, 1], [-2, 1], [-1, 1], [0, 1], [1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [6, 1], [7, 1], [8, 1], \
             [-4, 2], [-3, 2], [-2, 2], [-1, 2], [0, 2], [1, 2], [2, 2], [3, 2], [4, 2], [5, 2], [6, 2], [7, 2], [8, 2], [-4, 3], [-3, 3], \
             [-2, 3], [-1, 3], [0, 3], [1, 3], [2, 3], [3, 3], [4, 3], [5, 3], [6, 3], [7, 3], [8, 3], [-4, 4], [-3, 4], [-2, 4], [-1, 4], \
             [0, 4], [1, 4], [2, 4], [3, 4], [4, 4], [5, 4], [6, 4], [7, 4], [8, 4], [-4, 5], [-3, 5], [-2, 5], [-1, 5], [0, 5], [1, 5], \
             [2, 5], [3, 5], [4, 5], [5, 5], [6, 5], [7, 5], [8, 5], [-3, 6], [-2, 6], [-1, 6], [0, 6], [1, 6], [2, 6], [3, 6], [4, 6], \
             [5, 6], [6, 6], [7, 6], [8, 6], [-2, 7], [-1, 7], [0, 7], [1, 7], [2, 7], [3, 7], [4, 7], [5, 7], [6, 7], [7, 7], [8, 7], \
             [-1, 8], [0, 8], [1, 8], [2, 8], [3, 8], [4, 8], [5, 8], [6, 8], [7, 8], [8, 8], [0, 9], [1, 9], [2, 9], [3, 9], [4, 9], \
             [5, 9], [6, 9], [7, 9], [8, 9], [1, 10], [2, 10], [3, 10], [4, 10], [5, 10], [6, 10], [7, 10], [8, 10], [2, 11], [3, 11], \
             [4, 11], [5, 11], [6, 11], [7, 11], [8, 11], [3, 12], [4, 12], [5, 12], [6, 12], [7, 12], [8, 12], [4, 13], [5, 13], [6, 13], \
             [7, 13], [8, 13], [5, 14], [6, 14], [7, 14], [8, 14], [6, 15], [7, 15], [8, 15], [7, 16], [8, 16], [8, 17]]

# used to test all levels I generated
#levelsAsInt = []
# for file in os.listdir("EdgeCases"):
#   print "Finding path for: " + file
#   if file.endswith(".txt"):  

# try to find a path through a mario level and draw it on
def findPathThroughLevel(levelAsString):
  levelAsInt = stringLevelToIntLevel(levelAsString)
  
  # find the start and end locations
  for i in range(0, levelAsInt.shape[0]):
    for j in range(0, levelAsInt.shape[1]):
      if levelAsInt[i,j] == 2:
        start = (i,j)
      elif levelAsInt[i,j] == 3:
        end = (i,j)
  try:    
    path, masks = FindPath.astar(levelAsInt, start, end, neighbors)
  except TypeError:
    return -1
    
  # if the first point was jumped to
  if not masks[len(path) - 1][0][0] == -1:
    drawJump(masks[len(path) - 1], levelAsInt, start)
  
  levelAsInt[path[-1][0], path[-1][1]] = 4
  
  # looping backwards as the path goes from end to start
  for i in range(len(path)-2, -1, -1):
    # marks current point as visited
    levelAsInt[path[i][0], path[i][1]] = 4
  
    # if the current point was jumped to
    if not masks[i][0][0] == -1:
      # draw the jump from the previous point
      drawJump(masks[i], levelAsInt, path[i + 1])
      pass
    
  numpy.savetxt("GeneratedLevel/IntLevel.txt",levelAsInt, fmt="%i")
  levelAsIntFlat = levelAsInt.flatten()
  levelAsCharArray = list(levelAsString)
  newLines = 0
  for c in range(0, len(levelAsCharArray)):
    if levelAsCharArray[c] == '\n':
      newLines += 1
      continue
    if levelAsCharArray[c] == ' ' and levelAsIntFlat[c - newLines] == 4:
      levelAsCharArray[c] = 'a'
  levelWithPath = ''.join(levelAsCharArray)
  text_file = open("GeneratedLevel/PathLevel.txt", "w")
  text_file.write(levelWithPath)
  text_file.close()
  
  jumps = 0
  for mask in masks:
    # to avoid stairs skewing results single block jumps don't count
    if mask[0][0] != -1 and numpy.count_nonzero(mask == 3) > 1:
      jumps += 1
  return jumps
