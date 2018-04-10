import numpy

# Generates a 2D int array that's an appropriate size and marks on the start and end points
# blank spaces are 0, start is 1, end is 2
# points x, y wit y reversed
def generateBlankJumpMask(start, end):
  mask = numpy.zeros((abs(end[1] - start[1])/16 + 6, (end[0] - start[0])/16 + 2), int)
  mask[5,0] = 1
  mask[5 + (end[1] - start[1])/16, -2] = 2
  return mask

# Used to calculate if two points are in the same 16x16 block
def samePoint(point1, point2):
  return int(point1[0] / 16) == int(point2[0] / 16) and int(point1[1] / 16) == int(point2[1] / 16)

def outOfBounds(currentPoint, mask):
  return int(currentPoint[0] / 16) >= mask.shape[1] or int(currentPoint[1] / 16) >= mask.shape[0]

def generateFallMask(start, end):
  mask = generateBlankJumpMask(start, end)

  for i in range(mask.shape[1]):
    for j in range(mask.shape[0]):
      if mask[j,i] == 1:
        startMaskx = i
        startMasky = j

  for j in range(0, int((end[1] - start[1]) / 16)):
    mask[startMasky + j, startMaskx + 1] = 3

  return mask

def generateJumpOneMask(start, end):
  mask = generateBlankJumpMask(start, end)

  for i in range(mask.shape[1]):
    for j in range(mask.shape[0]):
      if mask[j,i] == 1:
        startMaskx = i
        startMasky = j

  for j in range(1, int((start[1] - end[1]) / 16) + 1):
    mask[startMasky - j, startMaskx] = 3

  return mask

# Given a start point, end point, number of steps the jump key is pressed and the current speed
# in the x direction will return a mask that shows all the squares the the point visited on its jump
# the jump behavoiur is modelled after Mario
# points are x, y with y reversed
def generateJumpMask(steps, start, end, speedX):
  # Mario can only move upward for 8 steps
  if steps > 8: 
    print "Too many jump steps"
    return [[-1]]
  if start[0] == end [0] and start[1] == end[1]:
    return [[-1]]

  mask = generateBlankJumpMask(start, end)
  speedJumpY = -1.9
  jumpTime = 7
  speedFall = 3

  # Find the start and end points labelled in the blank mask
  for i in range(mask.shape[1]):
    for j in range(mask.shape[0]):
      if mask[j,i] == 1:
        currentPoint = [i * 16 + 7, j * 16 + 7]
      elif mask[j,i] == 2:
        endPoint = [i * 16 + 7, j * 16 + 7] 

  # Calulates the position of the point after each time step
  # Each square visited is marked with a 3
  # Keeps going until either the goal is reached or falls out of bound of the mask
  if(steps > 0):
    currentPoint[0] += speedX
    currentPoint[1] += speedJumpY * jumpTime
   
    if outOfBounds(currentPoint, mask):
      return [[-1]]

    if samePoint(currentPoint, endPoint):
      return mask
    if mask[int((currentPoint[1]) / 16), int(currentPoint[0]) / 16] == 0 or mask[int((currentPoint[1]) / 16), int(currentPoint[0]) / 16] == 4:
      mask[int(currentPoint[1] / 16), int(currentPoint[0]) / 16] = 3
    checkSurroundingBlocks(currentPoint, mask)

  steps -= 1

  while steps > 0:
    currentPoint[0] += speedX
    currentPoint[1] += speedJumpY * jumpTime
    steps -= 1
    jumpTime -= 1

    if outOfBounds(currentPoint, mask):
      return [[-1]]
    
    if samePoint(currentPoint, endPoint):
      return mask
    if mask[int((currentPoint[1]) / 16), int(currentPoint[0]) / 16] == 0 or mask[int((currentPoint[1]) / 16), int(currentPoint[0]) / 16] == 4:
      mask[int(currentPoint[1] / 16), int(currentPoint[0]) / 16] = 3
    checkSurroundingBlocks(currentPoint, mask)

  fallSpeed = speedJumpY * 0.85 + 3

  # Once jumping ends Mario falls
  while True:
    currentPoint[0] += speedX
    currentPoint[1] += fallSpeed

    fallSpeed = fallSpeed * 0.85 + 3
    if fallSpeed > 8: fallSpeed = 8

    if outOfBounds(currentPoint, mask):
      return [[-1]]

    if samePoint(currentPoint, endPoint):
      return mask
    if mask[int((currentPoint[1]) / 16), int(currentPoint[0]) / 16] == 0 or mask[int((currentPoint[1]) / 16), int(currentPoint[0]) / 16] == 4:
      mask[int(currentPoint[1] / 16), int(currentPoint[0]) / 16] = 3
    checkSurroundingBlocks(currentPoint, mask)

def checkSurroundingBlocks(currentPoint, mask):
  try:
    if mask[int((currentPoint[1]-7) / 16), int(currentPoint[0]) / 16] == 0:
      mask[int((currentPoint[1]-7) / 16), int(currentPoint[0]) / 16] = 4
  except IndexError:
    pass
  try:
    if mask[int((currentPoint[1]+8) / 16), int(currentPoint[0]) / 16] == 0:
      mask[int((currentPoint[1]+8) / 16), int(currentPoint[0]) / 16] = 4
  except IndexError:
    pass
  try:
    if mask[int(currentPoint[1] / 16), int((currentPoint[0])-7) / 16] == 0:
      mask[int(currentPoint[1] / 16), int((currentPoint[0])-7) / 16] = 4
  except IndexError:
    pass
  try:
    if mask[int(currentPoint[1] / 16), int((currentPoint[0])+8) / 16] == 0:
      mask[int(currentPoint[1] / 16), int((currentPoint[0])+8) / 16] = 4
  except IndexError:
    pass

def reachablePositionsInBlock(height, length, start):
  reachable = numpy.zeros((height, length), int)

  for i in range(1, length):
    for j in range(0, height):
      x = i * 16 + 7
      y = j * 16 + 7

      for speed in range(0,9):
        for jumpSteps in range(0,9):
          mask = generateJumpMask(jumpSteps, start, [x,y], speed)
          if not mask[0][0] == -1:
            reachable[j,i] = 1
  
  return reachable

# get all the points reachable from a position
# used for neighbours list in A* algorithm
def allReachablePoints():
  reachable = reachablePositionsInBlock(13, 18, [7,71])
  print reachable
  points = []
  y = reachable.shape[0] - 1
  for i in range(0, reachable.shape[1]):
    for j in range(0, reachable.shape[0]):
      if reachable[j, i] == 1:
        points.append([j,i])
  
  for i in range(0, len(points)):
    points[i][0] -= 4
  
  print points

# used to test jumping
def maskTests():
#---Tests---
  #Furthest across and down
  startPoint = [7, 7]
  endPoint = [151, 135]
  
  #Furthest across and highest
  startPoint = [7, 71]
  endPoint = [71,7]
  
  #directly up
  startPoint = [7, 71]
  endPoint = [7,7]
  
  # directly up
  startPoint = [7, 71]
  endPoint = [23,7]
  
  #One across and down
  startPoint = [7, 7]
  endPoint = [23,135]
  
  #Too high
  startPoint = [7, 87]
  endPoint = [7,7]
  
  #Too far
  startPoint = [7, 7]
  endPoint = [167,7]
  
  startPoint = [7, 7]
  endPoint = [39,7]
  
  if int(endPoint[0]/16) - int(startPoint[0]/16) == 1 and int(endPoint[1]/16) - int(startPoint[1]/16) > 0:
    print generateFallMask(startPoint, endPoint)
  else:
    for speed in range(0,9):
      for jumpSteps in range(0,9):
        mask = generateJumpMask(jumpSteps, startPoint, endPoint, speed)
        if not mask[0][0] == -1:
          print "Speed: " + str(speed) + " JumpSteps: " + str(jumpSteps)
          print mask  
