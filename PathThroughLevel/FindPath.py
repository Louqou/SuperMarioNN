# A* algorithm (heuristic and astar) modified from:
# http://code.activestate.com/recipes/578919-python-a-pathfinding-with-binary-heap/ [Accessed: 4 Dec 16]
# Author: Christian Careaga (christian.careaga7@gmail.com)
import numpy
from heapq import *
import JumpMask

# See header for reference
def heuristic(a, b):
    return (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2

# See header for reference
def astar(array, start, goal, neighbors):
  close_set = set()
  came_from = {}
  gscore = {start:0}
  fscore = {start:heuristic(start, goal)}
  oheap = []

  heappush(oheap, (fscore[start], start))
    
  while oheap:
    current = heappop(oheap)[1]

    if current == goal:
      path = []
      masks = []

      while current in came_from:
        path.append(current)
        masks.append(came_from[current][1])
        current = came_from[current][0]

      return path, masks

    close_set.add(current)

    for i, j in neighbors:
      neighbor = current[0] + i, current[1] + j
      # increasing this means the path finder will favor moving one space at a time, but runs slower
      tentative_g_score = gscore[current] + heuristic(current, neighbor) * 200

      if 0 <= neighbor[0] < array.shape[0]:
        if 0 <= neighbor[1] < array.shape[1]:                
          if array[neighbor[0]][neighbor[1]] == 1:
            continue
          elif neighbor[0] + 1 == array.shape[0]:
            continue
          #path is only possible if wall underneath
          elif array[neighbor[0] + 1, neighbor[1]] != 1:
            continue
        else:
          # array bound y walls
          continue
      else:
        # array bound x walls
        continue
          
      if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
        continue
          
      if  tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [h[1]for h in oheap]:
        if abs(i) + abs(j) >= 2:
          mask = getJumpMask(current, neighbor, array)

          if mask[0][0] == -1:
            continue
          else:
            came_from[neighbor] = (current, mask)
        else:
          came_from[neighbor] = (current, [[-1]])

        gscore[neighbor] = tentative_g_score
        fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
        heappush(oheap, (fscore[neighbor], neighbor))

  return False

# get the mask that represents the jump made from start to end
def getJumpMask(start, end, levelMap):
  startPix = [start[1] * 16 + 7, start[0] * 16 + 7] 
  endPix = [end[1] * 16 + 7, end[0] * 16 + 7]

  if end[1] - start[1] == 1:
    if end[0] - start[0] > 0:
      mask = JumpMask.generateFallMask(startPix, endPix)
      if maskFitOverMap(mask, levelMap, start):
        return mask
    elif end[0] - start[0] < 0:
      mask = JumpMask.generateJumpOneMask(startPix, endPix)
      if maskFitOverMap(mask, levelMap, start):
        return mask
  else:
    for jumpSteps in range(0,9):
      for speed in range(0,9):
        mask = JumpMask.generateJumpMask(jumpSteps, startPix, endPix, speed)
        if not mask[0][0] == -1:
          if maskFitOverMap(mask, levelMap, start):
            return mask      

  return [[-1]]

# test if any level blocks will obstruct the jump
def maskFitOverMap(mask, levelMap, start):
  for j in range(0, mask.shape[0]):
    for i in range(0, mask.shape[1]):
      if mask[j][i] == 1:
        maskStart = [j,i]
  
  xMove = start[1] - maskStart[1]
  yMove = start[0] - maskStart[0]

  for j in range(0, mask.shape[0]):
    for i in range(0, mask.shape[1]):
      if j + yMove >= 0 and i + xMove < levelMap.shape[1]:
        if (mask[j,i] == 3 or mask[j,i] == 4) and levelMap[j + yMove, i + xMove] == 1:
          return False

  return True