#demo the path finder algorithm
import sys
sys.path.append('../../PathThroughLevel')
import GenerateLevelPaths

with open('NoPath.txt') as file:
  levelString = file.read()

GenerateLevelPaths.findPathThroughLevel(levelString)