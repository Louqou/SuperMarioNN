import ImageCom
import time

tileSize = 16;

gameLevels = ImageCom.getImages("WorldImages/MarioDS")
currentTiles = ImageCom.getImages("Tiles")

for gameLevel in gameLevels:
	width, height = gameLevel.size

	# make sure image dimentions are correct
	if width % tileSize != 0 | height % tileSize != 0:
		print "Image found with dimentions not divisible by tileSize!"

	noTilesWidth = width / tileSize
	noTilesHeight = height / tileSize

  # for all the tiles in the level check if found a similar tile, if not save it
	for tileNumberWidth in range(0, noTilesWidth):
		for tileNumberHeight in range(0, noTilesHeight):
			tileLocationWidth = tileNumberWidth * tileSize
			tileLocationHeight = tileNumberHeight * tileSize
			newTile = gameLevel.crop((tileLocationWidth, tileLocationHeight, tileLocationWidth + tileSize, tileLocationHeight + tileSize))

			if len(currentTiles) == 0:
				newTile.save("Tiles/0.png")
				currentTiles.append(newTile)

			isDiffTile = True;

			for currentTile in currentTiles:
				if ImageCom.sameImage(currentTile.convert("RGB").load(), newTile.convert("RGB").load(), tileSize, 50):
					isDiffTile = False;

			if isDiffTile:
				currentTiles.append(newTile)
				newTile.save("Tiles/" + str(len(currentTiles) - 1) + ".png")