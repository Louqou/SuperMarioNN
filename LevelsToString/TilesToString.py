import ImageCom
import time

tileSize = 16;

gameLevels = ImageCom.getImages("WorldImages/MarioDS")
currentTiles = ImageCom.getImages("MarioDSTileExamples")

for gameLevel in gameLevels:
	width, height = gameLevel.size

	# make sure image dimentions are correct
	if width % tileSize != 0 | height % tileSize != 0:
		print "Image found with dimentions not divisible by tileSize!"

	noTilesWidth = width / tileSize
	noTilesHeight = height / tileSize

	stringFileName = "MarioDSAsString/" + gameLevel.filename.split("/")[-1].split(".")[0] +".txt"
	print "Creating file " + stringFileName + "\n"
	#this will be the file to write the string into
	file = open(stringFileName, "w")

	for tileNumberHeight in range(0, noTilesHeight):
		for tileNumberWidth in range(0, noTilesWidth):
			tileLocationWidth = tileNumberWidth * tileSize
			tileLocationHeight = tileNumberHeight * tileSize
			newTile = gameLevel.crop((tileLocationWidth, tileLocationHeight, tileLocationWidth + tileSize, tileLocationHeight + tileSize))

			tileMatch = False
			matchedTile = ""

			for tile in currentTiles:
				if ImageCom.sameImage(tile.convert("RGB").load(), newTile.convert("RGB").load(), tileSize, 20):
					tileMatch = True
					matchedTile = tile.filename
					break

			if tileMatch:
				#gets the first letter of the file name
				charToUse = matchedTile.split("/")[-1][0]
				file.write(charToUse)
			else:
				file.write(" ")

		#at the end of each line write a new line
		file.write("\n")

	file.close()