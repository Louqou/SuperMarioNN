import os
import Image

# just for png, don't include "/" at the end of the dir location
def getImages(location):
	imageList = list()
	for fileName in os.listdir(location):
		if fileName.endswith(".png"):
			image = Image.open(location + "/" + fileName)
			imageList.append(image)

	return imageList

#compares two rgb images by pixel
#max perc difference
def sameImage(image1, image2, size, perc):
	noPixDiff = 0
	for i in range(0, size):
		for j in range(0, size):
			for p in range (0, 3):
				if(image1[i,j][p] != image2[i,j][p]):
					noPixDiff += 1

	if ((noPixDiff / float(size*size)) * 100) > perc:
		return False
	else:
		return True