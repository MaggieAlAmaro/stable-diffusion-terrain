import sys
from PIL import Image
import numpy as np
import collections

if __name__ == "__main__":



    colorCountDict = {
        "lightBlue" : 0,
        "blue" : 0,
        "darkBlue" : 0,
        "gray" : 0,
        "white" : 0,
        "green" : 0,
        "yellow" : 0,
        "orange" : 0,
        "red" : 0,
    }
    
    # presentColors = np.unique(data.reshape(-1, data.shape[2]), axis=0)
    # for x in range(width):
    #     for y in range(height):
    #         pixel =  list(data[x,y])
    #         if (pixel in colorLabel.values()):
    #             jey = list(colorLabel.keys())[list(colorLabel.values()).index(pixel)]
    #             colorCountDict[jey] += 1
    # print(colorCountDict)

    colorLabel = {
        "lightBlue" : [0, 255, 255], 
        "blue" : [0, 0, 255],
        "darkBlue" : [51, 51, 255],
        "gray" : [128, 128, 128],
        "white" : [255, 255, 255],
        "green" : [0, 255, 0],
        "yellow" : [255, 255, 0],
        "orange" : [255, 128, 0],
        "red" : [255, 0, 0],
    }


    image = Image.open("data/RGBA-Mask/1_1_0_0.png")
    data = np.array(image)
    width, height, c = data.shape
    totalNbrPixels = height*width
    colorValues, counts = np.unique(data.reshape(-1, 3), 
                           return_counts = True, 
                           axis = 0)
    colorValues = [list(color) for color in colorValues]
    colorsNames = [list(colorLabel.keys())[list(colorLabel.values()).index(values)] for values in colorValues]
    for i in range(len(colorsNames)):
        colorCountDict[colorsNames[i]] = counts[i]

    print(colorCountDict)
    caption = ""
    if colorCountDict['white'] + colorCountDict['green'] /totalNbrPixels >= 0.40:
        if caption:
            caption += ", "
        caption += "plain"
    if colorCountDict['orange']/totalNbrPixels >= 0.20:
        if caption:
            caption += ", "
        caption += "mountain"
    if colorCountDict['yellow'] /totalNbrPixels >= 0.20:
        if caption:
            caption += ", "
        caption += "hills"

    if colorCountDict['lightBlue'] >= 25*128:# arbitrary number of pixels
        if caption:
            caption += ", "
        caption += "river"
    if colorCountDict['blue'] >= 50*128:
        if caption:
            caption += ", "
        caption += "lake"

    print(str(caption))