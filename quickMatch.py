import os,math ,random
    

def splitIntoTrainTestValidationTxt(filenameTxt, validationPercent,testPercent):
    f = open(filenameTxt, "r")
    lines = f.readlines()
    f.close()
    print("Total files:" + str(len(lines)))

    validationSize = int(math.floor(len(lines) * validationPercent))
    testSize = int(math.floor(len(lines) * testPercent))
    trainSize = len(lines) - (validationSize + testSize)
    print("Train Size: " +str(trainSize))
    print("Test Size: " + str(testSize))
    print("Validation Size: " + str(validationSize))
    train = lines[:trainSize]
    val = lines[trainSize:(trainSize+validationSize)]
    test = lines[(trainSize+validationSize):]
            

    name = os.path.splitext(filenameTxt)[0]
    ftrain = name + '_train.txt'
    fval = name + '_validation.txt'
    ftest = name + '_test.txt'

    #train
    ft = open(ftrain, "w")
    ft.writelines(train)
    ft.close()

    #validation
    fv = open(fval, "w")
    fv.writelines(val)
    fv.close()

    
    #test
    fv = open(ftest, "w")
    fv.writelines(test)
    fv.close()


def shuffleFile( filename):
    with open(filename,'r') as f:
        lines = f.readlines()
        random.shuffle(lines)
    with open(filename,'w') as f:
        f.writelines(lines)


dir = "D:\\StableDiffusion\\stable-diffusion-terrain\\data\\RGBA-Mask"
dirRgb = "D:\\StableDiffusion\\stable-diffusion-terrain\\data\\RGBAv4_NewExpMean_FullData"

with open("littleMatch.txt",'a') as f:
        
    masklist = os.listdir(dir)
    for file in os.listdir(dirRgb):
        if file in masklist:
            f.write(file+"\n")

shuffleFile("littleMatch.txt")
splitIntoTrainTestValidationTxt("littleMatch.txt",0.10,0.05)