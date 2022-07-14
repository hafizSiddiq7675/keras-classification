from sklearn import datasets
from cancernet import config    # configuration file
from imutils import paths  #need to download this --- basic image processing functions like resizing 
import random, shutil, os # shutils offer function to perform operations
#   at original input directory all images are shuffles
originalPaths = list(paths,list_images(config.INPUT_DATASET))
#   randomly generate numbers i.e.,no =12345 , it will generate this sequence in 7 different ways  
random.seed(7)
#   shuffle takes a list that randomly generated and will re-organize them according to order
random.shuffle(originalPaths)
#  getting train_split data and converting it to integer 
#  index splitting data into training & testing.  
index           = int(len(originalPaths)*config.TRAIN_SPLIT)
trainPaths      = originalPaths[:index]
testPaths       = originalPaths[:index]
#same functonality perform on validation_split to set aside some of the train_data to perform validation
index           = int(len(trainPaths)*config.VAL_SPLIT)
valPaths        = trainPaths[:index]
trainPaths      = trainPaths[index:]
#training, testing, validation datasets are define here
datasets        = [("training", trainPaths, config.TRAIN_PATH),
                   ("validation", valPaths, config.VAL_PATH),
                   ("testing", testPaths, config.TEST_PATH)
                    ]
# loop over the datasets
for (setType, originalPaths,  basePath) in datasets:
    #show  which data split we are creating
    print(f'Building {setType} set')
    # creating if output directory, output base path not exist
    if not os.path.exists(basePath):
        print(f'Building directory {base_path}')
        os.makedirs(basePath)
        #loop over the input original path
        for path in originalPaths:
            #extract file of the input image and its class label
            file=path.split(os.path.sep)[-1]
            label=file[-5:-4]

            #path is build to label directory
            labelPath=os.path.sep.join([basePath,label])
            #creating if label directory or label path not exist 
            if not os.path.exists(labelPath):
                print(f'Building directory {labelPath}')
                os.makedirs(labelPath)
                # construct the path to the destination image and then copy
                # the file/image itself
                newPath=os.path.sep.join([labelPath, file])
                shutil.copy2(path, newPath)


