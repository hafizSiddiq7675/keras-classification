from sklearn import datasets
from cancernet import config
from imutils import paths  # basic image processing functions like resizing
import random, shutil, os # shutils offer function to perform operations

originalPaths = list(paths,list_images(config.INPUT_DATASET))
# randomly generate numbers 
random.seed(7)
random.shuffle(originalPaths)
#splitting training data and converting to integer 
index           = int(len(originalPaths)*config.TRAIN_SPLIT)
trainPaths      = originalPaths[:index]
testPaths       = originalPaths[:index]
# converting perform on validation
index           = int(len(trainPaths)*config.VAL_SPLIT)
valPaths        = trainPaths[:index]
trainPaths      = trainPaths[index:]
#dataset with path and their purpose
datasets        = [("training", trainPaths, config.TRAIN_PATH),
                   ("validation", valPaths, config.VAL_PATH),
                   ("testing", testPaths, config.TEST_PATH)
                    ]
for (setType, originalPaths,  basePath) in datasets:
    print(f'Building {setType} set')
    
    if not os.path.exists(basePath):
        print(f'Building directory {base_path}')
        os.makedirs(basePath)
        
        for path in originalPaths:
            file=path.split(os.path.sep)[-1]
            label=file[-5:-4]
            labelPath=os.path.sep.join([basePath,label])
            
            if not os.path.exists(labelPath):
                print(f'Building directory {labelPath}')
                os.makedirs(labelPath)
                newPath=os.path.sep.join([labelPath, file])
                shutil.copy2(path, newPath)


