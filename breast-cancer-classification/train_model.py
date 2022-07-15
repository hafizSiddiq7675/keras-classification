# matplotlib beckend is set to store figures in background
import matplotlib
matplotlib.use("Agg")

#	necessary packages will import
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adagrad
from keras.utils import np_utils
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from cancernet.cancernet import CancerNet
from cancernet import config
from imutils import paths
import matplotlib.pyplot as plt	
import numpy as np
import os

#	initial batch size and learning rate, define # of EPOCHS 
NUM_EPOCHS	=	40; 
INIT_LR		=	1e-2; 
#list conversion to length
BS			=	32
# the total number of image paths in training/validation 
# & testing directories will discover & conversion to length
trainPaths	=	list(paths.list_images(config.TRAIN_PATH))
lenTrain	=	len(trainPaths)
lenVal		=	len(list(paths.list_images(config.VAL_PATH)))
lenTest		=	len(list(paths.list_images(config.TEST_PATH)))
# transform skewed distribution to normal in the labeled data
trainLabels	=	[int(p.split(os.path.sep)[-2]) for p in trainPaths]
trainLabels	=	np_utils.to_categorical(trainLabels)
classTotals	=	trainLabels.sum(axis=0)
classWeight	=	classTotals.max()/classTotals
# initialize the training data augmentation object
# translats, randomly shifts, and flips each training data sample
trainAug 	= ImageDataGenerator(
	rescale				=	1/255.0,
	rotation_range		=	20,
	zoom_range			=	0.05,
	width_shift_range	=	0.1,
	height_shift_range	=	0.1,
	shear_range			=	0.05,
	horizontal_flip		=	True,
	vertical_flip		=	True,
	fill_mode			=	"nearest")

# initialize the validation/testing data augmentation object
valAug		= ImageDataGenerator(rescale=1 / 255.0)
# initialize the training generator
trainGen 	= trainAug.flow_from_directory(
	config.TRAIN_PATH,
	class_mode		=	"categorical",
	target_size		=	(48,48),
	color_mode		=	"rgb",
	shuffle			=	True,
	batch_size		=	BS
	)
# initialize the validation generator
valGen 				= valAug.flow_from_directory(
	config.VAL_PATH,
	class_mode		=	"categorical",
	target_size		=	(48,48),
	color_mode		=	"rgb",
	shuffle			=	False,
	batch_size		=	BS
	)	
# initialize the testing generator
testGen 			= 	valAug.flow_from_directory(
	config.TEST_PATH,
	class_mode="categorical",
	target_size=(48,48),
	color_mode="rgb",
	shuffle=False,
	batch_size=BS
	)
# our CancerNet model will initialize and compile 
model	=	CancerNet.build(width=48,height=48,depth=3,classes=2)
opt		=	Adagrad(lr=INIT_LR,decay=INIT_LR/NUM_EPOCHS)
model.compile(
	loss		=	"binary_crossentropy",	
	optimizer	=	opt,
	metrics		=	["accuracy"])

# generated model will fit
M=model.fit_generator(
	trainGen,
	steps_per_epoch		=	lenTrain//BS,
	validation_data		=	valGen,
	validation_steps	=	lenVal//BS,
	class_weight		=	classWeight,
	epochs				=	NUM_EPOCHS
	)
# to make predictions the testing generator will reset & then our train model will use
print("Now evaluating the model")
testGen.reset()
pred_indices	=	model.predict_generator(testGen,steps=(lenTest//BS)+1)

# for each testing image the index of the label with 
# largest predicted probability
pred_indices	=	np.argmax(pred_indices,axis=1)
# plot a nicely formatted classification report
print(classification_report(testGen.classes, pred_indices, target_names=testGen.class_indices.keys()))
# onfusion matrix will compute & use to derive 
# the raw accuracy, sensitivity, and specificity
cm=confusion_matrix(testGen.classes,pred_indices)
total		=	sum(sum(cm))
accuracy	=	(cm[0,0]+cm[1,1])/total
specificity	=	cm[1,1]/(cm[1,0]+cm[1,1])
sensitivity	=	cm[0,0]/(cm[0,0]+cm[0,1])
# plot result of confusion matrix, accuracy, sensitivity, and specificity
print(cm)
print(f'Accuracy: {accuracy}')
print(f'Specificity: {specificity}')
print(f'Sensitivity: {sensitivity}')
# plot the training loss and accuracy
N = NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,N), M.history["loss"], label="train_loss")
plt.plot(np.arange(0,N), M.history["val_loss"], label="val_loss")
plt.plot(np.arange(0,N), M.history["acc"], label="train_acc")
plt.plot(np.arange(0,N), M.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on the IDC Dataset")
plt.xlabel("Epoch No.")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig('plot.png')
