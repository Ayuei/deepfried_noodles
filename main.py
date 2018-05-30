import glob
import network
import numpy as  np
import pandas as pd
import sklearn.metrics as metrics
from scipy import misc
import imageio
from sklearn.model_selection import StratifiedShuffleSplit
import cv2

def load():

    #data = np.empty(shape=(0,3073))
    data = []
    labels = []
    for image_path in glob.glob("./train-set/*.png"):
        image = imageio.imread(image_path)
        imgNum = int(image_path[16:18]) - 1
        data.append(np.ravel(cv2.resize(np.array(image), (28,28),
                                        interpolation=True)))
        labels.append(imgNum)

    data = np.array(data)
    labels = np.atleast_2d(np.array(labels)).T

    comb = np.hstack((data, labels))
    np.random.shuffle(comb)

    d = comb[:,:-1]
    l = comb[:,-1:]

    return d, l

def normalize(data):
    return data / 255

def oneHot(labels, numclasses):
    vals = np.eye(numclasses)[np.array(labels).reshape(-1)]
    return vals

def reverseOneHot(onehotLabels):
    vals = np.argmax(onehotLabels,axis=1)
    return np.atleast_2d(vals).T

data, labels = load()

data = normalize(data)

print('Total amount:', len(data))

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.33)
train = None
train_label = None
val = None
val_label = None
test = None
test_label = None

for train_index, test_index in sss.split(data, labels):
    train = data[train_index]
    train_label = labels[train_index]

    test = data[test_index]
    test_label = labels[test_index]

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.70)
for val_index, test_index in sss.split(test, test_label):
    val = test[val_index]
    val_label = test_label[val_index]

    test = data[test_index]
    test_label = labels[test_index]

print('Train', 'Test', 'Val')
print(len(train), len(test), len(val))

classCount = len(np.unique(labels))
convLayers = np.array([[3, 32, 1], [3, 32, 1], [3,64,1], [3,64,1]])
fcLayers = np.array([1024, classCount])
lr = 0.5e-3
epochs = 13
nn = network.NeuralNetworkBase(imageSize=28,
                               imageChannels=1,
                               classCount=classCount,
                               batchSize=32,
                               convLayers=convLayers,
                               fcLayers=fcLayers,
                               learningRate=lr,
                               epochs=epochs,
                               name="test_model")

nn.optimize(train, train_label, val, val_label)
import time

# Garbage Collection
train = None
train_label = None
data = None

time.sleep(1)

preds = nn.predict(test, test_label)
score = metrics.accuracy_score(test_label, preds)

print(score)
