import glob
import network
import numpy as  np
import pandas as pd
import sklearn.metrics as metrics
from scipy import misc
import imageio
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

print(len(data))
train = data[0:30000]
test = data[30000:]
train_label = labels[0:30000]
test_label = labels[30000:]

classCount = len(np.unique(labels))
convLayers = np.array([[3, 32, 1], [3, 32, 1], [3,64,1], [3,64,1]])
fcLayers = np.array([1024, classCount])
lr = 1e-6
epochs = 1000
nn = network.NeuralNetworkBase(imageSize=28,
                               imageChannels=1,
                               classCount=classCount,
                               batchSize=32,
                               convLayers=convLayers,
                               fcLayers=fcLayers,
                               learningRate=lr,
                               epochs=epochs)

nn.optimize(train, train_label)
import pickle

pickle.dump(nn, open('network_MNIST.pkl', 'wb+'))

preds = nn.predict(test, test_label)
score = metrics.accuracy_score(test_label, preds)
