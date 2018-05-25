import glob
import network
import numpy as  np
import pandas as pd
import sklearn.metrics as metrics
from scipy import misc
import imageio

def load():
        
    #data = np.empty(shape=(0,3073))
    data = [] 
    labels = []
    for image_path in glob.glob("./train-set/*.png"):
        image = imageio.imread(image_path)
        imgNum = int(image_path[16:18]) - 1
        data.append(np.ravel(np.array(image)))
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

train = data[0:10000]
test = data[10000:11000]
train_label = labels[0:10000]
test_label = labels[10000:11000]

classCount = 19
convLayers = np.array([[3,12,2]])
fcLayers = np.array([1024, 400, classCount])
lr = 0.002
epochs = 20
nn = network.NeuralNetworkBase(imageSize=128,
                                                imageChannels=1,
                                                classCount=classCount,
                                                batchSize=100,
                                                convLayers=convLayers,
                                                fcLayers=fcLayers,
                                                learningRate=lr,
                                                epochs=epochs)

nn.optimize(train, train_label)
preds = nn.predict(test, test_label)
score = metrics.accuracy_score(test_label, preds)


