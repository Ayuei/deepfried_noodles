import numpy as np
import glob
import imageio
import cv2

img_size = 28

def load():
    data = []
    labels = []

    for image_path in glob.glob("./train-set/*.png"):
        image = imageio.imread(image_path)
        imgNum = int(image_path[16:18]) - 1
        data.append(np.ravel(np.array(cv2.resize(np.array(image), (img_size, img_size), interpolation=True))))
        labels.append(imgNum)
    val = []
    val_label = []
    for image_path in glob.glob("./vali-set/*.png"):
        image = imageio.imread(image_path)
        imgNum = int(image_path[15:17]) - 1
        val.append(np.ravel(np.array(cv2.resize(np.array(image), (img_size, img_size), interpolation=True))))
        val_label.append(imgNum)

    data = np.array(np.array(data)/255, dtype=np.uint8)
    labels = np.array(labels, dtype=np.uint8)
    val = np.array(np.array(val)/255, dtype=np.uint8)
    val_label = np.array(val_label, dtype=np.uint8)

    return data, labels, val, val_label

data, labels, val, val_labels = load()
print(data)
np.save('train.npy', data)
np.save('train_labels.npy', labels)
np.save('vali.npy', val)
np.save('vali_labels.npy', val_labels)
