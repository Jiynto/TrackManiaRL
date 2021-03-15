import os
import cv2 as cv
import numpy as np
import pickle
import matplotlib.pyplot as plt


path = os.path.expanduser('~/Documents/trackmaniaTrainingData')
training_data = []
IMG_SIZE = 64

def create_training_data():
    for img in os.listdir(path):
        try:
            fullpath = os.path.join(path, img)
            img_array = cv.imread(fullpath, cv.IMREAD_GRAYSCALE)
            new_array = cv.resize(img_array, (IMG_SIZE, IMG_SIZE))
            training_data.append([new_array])
            #plt.imshow(new_array, cmap="gray")
            #plt.show()
            #break
        except Exception as e:
            pass


create_training_data()
training_data = np.array(training_data).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
print(training_data)


pickle_out = open("training_data.pickle", "wb")
pickle.dump(training_data, pickle_out)
pickle_out.close()

