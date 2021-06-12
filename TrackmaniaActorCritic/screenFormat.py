import cv2 as cv
import numpy as np
import os
from time import time
from windowcapture import WindowCapture


winName = 'trackmania'  # input('Name of window to be captured:')
path = os.path.expanduser('~/Documents/{}TrainingData'.format(winName))
if not os.path.exists(path):
    os.mkdir(path)
os.chdir(os.path.dirname(path))

wincap = WindowCapture(winName)

#loop_time = time()


def capturescreen():
    # get an updated image of the specified window.
    screenshot = wincap.get_screenshot()

    scalar = 20
    width = int(screenshot.shape[1] * scalar / 100)
    height = int(screenshot.shape[0] * scalar / 100)
    screenshot = cv.resize(screenshot, (width, height))

    return screenshot
    # feed image to openCV and display in 'Computer Vision' window.
    #cv.imshow('Computer Vision', screenshot)
    #cv.imwrite('{}TrainingData/image{}.png'.format(winName, counter), screenshot)


    # determine time between loops.
    # formatted in FPS.
    #print('FPS {}'.format(1 / (time() - loop_time)))
    #loop_time = time()



