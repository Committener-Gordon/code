from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import gluoncv
from gluoncv import model_zoo, data, utils
import ffmpeg
import numpy as np
import os, sys
import cv2
import glob
import re


RESIZECONSTANT = 600
DETECTIONRADIUS = 200 
PATH_TO_COW_DATA = "/cow-data/Kamerasicherung1/Kamerasicherung1/mx10-11-175-103/10_11_175_103/"
FRAME_PREFIX = "out-"

def extractFrames(inputFile, directory):
    width, height = getInfo(inputFile)

    (
        ffmpeg
        .input(inputFile, f='mxg')
        .crop(width/2, height, width/2, height)
        .output(directory + '/' + FRAME_PREFIX + '%03d.jpg', r='2')
        .global_args('-loglevel', 'error')
        .global_args('-y')
        .run()
    )

    return min(width/2, height) / RESIZECONSTANT

def getInfo(inputFile):
	probe = ffmpeg.probe(inputFile)
	info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
	width = int(info['width'])
	height = int(info['height'])
	return width, height

def getSubdirs(fromDir):
    filenames = os.listdir(fromDir)
    dirs = []
    for filename in filenames:
        if os.path.isdir(os.path.join(fromDir, filename)):
            dirs.append(filename)

    return dirs

def getCowBoxes(box_ids, scores, bboxes, classes, resizeRatio):
    # not every cow in the img will be detected as 'cow'. Therefore and since we can expect
    # that there are no other animals in the image, we will also accept
    # other classes, that have similarities with cows (bigger mammals).
    acceptedClasses = ['cow', 'person', 'dog', 'cat', 'sheep', 'horse']

    npScores = scores.asnumpy()
    npIDs = box_ids.asnumpy()
    detected = []

    # gather all detection that are in the list of accepted classes and have a confidence of > 50%
    for i in range(npScores[0].size):
        myscore=npScores[0][i][0]
        myid=int(npIDs[0][i][0])

        if myscore > 0.5:
            if classes[myid] in acceptedClasses:
                print("Nice, found a " + classes[myid] + " and I'm " + str(myscore) + " sure")
                #detected.append(np.dot(bboxes[0][i].asnumpy(), resizeRatio))
                bbox = bboxes[0][i].asnumpy()

                detected.append(bbox * resizeRatio)
            else:
                print("Found a " + classes[myid] + ". RIP")

    return detected


def extractCowsFromFrames(detected, directory, padding):
    filePaths = []
    for filePath in glob.glob(directory + "/*.jpg"):
        filePaths.append(filePath)

    #crop the cows from the image
    for i in range(len(filePaths)):
        path = filePaths[i]
        img = cv2.imread(path)
        fileSuffix = str(i)
        if i < 10:
            fileSuffix = "0" + fileSuffix
        for j in range(len(detected)):
            x = int((detected[j][1] + detected[j][3]) / 2)
            y = int((detected[j][0] + detected[j][2]) / 2)
            
            if x < 200:
                xFrom, xTo = 0, 400
            elif x > len(img) - 200:
                xFrom, xTo = len(img) - 400, len(img)
            else:
                xFrom, xTo = x-200, x+200
            
            
            if y < 200:
                yFrom, yTo = 0, 400
            elif x > len(img[0]) - 200:
                yFrom, yTo = len(img[0]) - 400, len(img[0])
            else:
                yFrom, yTo = y-200, y+200
            
           
            cow = img[xFrom:xTo, yFrom:yTo]
            cv2.imwrite(directory + "/c" + str(j) + "img" + fileSuffix + ".jpg", cow)
            
    for i in range(len(detected)):
        assembleVideoFromFrames(directory, "c" + str(i) + "img")
        deleteVideoFrames(directory, "c" + str(i) + "img")

def deleteVideoFrames(directory, prefix):
    for filePath in glob.glob(directory + "/" + prefix + "*.jpg"):
        os.remove(filePath)

# This function can assemble a video from the cropped frames. Make sure that the dimensions of the frames are divisible by 2.
def assembleVideoFromFrames(directory, prefix):
    (
        ffmpeg
        .input(directory + "/" + prefix + "*.jpg", pattern_type="glob", framerate=2)
        .output(directory + "/" + prefix + ".mp4", pix_fmt='gray')
        .run()
    )




# Load a pretrained Faster RCNN model
net = model_zoo.get_model('faster_rcnn_resnet50_v1b_voc', pretrained=True)

firstLvlDirs = getSubdirs(PATH_TO_COW_DATA)
for firstLvlDir in firstLvlDirs:
    if not os.path.exists(firstLvlDir):
        os.mkdir(firstLvlDir)
    sndLvlDirs = getSubdirs(PATH_TO_COW_DATA + "/" + firstLvlDir)

    for sndLvlDir in sndLvlDirs:
        directory = firstLvlDir + "/" + sndLvlDir
        if not os.path.exists(directory):
            os.mkdir(directory)

        pathGlobal = os.path.join(PATH_TO_COW_DATA, directory)
        for filePath in glob.glob(pathGlobal + "/M*.jpg"):
            pattern = re.compile("^M[0-9]+$")
            filename = os.path.basename(filePath)[slice(0, -4)]
            if pattern.match(filename):
                frameDirectory = directory + "/" + filename

                print(frameDirectory)
                if not os.path.exists(frameDirectory):
                    os.mkdir(frameDirectory)

                resizeRatio = extractFrames(filePath, frameDirectory)
                detectionFrame = frameDirectory + "/" + FRAME_PREFIX + "001.jpg"
                x, orig_img = data.transforms.presets.rcnn.load_test(detectionFrame)

                # the model returns the predicted class ids, confidence scores and bounding boxes
                box_ids, scores, bboxes = net(x)
                classes = net.classes
                detectedCows = getCowBoxes(box_ids, scores, bboxes, classes, resizeRatio)
                if len(detectedCows) > 0:
                    extractCowsFromFrames(detectedCows, frameDirectory, 100)
                deleteVideoFrames(frameDirectory, FRAME_PREFIX)

                if not os.listdir(frameDirectory):
                    os.rmdir(frameDirectory)

        if not os.listdir(directory):
                os.rmdir(directory)