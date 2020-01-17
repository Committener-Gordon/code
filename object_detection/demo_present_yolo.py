from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import gluoncv
from gluoncv import model_zoo, data, utils
import ffmpeg
import numpy as np
import os, sys
import cv2
import glob

RESIZECONSTANT = 600
DETECTIONRADIUS = 200 
PATH_TO_COW_DATA = "/cow-data/Kamerasicherung1/mx10-11-175-103/10_11_175_103/"
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
                
                # use this if you want fixed size boxes for the cow
                #x = int(((bbox[0] + bbox[2]) / 2) * resizeRatio)
                #y = int(((bbox[1] + bbox[3]) / 2) * resizeRatio)

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
            xFrom = int(int(detected[j][1]))
            xTo = int(detected[j][3])
            yFrom = int(detected[j][0])
            yTo = int(detected[j][2])

            xFrom = max(0, xFrom - padding)
            xTo = min(len(img), xTo + padding)
            yFrom = max(0, yFrom - padding)
            yTo = min(len(img[0]), yTo + padding)

            cow = img[xFrom:xTo, yFrom:yTo]
            cv2.imwrite(directory + "/c" + str(j) + "img" + fileSuffix + ".jpg", cow)

def deleteVideoFrames(directory, prefix):
    for filePath in glob.glob(directory + "/" + prefix + "*.jpg"):
        os.remove(filePath)

# This function can assemble a video from the cropped frames. Make sure that the dimensions of the frames are divisible by 2.
#def assembleVideoFromFrames(directory, prefix):
    #(
        #ffmpeg
        #.input(directory + "/" + prefix + "*.jpg", pattern_type="glob", framerate=2)
        #.output(prefix + ".mp4")
        #.run()
    #)




# Load a pretrained Faster RCNN model
net = model_zoo.get_model('yolo3_darknet53_voc', pretrained=True)

x, orig_img = data.transforms.presets.yolo.load_test('output_image.jpg', short=512)
print('Shape of pre-processed image:', x.shape)

# the model returns the predicted class ids, confidence scores and bounding boxes
class_IDs, scores, bounding_boxs = net(x)

ax = utils.viz.plot_bbox(orig_img, bounding_boxs[0], scores[0],
                         class_IDs[0], class_names=net.classes)
plt.show()

