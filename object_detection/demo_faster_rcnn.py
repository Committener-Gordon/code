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
PATH_TO_COW_DATA = "/cow_data/Kamerasicherung1/mx10-11-175-103/10_11_175_103/115/"

def extractFrames(inputFile, directory):
    width, height = getInfo(inputFile)

    (
        ffmpeg
        .input(inputFile, f='mxg')
        .crop(width/2, height, width/2, height)
        .output(directory + '/out-%03d.jpg', r='2')
        .run(capture_stdout=True)
    )

    return min(width/2, height) / RESIZECONSTANT

    #video = (
    #    np
    #    .frombuffer(out, np.uint8)
    #    .reshape([-1, height, int(width/2), 3])
    #)

    #return video

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



#if not os.path.exists('test'):
    #os.mkdir("test")

#if not os.path.exists('detected'):
    #os.mkdir("detected")


def getCowBoxes(scores, bboxes, box_ids):
    acceptedClasses = ['cow', 'person', 'dog', 'cat', 'sheep', 'horse']

    npScores = scores.asnumpy()
    npIDs = box_ids.asnumpy()
    detected = []

    # gather all detection that are in the list of accepted classes and have a confidence of > 50%
    for i in range(npScores[0].size):
        myscore=npScores[0][i][0]
        myid=int(npIDs[0][i][0])

        if myscore > 0.5:
            if net.classes[myid] in acceptedClasses:
                print("Nice, found a " + net.classes[myid] + " and I'm " + str(myscore) + " sure")
                #detected.append(np.dot(bboxes[0][i].asnumpy(), resizeRatio))
                bbox = bboxes[0][i].asnumpy()
                
                # use this if you want fixed size boxes for the cow
                #x = int(((bbox[0] + bbox[2]) / 2) * resizeRatio)
                #y = int(((bbox[1] + bbox[3]) / 2) * resizeRatio)

                detected.append(bbox * resizeRatio)

    return detected


def extractCowsFromFrames(detected, directory):
    filePaths = []
    for filePath in glob.glob(directory + "/*.jpg"):
        filePaths.append(filePath)

    #crop the cows from the image
    for i in range(len(filePaths)):
        path = filePaths[i]
        img = cv2.imread(path)
        for j in range(len(detected)):
            cow = img[int(detected[j][1]):int(detected[j][3]), int(detected[j][0]):int(detected[j][2])]
            cv2.imwrite(directory + "/cow" + str(j) + "fromimg" + str(i) + ".jpg", cow)





net = model_zoo.get_model('faster_rcnn_resnet50_v1b_voc', pretrained=True)

sndLvlDirs = getSubdirs(PATH_TO_COW_DATA)
for directory in sndLvlDirs:
    if not os.path.exists(directory):
        os.mkdir(directory)

    pathGlobal = os.path.join(PATH_TO_COW_DATA, directory)
    for filePath in glob.glob(pathGlobal + "/M*.jpg"):
        extractFrames(filePath, directory)
        detectionFrame = directory + "/out-001.jpg"




file = "M00001.jpg"
#width, height = getInfo(file)
#print(width, height)

resizeRatio = extractFrames(file)
#print(np.shape(video))

######################################################################
# Load a pretrained model
# -------------------------
#
# Let's get an Faster RCNN model trained on Pascal VOC
# dataset with ResNet-50 backbone. By specifying
# ``pretrained=True``, it will automatically download the model from the model
# zoo if necessary. For more pretrained models, please refer to
# :doc:`../../model_zoo/index`.
#
# The returned model is a HybridBlock :py:class:`gluoncv.model_zoo.FasterRCNN`
# with a default context of `cpu(0)`.



######################################################################
# Pre-process an image
# --------------------
#
# Next we download an image, and pre-process with preset data transforms.
# The default behavior is to resize the short edge of the image to 600px.
# But you can feed an arbitrarily sized image.
#
# You can provide a list of image file names, such as ``[im_fname1, im_fname2,
# ...]`` to :py:func:`gluoncv.data.transforms.presets.rcnn.load_test` if you
# want to load multiple image together.
#
# This function returns two results. The first is a NDArray with shape
# `(batch_size, RGB_channels, height, width)`. It can be fed into the
# model directly. The second one contains the images in numpy format to
# easy to be plotted. Since we only loaded a single image, the first dimension
# of `x` is 1.
#
# Please beware that `orig_img` is resized to short edge 600px.





im_fname = "test/out-001.jpg"
x, orig_img = data.transforms.presets.rcnn.load_test(im_fname)


######################################################################
# Inference and display
# ---------------------
#
# The Faster RCNN model returns predicted class IDs, confidence scores,
# bounding boxes coordinates. Their shape are (batch_size, num_bboxes, 1),
# (batch_size, num_bboxes, 1) and (batch_size, num_bboxes, 4), respectively.
#
# We can use :py:func:`gluoncv.utils.viz.plot_bbox` to visualize the
# results. We slice the results for the first image and feed them into `plot_bbox`:

box_ids, scores, bboxes = net(x)

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
        if net.classes[myid] in acceptedClasses:
            print("Nice, found a " + net.classes[myid] + " and I'm " + str(myscore) + " sure")
            #detected.append(np.dot(bboxes[0][i].asnumpy(), resizeRatio))
            bbox = bboxes[0][i].asnumpy()
            
            # use this if you want fixed size boxes for the cow
            #x = int(((bbox[0] + bbox[2]) / 2) * resizeRatio)
            #y = int(((bbox[1] + bbox[3]) / 2) * resizeRatio)

            detected.append(bbox * resizeRatio)
            



# in case you want to plot the image, activate this line
#ax = utils.viz.plot_bbox(orig_img, bboxes[0], scores[0], box_ids[0], class_names=net.classes)


#read the filepaths
filePaths = []
for filePath in glob.glob("test/*.jpg"):
    filePaths.append(filePath)

#crop the cows from the image
for i in range(len(filePaths)):
    path = filePaths[i]
    img = cv2.imread(path)
    for j in range(len(detected)):
        cow = img[int(detected[j][1]):int(detected[j][3]), int(detected[j][0]):int(detected[j][2])]
        cv2.imwrite("detected/cow" + str(j) + "fromimg" + str(i) + ".jpg", cow)


#for i in range(6000):
  #if bboxes[0][i][0] != -1:
    #print("Line: " + str(i) + "\nBox is: " + str(bboxes[0][i])  + "\nScore is: " + str(scores[0][i]))

#plt.show()
