import ffmpeg
import numpy as np
import os, sys
import re

PATH_BACKUP_ONE_10_11 = "/cow_data/Kamerasicherung1/Kamerasicherung1/mx10-11-175-103/10_11_175_103/"
PATH_BACKUP_ONE_10_15 = "/cow_data/Kamerasicherung1/Kamerasicherung1/mx10-15-178-224/10_15_178_224/"
PATH_BACKUP_TWO_10_11 = "/cow_data/Kamerasicherung1/Kamerasicherung2/mx10-11-175-103/"
PATH_BACKUP_TWO_10_15 = "/cow_data/Kamerasicherung2/Kamerasicherung2/mx10-15-178-224/"

VIDEO_RANGE = 100

startingPoints = [{
    "superDir": 158,
    "subDir": 244
}]
   
def getVideoFilesFromDir(directory):
    filenames = os.listdir(directory)
    pattern = re.compile("^M[0-9]+.jpg")
    videos = []
    for filename in filenames: 
        if pattern.match(filename):
            videos.append(filename)   
    return videos

def getInfo(inputFile):
    probe = ffmpeg.probe(inputFile)
    info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
    width = int(info['width'])
    height = int(info['height'])
    return width, height
    
def getFolders(path, superDir, subDir):
    directories = []
    for i in range(VIDEO_RANGE):
        if subDir >= 1000:
            subDir -= 1000
            superDir += 1
        subDirString = ("00" + str(subDir))[-3:]
        superDirString = ("00" + str(superDir))[-3:]
        directories.append(path + str(superDirString) + "/" + str(subDirString))
        print(directories[i])
        subDir += 1
    return directories
      
    
for startingPoint in startingPoints:
    folders = getFolders(PATH_BACKUP_ONE_10_11, startingPoint["superDir"], startingPoint["subDir"])

    stream = None

    for key, folder in enumerate(folders):
        videoFiles = getVideoFilesFromDir(folder)
        videoFiles.sort()
        videos = []


        for video in videoFiles:
            width, height = getInfo(folder + "/" + video)
            videos.append(ffmpeg.input(folder + "/" + video, f='mxg').crop(width/2, height, width/2, height))

        for i in range(len(videos)):
            if not stream:
                stream = videos[i]
            else: 
                stream = ffmpeg.concat(stream, videos[i])
            print(str(key) + " " + folder)

    stream = ffmpeg.output(stream, str(startingPoint["subDir"]) + ".mp4", r='2')
    ffmpeg.run(stream)    