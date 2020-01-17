import ffmpeg
import numpy as np
import os, sys
import re

videos = [
    '../calving/1_10_11/244.mp4',
    '../calving/1_10_15/106.mp4',
    '../calving/1_10_15/128.mp4',
    '../calving/1_10_15/135.mp4',
    '../calving/1_10_15/276.mp4',
    '../calving/1_10_15/376.mp4',
    '../calving/1_10_15/486.mp4',
    '../calving/1_10_15/780.mp4',
    '../calving/1_10_15/894tot.mp4',
    '../calving/1_10_15/911tot.mp4',
    '../calving/2_10_11/223.mp4',
    '../calving/2_10_11/386.mp4',
    '../calving/2_10_15/22.mp4',
    '../calving/2_10_15/154.mp4',
    '../calving/2_10_15/448.mp4',
    '../calving/2_10_15/462.mp4',
    '../calving/2_10_15/545.mp4',
    '../calving/2_10_15/680.mp4',
    '../calving/2_10_15/709.mp4',
    '../calving/2_10_15/866.mp4',
    '../calving/2_10_15/908.mp4',
    '../calving/2_10_15/958.mp4',
    
]

directories = [
    "244",
    "106",
    "128",
    "135",
    "276",
    "376",
    "486",
    "780",
    "894tot",
    "911tot",
    "223",
    "386",
    "22",
    "154",
    "448",
    "462",
    "545",
    "680",
    "709",
    "866",
    "908",
    "958"
]

def getDuration(path):
    probe = ffmpeg.probe(path)
    info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
    return int(float(info["duration"]))


def cutFromStartingPoint(inputFile, start, directory, fileName):
    out, _ = (
        ffmpeg
        .input(inputFile)
        .output(directory + "/" + fileName + ".mp4", pix_fmt='gray', ss=str(start), t='10')
        .run(capture_stdout=True)
    )
    

for i in range(len(videos)):
    video = videos[i]
    directory = directories[i]
    duration = getDuration(video)
    
    if not os.path.exists(directory):
            os.mkdir(directory)
    
    for i in range(0, duration, 10):
        videoName = "0000" + str(i)
        videoName = videoName[-5:-1]
        cutFromStartingPoint(video, i, directory, videoName)