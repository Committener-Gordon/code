import ffmpeg
import numpy as np
import os

def get_data_paths(path):
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '.mp4' in file:
                files.append(os.path.join(r, file))            
    return files


data = get_data_paths("./")
correct = 0
short = 0
long = 0
corrupt = 0

for file in data:
    probe = ffmpeg.probe(file)
    info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
    width = int(info['width'])
    height = int(info['height'])
    if height != 400 or width != 400:
        os.remove(file)
        corrupt += 1
        continue

    out, _ = (
        ffmpeg
        .input(file)
        .output('pipe:', format='rawvideo', pix_fmt='gray')
        .run(capture_stdout=True)
    )
    video = (
        np
        .frombuffer(out, np.uint8)
        .reshape([-1, 400, 400, 1])
    )
    
    if len(video) == 20:
        correct += 1
    elif len(video) < 20:
        os.remove(file)
        short += 1
    else:
        new_file = file[:-4] + "cut.mp4"
        out, _ = (
            ffmpeg
            .input(file)
            .output(new_file, pix_fmt='gray', ss=0, t='10', r='2')
            .run(capture_stdout=True)
        )
        long += 1
        os.remove(file)
        
print("Correct: " + str(correct))
print("Short: " + str(short))
print("Long: " + str(long))
print("Corrupt: " + str(corrupt))
