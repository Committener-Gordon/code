import ffmpeg
import numpy as np

probe = ffmpeg.probe('calving/22/0000.mp4')
video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
width = int(video_stream['width'])
height = int(video_stream['height'])

out, _ = (
    ffmpeg
    .input('calving/22/0000.mp4')
    .output('pipe:', format='rawvideo', pix_fmt='gray')
    .run(capture_stdout=True)
)
video = (
    np
    .frombuffer(out, np.uint8)
    .reshape([-1, height, width, 1])
)

frame = video[0]
print(np.shape(frame))
