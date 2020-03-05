import ffmpeg
import numpy as np
from keras.models import Model, load_model
from matplotlib import pyplot as plt


cnn_model = load_model("model_cnn_run7.h5")
cnn_encoder = Model(cnn_model.inputs, cnn_model.layers[-5].output)

out, _ = (
    ffmpeg
    .input("calving/958/0192.mp4")
    .output('pipe:', format='rawvideo', pix_fmt='gray')
    .run(quiet=True)
)
video = (
    np
    .frombuffer(out, np.uint8)
    .reshape([-1, 400, 400, 1])
)

frame = video[0]/255

frame = np.reshape(frame, (1, 400, 400, 1))
show_frame = np.reshape(frame, (400, 400))
#plt.imshow(show_frame, cmap="gray")
#plt.show()


predicted = cnn_encoder.predict(frame)
emb_size = predicted.size
emb_length = emb_size ** .5

predicted = np.reshape(predicted, (int(emb_length), int(emb_length)))


plt.imshow(predicted, cmap="gray")
plt.show()
