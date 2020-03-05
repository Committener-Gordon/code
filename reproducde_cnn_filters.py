import ffmpeg
import numpy as np
from keras.models import Model, load_model
from matplotlib import pyplot as plt


cnn_model = load_model("stats/models/model_cnn_all_filter_8_run3.h5")
cnn_encoder = Model(cnn_model.inputs, cnn_model.layers[-1].output)

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

shape = predicted.shape

predicted = np.reshape(predicted, (shape[1], shape[2], shape[3]))

print(predicted.shape)


plt.figure(figsize=(10,10))
plt.subplot(2,4,1)
plt.imshow(predicted[:,:,0], cmap="gray")
plt.subplot(2,4,2)
plt.imshow(predicted[:,:,1], cmap="gray")
plt.subplot(2,4,3)
plt.imshow(predicted[:,:,2], cmap="gray")
plt.subplot(2,4,4)
plt.imshow(predicted[:,:,3], cmap="gray")
plt.subplot(2,4,5)
plt.imshow(predicted[:,:,4], cmap="gray")
plt.subplot(2,4,6)
plt.imshow(predicted[:,:,5], cmap="gray")
plt.subplot(2,4,7)
plt.imshow(predicted[:,:,6], cmap="gray")
plt.subplot(2,4,8)
plt.imshow(predicted[:,:,7], cmap="gray")
plt.rcParams["figure.figsize"] = (20,10)
plt.show()
