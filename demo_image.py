import cv2
import tensorflow as tf
from model.pspunet import pspunet
from data_loader.display import create_mask
import numpy as np
from PIL import Image
from keras.models import load_model
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=500)])
    except RuntimeError as e:
        print(e)

IMG_WIDTH = 480
IMG_HEIGHT = 272
n_classes = 7

model = load_model('pspunet_weight.h5')
img = cv2.imread('./surface_img/data1.jpeg')

img = cv2.resize(img, (IMG_WIDTH,IMG_HEIGHT))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img = img / 255

img = tf.expand_dims(img, 0)
print("===============")
print(img.shape)
pre = model.predict(img)
print("=====pre shape=========")
print(pre.shape)
pre = create_mask(pre).numpy()

print(pre)

frame2 = img/2


#frame2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2BGR)
#cv2.imwrite('./output/result.png', img1)