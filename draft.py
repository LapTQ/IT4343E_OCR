import tensorflow as tf
from tensorflow import keras

from models.crnn import *
from utils.data_utils import *
import numpy as np
from utils.generals import *
import matplotlib.pyplot as plt
import json

import cv2

img = cv2.imread('data/data_samples_1/1.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

img = cv2.resize(img, (2048, 96))

img = np.expand_dims(img, axis=0)

base_model = get_base_model(96, 214)
# model = get_CTC_model(base_model)
print(base_model.summary())
#
# time_steps = base_model.output.shape[1]
#
# train_dataset = get_tf_dataset(
#     img_dir='data/data_samples_2',
#     label_path='data/data_samples_2/labels.json',
#     target_size=(118, 2202),
#     # grayscale=True,
#     # invert_color=True,
#     time_steps=time_steps,
#     batch_size=4,
#     # shuffle=True
# )
# for batch in train_dataset.take(1):
#     pass
#
#
# input_img, y_true, input_length, label_length = batch.values()
# input_img = model.predict(input_img)
#
# # print(input_img)
# # print(y_true.shape)
# # print(input_length.shape)
# # print(label_length.shape)
#
# plt.figure(figsize=(20, 8))
# i = 0
# for img, label, input_len, label_len in zip(input_img, y_true, input_length, label_length):
#     plt.subplot(4, 1, i + 1)
#     label = tf.strings.reduce_join(NUM_TO_CHAR(label)).numpy().decode('utf-8')
#     plt.imshow(np.squeeze(img), cmap='gray')
#     plt.title(label)
#     # plt.axis('off')
#     plt.tight_layout()
#     i += 1
# plt.savefig('draft.jpg')
# plt.show()
#
# # img_path = 'data/data_samples_1/1.jpg'
# # img = plt.imread(img_path)
# # plt.imshow(img)
# # plt.show()
# # # h, w, c = img.shape
# # # img = tf.image.resize(img, (118, int(w * 118/h)))
# # # h, w, c = img.shape
# # # img = np.pad(img, ((0, 0), (0, 2167 - w), (0, 0)), mode='median')
# # img = Resize(118, 2167)(img)
# # plt.imshow(tf.cast(img, tf.uint8))
# # plt.show()
# # img = RGB2Gray()(img)
# # plt.imshow(tf.cast(img, tf.uint8))
# # plt.show()



# model = get_CTC_model(base_model)
print(base_model.predict(img).shape)







