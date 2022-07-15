import tensorflow as tf
from tensorflow import keras

from models.crnn import *
from utils.data_utils import *
import numpy as np
from utils.generals import *
import matplotlib.pyplot as plt
import json

import cv2
#
# img = cv2.imread('data/data_samples_1/1.jpg')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#
# img = cv2.resize(img, (2048, 96))

# img = np.expand_dims(img, axis=0)

dummy_input = np.zeros((1, 96, 2048, 3))

base_model = get_base_model(96, 214)
output = base_model(dummy_input)
time_steps = output.shape[1]
model = get_CTC_model(base_model)
# print(base_model.summary())


train_dataset = get_tf_dataset(
    img_dir='data/data_samples_1',
    label_path='data/data_samples_1/labels.json',
    target_size=(96, 2048),
    time_steps=time_steps,
    batch_size=4,
)
for batch in train_dataset.take(1):
    pass

print(batch.values())

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


print(output.shape)







