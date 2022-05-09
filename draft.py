import tensorflow as tf
from tensorflow import keras

from models.crnn import *
from utils.datasets import *
import numpy as np
from utils.generals import *
from utils.preprocessing import *
import matplotlib.pyplot as plt
import json

# train_dataset = AddressDataset(
#     img_dir='data/data_samples_2',
#     label_path='data/data_samples_2/labels.json',
#     target_size=(118, 2167),
#     grayscale=True,
#     time_steps=240,
#     batch_size=4,
# )
#
# inputs, y_true, input_length, label_length = next(iter(train_dataset))
#
# print(inputs.shape)
# print(y_true.shape)
# print(input_length.shape)
# print(label_length.shape)
#
# plt.figure(figsize=(20, 8))
# i = 0
# for img, label, input_len, label_len in zip(inputs, y_true, input_length, label_length):
#     print(label)
#     print(input_len)
#     print(label_len)
#     plt.subplot(4, 1, i + 1)
#     label = tf.strings.reduce_join(NUM_TO_CHAR(label)).numpy().decode('utf-8')
#     plt.imshow(np.squeeze(img))
#     plt.title(label)
#     # plt.axis('off')
#     plt.tight_layout()
#     i += 1
# plt.savefig('draft.jpg')
# plt.show()

# img_path = 'data/data_samples_1/1.jpg'
# img = plt.imread(img_path)
# plt.imshow(img)
# plt.show()
# # h, w, c = img.shape
# # img = tf.image.resize(img, (118, int(w * 118/h)))
# # h, w, c = img.shape
# # img = np.pad(img, ((0, 0), (0, 2167 - w), (0, 0)), mode='median')
# img = Resize(118, 2167)(img)
# plt.imshow(tf.cast(img, tf.uint8))
# plt.show()
# img = RGB2Gray()(img)
# plt.imshow(tf.cast(img, tf.uint8))
# plt.show()

base_model = get_base_model((118, 2202, 1), 215)
# model = get_CTC_model(base_model)
print(base_model.summary())
# print(model.summary())
#
# i = iter(train_dataset)
# for _ in range(2):
#     dt = model(next(i))

