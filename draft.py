from models.crnn import *
from utils.datasets import *
import numpy as np
import matplotlib.pyplot as plt

# a = tf.pad(
#     np.array([1, 2, 3, 4]),
#     [[0, 5]],
#     'CONSTANT',
#     constant_values=0
# )
# print(a)

# USE TF.DATA
train_dataset = get_tf_dataset(
    img_dir='data/data_samples_2',
    target_size=(69, 773),
    label_length=125,
    batch_size=4,
    grayscale=True,
    normalize=True
)

plt.figure(figsize=(40, 12))
i = 0
for imgs, labels in train_dataset.take(1):
    print(imgs.shape)
    print(labels.shape)

    for img, label in zip(imgs, labels):
        plt.subplot(4, 1, i + 1)
        plt.figure(figsize=(40, 3))
        label = tf.strings.reduce_join(NUM_TO_CHAR(label)).numpy().decode('utf-8')
        plt.imshow(img)
        plt.title(label)
        plt.axis('off')
        plt.tight_layout()
        i += 1
plt.show()


# # USE KERAS SEQUENCE
# dataset = AddressDataset(
#     img_dir='data/data_samples_2',
#     target_size=(133, 1925),
#     batch_size=4,
#     grayscale=True,
#     normalize=True
# )
#
# imgs = next(iter(dataset))
# print(imgs.shape)
# plt.figure(figsize=(40, 3))
# plt.imshow(imgs[0])
# plt.axis('off')
# plt.tight_layout()
# plt.show()


















# a = 60
# a = (a-1)*2+1
# a = (a-1)*2+1
# a = (a-1)*2+1
# a = (a-1)*2+3
# a = (a-1)*2+7
# print(a)

# print(
#     tf.image.rgb_to_grayscale(
#         plt.imread('data/data_samples_1/1.jpg')
#     ).shape
# )