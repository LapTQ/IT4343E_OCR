from models.crnn import *
from utils.datasets import *
import numpy as np
import matplotlib.pyplot as plt

get_model((133, 1925, 1), len(CHARACTERS)).summary()

# USE TF.DATA
# train_dataset = get_tf_dataset(
#     img_dir='data/data_samples_2',
#     label_path='data/data_samples_2/labels.json',
#     target_size=(133, 1925),
#     label_length=125,
#     batch_size=4,
#     grayscale=True,
#     invert_color=True,
#     dilate=2,
#     normalize=True
# )
# for imgs, labels in train_dataset.take(1):
#     pass



# # USE KERAS SEQUENCE
# dataset = AddressDataset(
#     img_dir='data/data_samples_2',
#     label_path='data/data_samples_2/labels.json',
#     target_size=(133, 1925),
#     label_length=125,
#     batch_size=4,
#     grayscale=True,
#     normalize=True
# )
# imgs, labels = next(iter(dataset))


# print(imgs.shape)
# print(labels.shape)
#
# plt.figure(figsize=(20, 6))
# i = 0
# for img, label in zip(imgs, labels):
#     plt.subplot(4, 1, i + 1)
#     label = tf.strings.reduce_join(NUM_TO_CHAR(label)).numpy().decode('utf-8')
#     plt.imshow(np.squeeze(img))
#     plt.title(label)
#     plt.axis('off')
#     plt.tight_layout()
#     i += 1
# plt.savefig('draft.jpg')
# plt.show()
