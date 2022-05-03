import keras.backend

from models.crnn import *
from utils.datasets import *
import numpy as np
from utils.generals import *
import matplotlib.pyplot as plt
import json

model = get_model((124, 1900, 1), CHAR_TO_NUM.vocabulary_size(), option=2)
print(model.summary())

# USE TF.DATA
train_dataset = get_tf_dataset(
    img_dir='data/data_samples_2',
    label_path='data/data_samples_2/labels.json',
    target_size=(124, 1900),
    label_length=80,
    batch_size=4,
    grayscale=True,
    invert_color=True,
    # dilate=2,
    normalize=True
)
for images, labels in train_dataset.take(1):
    pass



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


print(images.shape)
print(labels.shape)

# plt.figure(figsize=(20, 8))
# i = 0
# for img, label in zip(images, labels):
#     # print(label)
#     plt.subplot(4, 1, i + 1)
#     label = tf.strings.reduce_join(NUM_TO_CHAR(label)).numpy().decode('utf-8')
#     plt.imshow(np.squeeze(img))
#     plt.title(label)
#     # plt.axis('off')
#     plt.tight_layout()
#     i += 1
# plt.savefig('draft.jpg')
# plt.show()

targets = []
preds = model.predict(images)
print(CTCLoss(labels, preds))
print(preds.shape)
preds = decode_batch_predictions(preds)
for label in labels:
    label = tf.strings.reduce_join(NUM_TO_CHAR(label)).numpy().decode('utf-8')
    targets.append(label)
for t, p in zip(targets, preds):
    print(f'T: {t}')
    print(f'P: {p}')


# chars = ['a', 'b', 'c']
# ctn = keras.layers.StringLookup(vocabulary=chars, oov_token='')
# ntc = keras.layers.StringLookup(vocabulary=chars, oov_token='', invert=True)
#
# def decode_batch_predictions(pred):
#     input_len = np.ones(pred.shape[0]) * pred.shape[1]
#     # Use greedy search. For complex tasks, you can use beam search
#     results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
#     # Iterate over the results and get back the text
#     output_text = []
#     for result in results:
#         result = tf.strings.reduce_join(ntc(result)).numpy().decode("utf-8")
#         output_text.append(result)
#     return output_text
# def CTCLoss(y_true, y_pred):
#     # Compute the training-time loss value
#     batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
#     input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
#     label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
#
#     input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
#     label_length = tf.math.count_nonzero(y_true, axis=-1, keepdims=True, dtype="int64") #label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
#     loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
#     return loss
#
# def process_label(label, target_length=None, padding_values=0):
#     """
#
#     :param label: string of transcript
#     :return:
#     """
#     label = tf.strings.unicode_split(label, input_encoding='UTF-8')
#     label = ctn(label)
#     if target_length is not None:
#         # pad label with padding_values to a unified length (padded_shapes)
#         label = tf.pad(
#             label,
#             [[0, target_length - len(label)]],
#             'CONSTANT',
#             constant_values=padding_values
#         )
#
#     return label
#
# pred = np.array([
#     [[0.1, 0.5, 0.3, 0.1],
#      [0.2, 0.5, 0.2, 0.2],
#      [0.1, 0.1, 0.1, 0.7],
#      [0.1, 0.2, 0.5, 0.2],
#      [0.1, 0.2, 0.5, 0.2],
#      [0.1, 0.5, 0.3, 0.1],
#      [0.2, 0.5, 0.2, 0.2],
#      [0.1, 0.1, 0.1, 0.7],
#      [0.2, 0.5, 0.2, 0.2]
#      ]
# ])
# label = 'abaa'
# label = process_label(label)
# label = np.expand_dims(label, axis=0)
# label = np.array([[1, 2, 1, 1, 0, 0, 0, 0, 0]])
# print(label)
#
# a = decode_batch_predictions(pred)
# loss = CTCLoss(label, pred)
# print(a)
# print(loss)


