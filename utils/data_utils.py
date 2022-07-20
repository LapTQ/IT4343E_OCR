import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import json
import os


H_AXIS = -3
W_AXIS = -2
C_AXIS = -1

CHARACTERS = [x for x in " !%'#&()*+,-./:;?0123456789AÁẢÀÃẠÂẤẨẦẪẬĂẮẲẰẴẶBCDĐEÉẺÈẼẸÊẾỂỀỄỆFGHIÍỈÌĨỊJKLMNOÓỎÒÕỌÔỐỔỒỖỘƠỚỞỜỠỢPQRSTUÚỦÙŨỤƯỨỬỪỮỰVWXYÝỶỲỸỴZaáảàãạâấẩầẫậăắẳằẵặbcdđeéẻèẽẹêếểềễệfghiíỉìĩịjklmnoóỏòõọôốổồỗộơớởờỡợpqrstuúủùũụưứửừữựvwxyýỷỳỹỵz"]

# for out-of-vocab token, use '' and the corresponding 0.
CHAR_TO_NUM = keras.layers.StringLookup(
    vocabulary=CHARACTERS,
    oov_token=""
)

NUM_TO_CHAR = keras.layers.StringLookup(
    vocabulary=CHAR_TO_NUM.get_vocabulary(),
    oov_token="",
    invert=True
)


def load_img(path):
    img_string = tf.io.read_file(path)
    img = tf.image.decode_png(img_string, channels=3)
    return img


def process_img_and_label(
        img,
        label,
        target_size,
        time_steps
):
    img = 255 - img

    target_height, target_width = target_size
    H, W = target_height, int(tf.shape(img)[W_AXIS] * target_height / tf.shape(img)[H_AXIS])
    img = tf.image.resize(img, (H, W))
    img = tf.pad(img, ((0, 0), (0, target_width - W), (0, 0)), mode='CONSTANT', constant_values=0)

    y_true = tf.strings.unicode_split(label, input_encoding='UTF-8')
    y_true = CHAR_TO_NUM(y_true)
    input_length = time_steps
    label_length = len(y_true)
    y_true = tf.pad(y_true, ((0, time_steps - label_length),), mode='CONSTANT', constant_values=0)

    return {'input_img': img, 'y_true': y_true, 'input_length': input_length, 'label_length': label_length}

def get_tf_dataset(
        img_dir,
        label_path,
        target_size,
        time_steps,
        batch_size=None,
        shuffle=False,
        cache=False
):

    # dataset = {img_name: label}
    dataset = json.load(open(label_path, 'r'))

    # dataset = {img_path: label}`
    dataset = {os.path.join(img_dir, img_name): label for img_name, label in dataset.items()}

    dataset = tf.data.Dataset.from_tensor_slices((dataset.keys(), dataset.values()))

    # dataset = [(img_array, label_string),...]
    dataset = dataset.map(lambda x, y: (load_img(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(lambda x, y: process_img_and_label(
        x,
        y,
        target_size,
        time_steps), num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.prefetch(buffer_size=500)
    if cache: dataset = dataset.cache()
    if shuffle: dataset = dataset.shuffle(500)
    if batch_size is not None: dataset = dataset.batch(batch_size)

    return dataset


# class AddressDataset(keras.utils.Sequence):
#     """Iterate over the data as Numpy array.
#     Reference: https://keras.io/examples/vision/oxford_pets_image_segmentation/
#     """
#
#     def __init__(
#             self,
#             img_dir,
#             label_path,
#             target_size,
#             grayscale,
#             time_steps,
#             batch_size=None,
#     ):
#         self.img_paths = [str(path) for path in Path(img_dir).glob('*.png')]
#         self.labels = json.load(open(label_path, 'r'))
#         self.target_size = target_size
#         self.grayscale = grayscale,
#         self.time_steps = time_steps
#         self.batch_size = batch_size
#
#         # self.labels = {os.path.join(img_dir, img_name): label for img_name, label in self.labels.items()}
#
#     def __len__(self):
#         return len(self.img_paths) // (self.batch_size if self.batch_size is not None else 1)
#
#     def __getitem__(self, idx):
#         """Return images in batch if batch_size is not None."""
#         img_num = self.batch_size if self.batch_size is not None else 1
#         i = img_num * idx
#         input_ = np.empty(shape=(img_num,) + self.target_size + ((3,) if not self.grayscale else (1,)),
#                      dtype=np.float32)
#         y_true = np.empty(shape=(img_num, self.time_steps), dtype=np.float32)
#         input_length = np.empty(shape=(img_num,), dtype=np.int32)
#         label_length = np.empty(shape=(img_num,), dtype=np.int32)
#
#         for j, path in enumerate(self.img_paths[i: i + img_num]):
#             img = load_img(path)
#             label = self.labels[path.split(os.path.sep)[-1]]
#             input_[j], y_true[j], input_length[j], label_length[j] = process_img_and_label(img, label, target_size=self.target_size, grayscale=self.grayscale, time_steps=self.time_steps)
#
#         if self.batch_size is None:
#             input_ = np.squeeze(input_, axis=0)
#             y_true = np.squeeze(y_true, axis=0)
#             input_length = np.squeeze(input_length, axis=0)
#             label_length = np.squeeze(label_length, axis=0)
#
#         return input_, y_true, input_length, label_length