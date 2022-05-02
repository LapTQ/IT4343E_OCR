import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import json
import os

CHARACTERS = [x for x in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYabcdeghiklmnopqrstuvxyzÂÊÔàáâãèéêìíòóôõùúýăĐđĩũƠơưạảấầẩậắằẵặẻẽếềểễệỉịọỏốồổỗộớờởỡợụủỨứừửữựỳỵỷỹ#'()+,-./: "]

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

def dilate_img(img):
    """
    Grow a single image.
    :param img: numpy array of shape (H, W, C)
    :return: numpy array of shape (H, W, C)
    """

    kernel = tf.ones((3, 3, img.shape[-1]), dtype=img.dtype)
    # tf.nn.dilation2d works with batch of images, not a single image
    img = tf.nn.dilation2d(
        tf.expand_dims(img, axis=0),
        filters=kernel,
        strides=(1, 1, 1, 1),
        padding='SAME',
        data_format='NHWC',
        dilations=(1, 1, 1, 1)
    )[0]
    img = img - tf.ones_like(img)
    return img

def process_img(
        img,
        grayscale=False,
        invert_color=False,
        dilate=0,
        target_size=None,
        normalize=False,
        binarize=False,
        threshold=0.5
):
    """
    Arguments:
        img: array-like image of type int32 [0, 255]
    if normalize is False, output is unit8 [0, 255], else float32 [0., 1.].
    """

    if grayscale:
        # shape after convert (H, W, 1)
        img = tf.image.rgb_to_grayscale(img)

    if invert_color:
        img = 255 - img

    for _ in range(dilate):
        img = dilate_img(img)

    if target_size is not None:
        target_height, target_width = target_size

        # this function cast the output to be float [0., 255.]
        img = tf.image.resize_with_pad(
            img,
            target_height=target_height,
            target_width=target_width
        )#.numpy()

        # img = img.astype(np.uint8)
        img = tf.cast(img, tf.uint8)

    if normalize:
        img = img / 255

    if binarize:
        assert not normalize, 'Image is expected to be normalized before binarize.'
        img = tf.where(img > threshold, 1., 0.)

    return img

def process_label(label, target_length=None, padding_values=0):
    """

    :param label: string of transcript
    :return:
    """
    label = tf.strings.unicode_split(label, input_encoding='UTF-8')
    label = CHAR_TO_NUM(label)
    if target_length is not None:
        # pad label with padding_values to a unified length (padded_shapes)
        label = tf.pad(
            label,
            [[0, target_length - len(label)]],
            'CONSTANT',
            constant_values=padding_values
        )

    return label

class AddressDataset(keras.utils.Sequence):
    """Iterate over the data as Numpy array.
    Reference: https://keras.io/examples/vision/oxford_pets_image_segmentation/
    """

    def __init__(
            self,
            img_dir,
            label_path,
            target_size,
            label_length,
            batch_size=None,
            grayscale=False,
            invert_color=False,
            dilate=0,
            normalize=False,
            binarize=False,
            threshold=0.5
    ):
        self.img_paths = [str(path) for path in Path(img_dir).glob('*.png')]
        self.labels = json.load(open(label_path, 'r'))
        self.target_size = target_size
        self.label_length = label_length
        self.batch_size = batch_size
        self.invert_color = invert_color
        self.dilate = dilate
        self.grayscale = grayscale
        self.normalize = normalize
        self.binarize = binarize
        self.threshold = threshold

    def __len__(self):
        return len(self.img_paths) // (self.batch_size if self.batch_size is not None else 1)

    def __getitem__(self, idx):
        """Return images in batch if batch_size is not None."""
        img_num = self.batch_size if self.batch_size is not None else 1
        i = img_num * idx
        x = np.empty(shape=(img_num,) + self.target_size + ((3,) if not self.grayscale else (1,)),
                     dtype=np.uint8 if not self.normalize else np.float32)
        y = np.zeros(shape=(img_num, self.label_length), dtype=np.int32)
        for j, path in enumerate(self.img_paths[i: i + img_num]):
            img = load_img(path)
            label = self.labels[path.split(os.path.sep)[-1]]
            img = process_img(
                img,
                grayscale=self.grayscale,
                invert_color=self.invert_color,
                dilate=self.dilate,
                target_size=self.target_size,
                normalize=self.normalize,
                binarize=self.binarize,
                threshold=self.threshold
            )
            label = process_label(
                label,
                target_length=self.label_length,
                padding_values=0
            )
            x[j] = img
            y[j] = label

        if self.batch_size is None:
            x = np.squeeze(x, axis=0)
            y = np.squeeze(y, axis=0)

        return x, y

def get_tf_dataset(
        img_dir,
        label_path,
        target_size,
        label_length,
        batch_size=None,
        grayscale=False,
        invert_color=False,
        dilate=0,
        normalize=False,
        binarize=False,
        threshold=0.5,
        shuffle=False,
        cache=False
):

    # load annotation file: {img_name: label}
    dataset = json.load(open(label_path, 'r'))
    dataset = {os.path.join(img_dir, img_name): label for img_name, label in dataset.items()}

    dataset = tf.data.Dataset.from_tensor_slices(
        (dataset.keys(), dataset.values())
    )

    # dataset = [(img_array, label_string),...]
    dataset = dataset.map(lambda x, y: (load_img(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    # dataset = [(img_array, label_numbers),...]
    dataset = dataset.map(
        lambda x, y: (
            process_img(
                x,
                grayscale=grayscale,
                invert_color=invert_color,
                dilate=dilate,
                target_size=target_size,
                normalize=normalize,
                binarize=binarize,
                threshold=threshold
            ),
            process_label(
                y,
                target_length=label_length,
                padding_values=0)
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.prefetch(buffer_size=500)
    if cache:
        dataset = dataset.cache()
    if shuffle:
        dataset = dataset.shuffle(500)
    if batch_size is not None:
        dataset = dataset.batch(batch_size)

    return dataset