import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from keras import backend


H_AXIS = -3
W_AXIS = -2
C_AXIS = -1


class CTCLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CTCLayer, self).__init__(**kwargs)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred, input_length, label_length):
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        return y_pred


def get_base_model2(height, vocab_size):
    input_ = keras.layers.Input(shape=(height, None, 3), name='input_img')

    x = keras.layers.Rescaling(1/255., name='rescale')(input_)

    # TODO thay doi kien truc -> mobilenet
    x = keras.layers.Conv2D(64, (3, 3), padding='same', name='conv_1')(x)
    x = tfa.activations.mish(x)
    x = keras.layers.MaxPooling2D((3, 3), strides=3, name='max_1')(x)
    x = keras.layers.Conv2D(128, (3, 3), padding='same', name='conv_2')(x)
    x = tfa.activations.mish(x)
    x = keras.layers.MaxPooling2D((3, 3), strides=3, name='max_2')(x)
    x = keras.layers.Conv2D(256, (3, 3), padding='same', name='conv3_3')(x)
    x = keras.layers.BatchNormalization(name='bn_1')(x)
    x = tfa.activations.mish(x)
    x_shortcut = x
    x = keras.layers.Conv2D(256, (3, 3), padding='same', name='conv_4')(x)
    x = keras.layers.BatchNormalization(name='bn_2')(x)
    x = keras.layers.Add(name='add_1')([x, x_shortcut])
    x = tfa.activations.mish(x)
    x = keras.layers.Conv2D(512, (3, 3), padding='same', name='conv_5')(x)
    x = keras.layers.BatchNormalization(name='bn_3')(x)
    x = tfa.activations.mish(x)
    x_shortcut = x
    x = keras.layers.Conv2D(512, (3, 3), padding='same', name='conv_6')(x)
    x = keras.layers.BatchNormalization(name='bn_4')(x)
    x = keras.layers.Add(name='add_2')([x, x_shortcut])
    x = tfa.activations.mish(x)
    x = keras.layers.Conv2D(1024, (3, 3), padding='same', name='conv_7')(x)
    x = keras.layers.BatchNormalization(name='bn_5')(x)
    x = tfa.activations.mish(x)
    x = keras.layers.MaxPooling2D((3, 1), strides=(3, 1), name='max_3')(x)
    x = keras.layers.MaxPooling2D((3, 1), strides=(3, 1), name='max_4')(x)

    x = keras.layers.Reshape((-1, x.shape[-1]), name='reshape')(x)

    # TODO check
    # x = keras.layers.Permute((2, 1, 3))(x)
    # x = keras.layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)

    # TODO attention layer
    for _ in range(2):
        x = keras.layers.Bidirectional(
            keras.layers.LSTM(
                512,
                return_sequences=True,
                dropout=0.2,
                name=f'lstm_{_ + 1}'),
            name=f'bdr_{_ + 1}'
        )(x)

    output = keras.layers.Dense(vocab_size + 1, activation='softmax', name='dense')(x)

    model = keras.Model(input_, output, name='crnn')

    return model

def get_CTC_model(base_model):
    input_ = base_model.input
    y_pred = base_model(input_)     # thay cho base_model.output

    # the length of label after padding is equal to time steps
    # to we get the 2nd dimention
    time_steps = y_pred.shape[1]

    y_true = keras.layers.Input(shape=(time_steps,), dtype=tf.float32, name='y_true')
    input_length = keras.layers.Input(shape=(1,), dtype=tf.int32, name='input_length')
    label_length = keras.layers.Input(shape=(1,), dtype=tf.int32, name='label_length')

    y_pred = CTCLayer()(y_true, y_pred, input_length, label_length)

    model = keras.Model(inputs=[input_, y_true, input_length, label_length], outputs=y_pred, name='CTC_model')

    return model

def _inverted_res_block(
    x,
    expansion,
    out_channels,
    kernel_size,
    strides,
    se_ratio,
    block_id,
):
    shortcut = x
    in_channels = backend.int_shape(x)[C_AXIS]

    x = keras.layers.Conv2D(
        in_channels * expansion,
        kernel_size=1,
        name=f'expand_{block_id}'
    )(x)

    x = keras.layers.BatchNormalization(
        axis=C_AXIS,
        name=f'expand/bn_{block_id}'
    )(x)

    x = tfa.activations.mish(x)

    # if stride

    x = keras.layers.DepthwiseConv2D(
        kernel_size,
        strides=strides,
        padding='same',
        name=f'depthwise_{block_id}'
    )(x)

    x = keras.layers.BatchNormalization(
        axis=C_AXIS,
        name=f'depthwise/bn_{block_id}'
    )(x)

    x = _se_block(x, in_channels * expansion, se_ratio, block_id)

    x = keras.layers.Conv2D(
        out_channels,
        kernel_size=1,
        name=f'project_{block_id}'
    )(x)

    x = keras.layers.BatchNormalization(
        axis=C_AXIS,
        name=f'project/bn_{block_id}'
    )(x)

    if strides == 1 and in_channels == out_channels:
        x = keras.layers.Add(name=f'add_{block_id}')([shortcut, x])

    return x


def _se_block(input_, in_channels, se_ratio, block_id):
    x = keras.layers.GlobalAveragePooling2D(
        keepdims=True,
        name=f'se/avgpool_{block_id}'
    )(input_)

    x = keras.layers.Conv2D(
        int(se_ratio * in_channels),
        kernel_size=1,
        name=f'se/conv1_{block_id}'
    )(x)

    x = tfa.activations.mish(x)

    x = keras.layers.Conv2D(
        in_channels,
        kernel_size=1,
        name=f'se/conv2_{block_id}'
    )(x)

    x = tfa.activations.mish(x)
    x = keras.layers.Multiply(name=f'se/mul_{block_id}')([input_, x])

    return x

def get_base_model(height, vocab_size):
    input_ = keras.layers.Input(shape=(height, None, 3), name='input_img')

    x = keras.layers.Rescaling(1/127.5, offset=-1, name='rescale')(input_)

    print(x.shape)

    x = keras.layers.Conv2D(
        16,
        kernel_size=5,
        strides=(2, 2),
        padding='same',
        name='init/conv'
    )(x)

    x = keras.layers.BatchNormalization(
        axis=C_AXIS,
        name='init/bn'
    )(x)

    x = tfa.activations.mish(x)

    print(x.shape)
    x = _inverted_res_block(x, expansion=1, out_channels=16, kernel_size=5, strides=1, se_ratio=0.25, block_id=1)
    print(x.shape)
    x = _inverted_res_block(x, expansion=4, out_channels=32, kernel_size=5, strides=2, se_ratio=0.25, block_id=2)
    print(x.shape)
    x = _inverted_res_block(x, expansion=3, out_channels=32, kernel_size=5, strides=1, se_ratio=0.25, block_id=3)
    print(x.shape)
    x = _inverted_res_block(x, expansion=3, out_channels=64, kernel_size=5, strides=2, se_ratio=0.25, block_id=4)
    print(x.shape)
    x = _inverted_res_block(x, expansion=3, out_channels=64, kernel_size=5, strides=1, se_ratio=0.25, block_id=5)
    print(x.shape)
    x = _inverted_res_block(x, expansion=3, out_channels=64, kernel_size=5, strides=1, se_ratio=0.25, block_id=6)
    print(x.shape)
    # x = _inverted_res_block(x, expansion=6, out_channels=80, kernel_size=5, strides=(2, 1), se_ratio=0.25, block_id=7)
    x = keras.layers.Conv2D(filters=128, kernel_size=5, strides=(2, 1), padding='same', name='conv_7')(x)
    x = _se_block(x, in_channels=128, se_ratio=0.25, block_id=7)
    # print(x.shape)
    x = _inverted_res_block(x, expansion=2.5, out_channels=128, kernel_size=5, strides=1, se_ratio=0.25, block_id=8)
    print(x.shape)
    x = _inverted_res_block(x, expansion=2.3, out_channels=128, kernel_size=5, strides=1, se_ratio=0.25, block_id=9)
    print(x.shape)
    # x = _inverted_res_block(x, expansion=6, out_channels=112, kernel_size=5, strides=(2, 1), se_ratio=0.25, block_id=10)
    x = keras.layers.Conv2D(filters=256, kernel_size=5, strides=(2, 1), padding='same', name='conv_10')(x)
    x = _se_block(x, in_channels=256, se_ratio=0.25, block_id=10)
    print(x.shape)
    x = _inverted_res_block(x, expansion=6, out_channels=256, kernel_size=5, strides=1, se_ratio=0.25, block_id=11)
    print(x.shape)
    # x = _inverted_res_block(x, expansion=6, out_channels=160, kernel_size=5, strides=(3, 1), se_ratio=0.25, block_id=12)
    x = keras.layers.Conv2D(filters=512, kernel_size=5, strides=(3, 1), padding='same', name='conv_12')(x)
    x = _se_block(x, in_channels=512, se_ratio=0.25, block_id=12)
    print(x.shape)

    x = keras.layers.Reshape((-1, x.shape[-1]), name='reshape')(x)

    # TODO check
    # x = keras.layers.Permute((2, 1, 3))(x)
    # x = keras.layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)

    # TODO attention layer
    for _ in range(2):
        x = keras.layers.Bidirectional(
            keras.layers.LSTM(
                512,
                return_sequences=True,
                dropout=0.2,
                name=f'lstm_{_ + 1}'),
            name=f'bdr_{_ + 1}'
        )(x)

    output = keras.layers.Dense(vocab_size + 1, activation='softmax', name='dense')(x)

    model = keras.Model(input_, output, name='crnn')

    return model

    # model = keras.Model(input_, x, name='crnn')
    #
    # return model




