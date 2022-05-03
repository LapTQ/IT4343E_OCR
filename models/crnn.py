import tensorflow as tf
from tensorflow import keras
from utils.losses import *

def get_model(image_shape, vocab_size, option=2):
    # must be None???
    input = keras.layers.Input(shape=(image_shape[0], None, image_shape[2]), name='image')

    if option == 1:
        x = keras.layers.Conv2D(
            filters=32,
            kernel_size=(11, 41),
            strides=(2, 2),
            padding='same',
            kernel_initializer='he_normal'
            # use_bias=False
        )(input)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU()(x)
        x = keras.layers.Conv2D(
            filters=32,
            kernel_size=(11, 21),
            strides=(1, 2),
            padding='same',
            kernel_initializer='he_normal'
            # use_bias=False
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU()(x)

    elif option == 2:
        x = keras.layers.Conv2D(filters=16, kernel_size=(4, 14), strides=2, padding='same',
                                kernel_initializer='he_normal')(input)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU()(x)

        x = keras.layers.Conv2D(filters=32, kernel_size=(5, 14), padding='same', kernel_initializer='he_normal')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU()(x)

        x = keras.layers.Conv2D(filters=64, kernel_size=(5, 14), strides=2, padding='same',
                                kernel_initializer='he_normal')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU()(x)

        x = keras.layers.Conv2D(filters=128, kernel_size=(5, 14), padding='same', kernel_initializer='he_normal')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU()(x)

        x = keras.layers.Conv2D(filters=256, kernel_size=(5, 14), strides=2, padding='same',
                                kernel_initializer='he_normal')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU()(x)


    x = keras.layers.Permute((2, 1, 3))(x)
    # reshape to feed to RNNs
    x = keras.layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)
    rnn_layers = 4
    rnn_units = 128
    for i in range(1, rnn_layers + 1):
        x = keras.layers.Bidirectional(
            keras.layers.GRU(
                units=rnn_units,
                activation='tanh',
                recurrent_activation='sigmoid',
                use_bias=True,
                return_sequences=True,
                reset_after=True,
                name=f"gru_{i}",
            ),
            name=f"bidirectional_{i}",
            merge_mode='concat'
        )(x)
        if i < rnn_layers:
            x = keras.layers.Dropout(rate=0.2)(x)
    x = keras.layers.Dense(2 * rnn_units)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dropout(rate=0.2)(x)
    # +1 for the blank token: oov, 'a', 'b',..., blank
    # +2 is to account for the two special tokens introduced by the CTC loss.
    # The recommendation comes here: https://git.io/J0eXP.
    x = keras.layers.Dense(vocab_size + 1, activation='softmax')(x)
    labels = keras.layers.Input(shape=(None,), name='label')
    output = CTCLayer()(labels, x)

    model = keras.Model(inputs=[input, labels], outputs=output)

    return model

