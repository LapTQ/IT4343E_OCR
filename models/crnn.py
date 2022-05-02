import tensorflow as tf
from tensorflow import keras

def get_model(image_shape, vocab_size):
    # must be None???
    input = keras.layers.Input((image_shape[0], None, image_shape[2]))
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
        strides=(1, 1),#(1, 2),
        padding='same',
        kernel_initializer='he_normal'
        # use_bias=False
    )(x)
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
    output = keras.layers.Dense(vocab_size + 1, activation='softmax')(x)

    return keras.Model(input, output)

