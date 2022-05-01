import tensorflow as tf
from tensorflow import keras

def get_model(image_shape, vocab_size):
    input = keras.layers.Input(image_shape)
    x = keras.layers.Conv2D(
        filters=32,
        kernel_size=(11, 41),
        strides=(2, 2),
        padding='same',
        use_bias=False
    )(input)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Conv2D(
        filters=32,
        kernel_size=(11, 21),
        strides=(1, 2),
        padding='same',
        use_bias=False
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    # reshape to feed to RNNs
    x = keras.layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)
    rnn_layers = 5
    for i in range(1, rnn_layers + 1):
        x = keras.layers.Bidirectional(
            keras.layers.GRU(
                units=128,
                activation='tanh',
                recurrent_activation='sigmoid',
                use_bias=True,
                return_sequences=True,
                reset_after=True
            ),
            merge_mode='concat'
        )(x)
        if i < rnn_layers:
            x = keras.layers.Dropout(rate=0.5)(x)
    x = keras.layers.Dense(256)(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Dropout(rate=0.5)(x)
    # why +1?
    output = keras.layers.Dense(vocab_size + 1, activation='softmax')(x)

    return keras.Model(input, output)

