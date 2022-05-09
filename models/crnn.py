import tensorflow as tf
from tensorflow import keras

class CTCLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CTCLayer, self).__init__(**kwargs)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred, input_length, label_length):
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        return y_pred


def get_base_model(input_shape, vocab_size):
    inputs = keras.layers.Input(shape=input_shape)

    x = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x = keras.layers.MaxPooling2D((3, 3), strides=3)(x)
    x = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.MaxPooling2D((3, 3), strides=3)(x)
    x = keras.layers.Conv2D(256, (3, 3), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x_shortcut = x
    x = keras.layers.Conv2D(256, (3, 3), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Add()([x, x_shortcut])
    x = keras.layers.ReLU()(x)
    x = keras.layers.Conv2D(512, (3, 3), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x_shortcut = x
    x = keras.layers.Conv2D(512, (3, 3), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Add()([x, x_shortcut])
    x = keras.layers.ReLU()(x)
    x = keras.layers.Conv2D(1024, (3, 3), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.MaxPooling2D((3, 1), strides=(3, 1))(x)
    x = keras.layers.MaxPooling2D((3, 1), strides=(3, 1))(x)
    x = keras.layers.Reshape((x.shape[-2], x.shape[-1]))(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(512, return_sequences=True, dropout=0.2))(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(512, return_sequences=True, dropout=0.2))(x)

    outputs = keras.layers.Dense(vocab_size + 1, activation='softmax')(x)

    model = keras.Model(inputs, outputs)

    return model

def get_CTC_model(base_model):
    input_ = base_model.input
    y_pred = base_model.output

    # the length of label after padding is equal to time steps
    # to we get the 2nd dimention
    time_steps = y_pred.shape[1]

    y_true = keras.layers.Input(shape=(time_steps,), dtype=tf.float32)
    input_length = keras.layers.Input(shape=(1,), dtype=tf.int32)
    label_length = keras.layers.Input(shape=(1,), dtype=tf.int32)

    y_pred = CTCLayer()(y_true, y_pred, input_length, label_length)

    model = keras.Model(inputs=[input_, y_true, input_length, label_length], outputs=y_pred)

    return model









