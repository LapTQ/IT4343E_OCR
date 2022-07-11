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


def get_base_model(height, vocab_size):
    input_ = keras.layers.Input(shape=(height, None, 3), name='input_img')

    x = keras.layers.Rescaling(1/255., name='rescale')(input_)

    # TODO thay doi kien truc -> mobilenet
    x = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='conv_1')(x)
    x = keras.layers.MaxPooling2D((3, 3), strides=3, name='max_1')(x)
    x = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='conv_2')(x)
    x = keras.layers.MaxPooling2D((3, 3), strides=3, name='max_2')(x)
    x = keras.layers.Conv2D(256, (3, 3), padding='same', name='conv3_3')(x)
    x = keras.layers.BatchNormalization(name='bn_1')(x)
    x = keras.layers.ReLU()(x)
    x_shortcut = x
    x = keras.layers.Conv2D(256, (3, 3), padding='same', name='conv_4')(x)
    x = keras.layers.BatchNormalization(name='bn_2')(x)
    x = keras.layers.Add(name='add_1')([x, x_shortcut])
    x = keras.layers.ReLU()(x)
    x = keras.layers.Conv2D(512, (3, 3), padding='same', name='conv_5')(x)
    x = keras.layers.BatchNormalization(name='bn_3')(x)
    x = keras.layers.ReLU()(x)
    x_shortcut = x
    x = keras.layers.Conv2D(512, (3, 3), padding='same', name='conv_6')(x)
    x = keras.layers.BatchNormalization(name='bn_4')(x)
    x = keras.layers.Add(name='add_2')([x, x_shortcut])
    x = keras.layers.ReLU()(x)
    x = keras.layers.Conv2D(1024, (3, 3), padding='same', name='conv_7')(x)
    x = keras.layers.BatchNormalization(name='bn_5')(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.MaxPooling2D((3, 1), strides=(3, 1), name='max_3')(x)
    x = keras.layers.MaxPooling2D((3, 1), strides=(3, 1), name='max_4')(x)

    x = keras.layers.Reshape((-1, x.shape[-1]), name='reshape')(x)

    # TODO check
    # x = keras.layers.Permute((2, 1, 3))(x)
    # x = keras.layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)

    # TODO attention layer

    x = keras.layers.Bidirectional(keras.layers.LSTM(512, return_sequences=True, dropout=0.2, name='lstm_1'), name='bdr_1')(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(512, return_sequences=True, dropout=0.2, name='lstm_2'), name='bdr_2')(x)
    output = keras.layers.Dense(vocab_size + 1, activation='softmax', name='dense')(x)

    model = keras.Model(input_, output)

    return model

def get_CTC_model(base_model):
    input_ = base_model.input
    y_pred = base_model.output

    # the length of label after padding is equal to time steps
    # to we get the 2nd dimention
    time_steps = y_pred.shape[1]

    y_true = keras.layers.Input(shape=(time_steps,), dtype=tf.float32, name='y_true')
    input_length = keras.layers.Input(shape=(1,), dtype=tf.int32, name='input_length')
    label_length = keras.layers.Input(shape=(1,), dtype=tf.int32, name='label_length')

    y_pred = CTCLayer()(y_true, y_pred, input_length, label_length)

    model = keras.Model(inputs=[input_, y_true, input_length, label_length], outputs=y_pred)

    return model









