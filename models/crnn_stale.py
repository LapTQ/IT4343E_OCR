import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, applications, models
from tensorflow.keras import backend as K


letters = " #'()+,-./:0123456789ABCDEFGHIJKLMNOPQRSTUVWXYabcdeghiklmnopqrstuvxyzÂÊÔàáâãèéêìíòóôõùúýăĐđĩũƠơưạảấầẩậắằẵặẻẽếềểễệỉịọỏốồổỗộớờởỡợụủỨứừửữựỳỵỷỹ"
MAX_LEN = 70
SIZE = 2560, 160
CHAR_DICT = len(letters) + 1

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def get_model(input_shape, training, finetune):
    inputs = layers.Input(shape=input_shape, dtype='float32')
    base_model = applications.VGG16(include_top=False, weights='imagenet')
    # output shape (N, H, W, C)
    output = base_model(inputs)
    # output shape (N, H, W*C)
    output = layers.Reshape((output.shape[1], -1))(output)
    # output shape (N, H, 512)
    output = layers.Dense(units=512, activation='relu', kernel_initializer='he_normal')(output)
    output = layers.Dropout(0.25)(output)
    output = layers.Bidirectional(
        # https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
        layers.LSTM(
            units=512,
            return_sequences=True,
            kernel_initializer='he_normal',
            dropout=0.25,
            recurrent_dropout=0.25,
        )
    )(output)

    # y_pred shape (N, H, CHAR_DICT)?
    y_pred = layers.Dense(CHAR_DICT, activation='softmax', kernel_initializer='he_normal')(output)
    labels = layers.Input(shape=(MAX_LEN), dtype='float32')
    input_length = layers.Input(shape=(1,), dtype='int64')
    label_length = layers.Input(shape=(1,), dtype='int64')

    loss = layers.Lambda(ctc_lambda_func, output_shape=(1,))([y_pred, labels, input_length, label_length])

    for layer in base_model.layers:
        layer.trainable = finetune

    y_func = K.function([inputs], [y_pred])

    if training:
        return models.Model(inputs=[inputs, labels, input_length, label_length], outputs=loss), y_func
    else:
        return models.Model(inputs=[inputs], outputs=y_pred)

print(get_model((120, 1900, 3), training=True, finetune=False)[0].summary())

def train(datapath, labelpath, epoch):
    pass

class Dataset(keras.utils.Sequence):
    pass




#
# def train_kfold(idx, kfold, datapath, labelpath, epochs, batch_size, lr, finetune):
#     sess = tf.Session()
#     K.set_session(sess)
#
#     model, y_func = get_model((*SIZE, 3), training=True, finetune=finetune)
#     ada = Adam(lr=lr)
#     model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=ada)
#
#     ## load data
#     train_idx, valid_idx = kfold[idx]
#     train_generator = TextImageGenerator(datapath, labelpath, *SIZE, batch_size, 32, train_idx, True, MAX_LEN)
#     train_generator.build_data()
#     valid_generator = TextImageGenerator(datapath, labelpath, *SIZE, batch_size, 32, valid_idx, False, MAX_LEN)
#     valid_generator.build_data()
#
#     ## callbacks
#     weight_path = 'model/best_%d.h5' % idx
#     ckp = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
#     vis = VizCallback(sess, y_func, valid_generator, len(valid_idx))
#     earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min')
#
#     if finetune:
#         print('load pretrain model')
#         model.load_weights(weight_path)
#
#     model.fit_generator(generator=train_generator.next_batch(),
#                         steps_per_epoch=int(len(train_idx) / batch_size),
#                         epochs=epochs,
#                         callbacks=[ckp, vis, earlystop],
#                         validation_data=valid_generator.next_batch(),
#                         validation_steps=int(len(valid_idx) / batch_size))
#
#
# def train(datapath, labelpath, epochs, batch_size, lr, finetune=False):
#     nsplits = 5
#
#     nfiles = np.arange(len(os.listdir(datapath)))
#
#     kfold = list(KFold(nsplits, random_state=2018).split(nfiles))
#     for idx in range(nsplits):
#         train_kfold(idx, kfold, datapath, labelpath, epochs, batch_size, lr, finetune)
#


