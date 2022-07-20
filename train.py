import argparse
from models.crnn import *
from utils.generals import *
from utils.data_utils import *
from utils.model_utils import *
import tensorflow as tf
from tensorflow import keras
import os
import matplotlib.pyplot as plt


def parse_opt():
    ap = argparse.ArgumentParser()

    ap.add_argument('--pretrained', default=None, type=str)
    ap.add_argument('--epochs', default=100, type=int)
    ap.add_argument('--batch_size', default=8, type=int)
    ap.add_argument('--train_data', default='data/data_samples_2', type=str)
    ap.add_argument('--val_data', default='data/private_test', type=str)
    ap.add_argument('--lr', default=1e-5, type=float)
    ap.add_argument('--reduce_lr_patience', default=3, type=int)
    ap.add_argument('--early_stop_patience', default=10, type=int)
    ap.add_argument('--target_height', default=96, type=int)
    ap.add_argument('--target_width', default=2048, type=int)
    ap.add_argument('--shuffle', default=True, type=bool)
    ap.add_argument('--cache', default=True, type=bool)

    opt = vars(ap.parse_args())

    return opt


def main(opt):

    info('GPU: ' + str(tf.config.list_physical_devices('GPU')))

    if opt['pretrained'] is not None:
        info(f"Loading pretrained model at {opt['pretrained']} ...")
        base_model = keras.models.load_model(opt['pretrained']).get_layer('crnn')
        info('Done')

        opt['target_height'] = base_model.input.shape[1]
    else:
        info("Loading new model...")
        base_model = get_base_model(
            height=opt['target_height'],
            vocab_size=CHAR_TO_NUM.vocabulary_size(),
            )
        info('Done')

    # given input shape (N, 96, 2048, 3), the output the
    # base_model will be (N, 256, vocab_size + 1).
    # so time_steps will be dim 1
    dummy_input = np.zeros((1, opt['target_height'], opt['target_width'], 3))
    time_steps = base_model(dummy_input).shape[1]

    print(time_steps)

    model = get_CTC_model(base_model)
    print(model.summary())

    train_dt = get_tf_dataset(
        img_dir=opt['train_data'],
        label_path=os.path.join(opt['train_data'], 'labels.json'),
        target_size=(opt['target_height'], opt['target_width']),
        time_steps=time_steps,
        batch_size=opt['batch_size'],
        shuffle=opt['shuffle'],
        cache=opt['cache']
    )

    val_dt = get_tf_dataset(
        img_dir=opt['val_data'],
        label_path=os.path.join(opt['val_data'], 'labels.json'),
        target_size=(opt['target_height'], opt['target_width']),
        time_steps=time_steps,
        batch_size=opt['batch_size'],
        shuffle=False,
        cache=opt['cache']
    )

    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=opt['lr'], momentum=0.9, nesterov=True),
    )

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=opt['reduce_lr_patience'], verbose=1),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=opt['early_stop_patience'], verbose=1, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(filepath='/content/drive/MyDrive/checkpoint/crnn', save_best_only=True, verbose=1), # saved_models
        # TODO custom metric
        CallbackEval(val_dt)
    ]

    history = model.fit(
        train_dt,
        epochs=opt['epochs'],
        shuffle=True,
        validation_data=val_dt,
        callbacks=callbacks
    )

    epoch_range = range(1, len(history.history['loss']) + 1)
    plt.plot(epoch_range, history.history['loss'], label='loss')
    plt.plot(epoch_range, history.history['val_loss'], label='val_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('loss.jpg')
    plt.show()

if __name__ == '__main__':

    opt = parse_opt()

    info('Arguments are set:')
    for k, v in opt.items():
        print(f'{k:24}: {v}')

    main(opt)



