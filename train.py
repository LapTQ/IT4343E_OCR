import argparse
from models.crnn import *
from utils.generals import *
from utils.datasets import *
from utils.callbacks import *
import tensorflow as tf
from tensorflow import keras
import os
import matplotlib.pyplot as plt


def parse_opt():
    ap = argparse.ArgumentParser()

    ap.add_argument('--pretrained', default=None, type=str)
    ap.add_argument('--epochs', default=100, type=int)
    ap.add_argument('--batch_size', default=32, type=int)
    ap.add_argument('--train_data', default='data/data_samples_2', type=str)
    ap.add_argument('--val_data', default='data/private_test', type=str)
    ap.add_argument('--lr', default=1e-3, type=float)
    ap.add_argument('--reduce_lr_patience', default=4, type=int)
    ap.add_argument('--early_stop_patience', default=10, type=int)
    # ap.add_argument('--target_height', default=118, type=int)
    # ap.add_argument('--target_width', default=2202, type=int)
    ap.add_argument('--grayscale', default=True, type=bool)
    ap.add_argument('--invert_color', default=True, type=bool)
    ap.add_argument('--dilate', default=0, type=int)
    ap.add_argument('--shuffle', default=True, type=bool)
    ap.add_argument('--cache', default=True, type=bool)

    opt = vars(ap.parse_args())

    return opt


def run(opt):

    if opt['pretrained'] is not None:
        info(f"Loading pretrained model at {opt['pretrained']} ...")
        base_model = keras.models.load_model(opt['pretrained'])
        info('Done')

        _, target_height, target_width, depth = base_model.input.shape
        opt['grayscale'] = True if 'rgb2_gray' in [layer.name for layer in base_model.layers] else False
        print(f'[LOG] Image will be resized to configuration of pretrained model: {(target_height, target_width, depth)}')
    else:
        target_height, target_width = 118, 2202
        input_shape = (target_height, target_width, 3)
        print("[LOG] Loading new model...")
        base_model = get_base_model(
            input_shape=input_shape,
            vocab_size=CHAR_TO_NUM.vocabulary_size(),
            grayscale=opt['grayscale'],             # TOO phai bo
            invert_color=opt['invert_color'],       # TODO xem xet bo
            input_normalized=False,     # TODO phai bo, we won't normalize the image before feeding to the model
        )

    model = get_CTC_model(base_model)
    print(model.summary())

    print(f"[LOG] Image will {'' if opt['grayscale'] else 'NOT '}be converted to grayscale")

    # given input shape (N, 118, 2202, 3), the output the
    # base_model will be (N, 244, vocab_size + 1).
    # so time_steps will be dim 1
    time_steps = base_model.output.shape[1]

    train_dt = get_tf_dataset(
        img_dir=opt['train_data'],
        label_path=os.path.join(opt['train_data'], 'labels.json'),
        target_size=(target_height, target_width),
        # grayscale=opt['grayscale'],
        # invert_color=opt['invert_color'],
        time_steps=time_steps,
        batch_size=opt['batch_size'],
        shuffle=opt['shuffle'],
        cache=opt['cache']
    )

    val_dt = get_tf_dataset(
        img_dir=opt['val_data'],
        label_path=os.path.join(opt['val_data'], 'labels.json'),
        target_size=(target_height, target_width),
        # grayscale=opt['grayscale'],
        # invert_color=opt['invert_color'],
        time_steps=time_steps,
        batch_size=opt['batch_size'],
        shuffle=False,
        cache=opt['cache']
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=opt['lr']),
    )

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=opt['reduce_lr_patience'], verbose=1),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=opt['early_stop_patience'], verbose=1, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(filepath='saved_models/htr', save_best_only=True),
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

    print('[LOG] Config are set:')
    for k, v in opt.items():
        print(f'{k:24}: {v}')

    run(opt)



