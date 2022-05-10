import argparse
from models.crnn import *
from utils.generals import *
from utils.datasets import *
from utils.callbacks import *
import tensorflow as tf
from tensorflow import keras
import os
import matplotlib.pyplot as plt

def run(
        pretrained,
        epochs,
        batch_size,
        train_data,
        val_data,
        lr,
        reduce_lr_patience,
        early_stop_patience,
        # target_height,
        # target_width,
        grayscale,
        invert_color,
        dilate,
        shuffle,
        cache,
):


    if pretrained is not None:
        print(f'[LOG] Loading pretrained model at {pretrained} ...')
        base_model = keras.models.load_model(pretrained)

        target_height, target_width, depth = base_model.input.shape[1:]
        grayscale = True if 'rgb2_gray' in [layer.name for layer in base_model.layers] else False
        print(f'[LOG] Image will be resized to configuration of pretrained model: {(target_height, target_width, depth)}')
    else:
        target_height, target_width = 118, 2202
        input_shape = (target_height, target_width, 3)
        print("[LOG] Loading new model...")
        base_model = get_base_model(
            input_shape=input_shape,
            vocab_size=CHAR_TO_NUM.vocabulary_size(),
            grayscale=grayscale,
            invert_color=invert_color,
            input_normalized=False,     # we won't normalize the image before feeding to the model
        )

    model = get_CTC_model(base_model)
    print(model.summary())

    print(f'[LOG] Image will {"" if grayscale else "NOT "}be converted to grayscale')

    # given input shape (N, 118, 2202, 3), the output the
    # base_model will be (N, 244, vocab_size + 1).
    # so we take the dim 1
    time_steps = base_model.output.shape[1]

    train_dt = get_tf_dataset(
        img_dir=train_data,
        label_path=os.path.join(train_data, 'labels.json'),
        target_size=(target_height, target_width),
        # grayscale=grayscale,
        # invert_color=invert_color,
        time_steps=time_steps,
        batch_size=batch_size,
        shuffle=shuffle,
        cache=cache
    )

    val_dt = get_tf_dataset(
        img_dir=val_data,
        label_path=os.path.join(val_data, 'labels.json'),
        target_size=(target_height, target_width),
        # grayscale=grayscale,
        # invert_color=invert_color,
        time_steps=time_steps,
        batch_size=batch_size,
        shuffle=False,
        cache=cache
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
    )

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=reduce_lr_patience, verbose=1),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stop_patience, verbose=1, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(filepath='saved_models/htr', save_best_only=True),
        CallbackEval(val_dt)
    ]

    history = model.fit(
        train_dt,
        epochs=epochs,
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

    args = vars(ap.parse_args())

    print('[LOG] Config are set:')
    for k, v in args.items():
        print(f'{k:24}: {v}')

    run(**args)



