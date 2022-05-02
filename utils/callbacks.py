import tensorflow as tf
from tensorflow import keras
from jiwer import wer
from .generals import decode_batch_predictions
from .datasets import *
import numpy as np

class CallbackEval(keras.callbacks.Callback):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def on_epoch_end(self, epoch, logs=None):
        predictions = []
        targets = []
        for images, labels in self.dataset:
            preds = self.model.predict(images)
            preds = decode_batch_predictions(preds)
            predictions.extend(preds)
            for label in labels:
                label = tf.strings.reduce_join(NUM_TO_CHAR(label)).numpy().decode('utf-8')
                targets.append(label)
        wer_score = wer(targets, predictions)
        print(f'WER: {wer_score:.4f}')
        for i in np.random.randint(0, len(predictions), 100):
            print(f'True: {targets[i]}')
            print(f'Pred: {predictions[i]}')
