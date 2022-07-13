import tensorflow as tf
from tensorflow import keras
from jiwer import wer
from .data_utils import *
import numpy as np
from tqdm import tqdm


def decode_batch(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=False, beam_width=4)[0][0]
    # Iterate over the results and get back the text
    output_text = []
    for result in results:
        result = tf.strings.reduce_join(NUM_TO_CHAR(result)).numpy().decode("utf-8")
        output_text.append(result)
    return output_text


class CallbackEval(keras.callbacks.Callback):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def on_epoch_end(self, epoch, logs=None):
        predictions = []
        targets = []
        for batch in tqdm(self.dataset):
            y_pred = self.model.predict(batch)
            y_pred = decode_batch(y_pred)
            predictions.extend(y_pred)
            for label in batch['y_true']:
                label = tf.strings.reduce_join(NUM_TO_CHAR(label)).numpy().decode('utf-8')
                targets.append(label)
        wer_score = wer(targets, predictions)
        print(f'WER: {wer_score:.4f}')
        for i in np.random.randint(0, len(predictions), 24):
            print(f'True: {targets[i]}')
            print(f'Pred: {predictions[i]}')
