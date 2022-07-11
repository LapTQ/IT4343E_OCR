import tensorflow as tf
from tensorflow import keras
import numpy as np
from .datasets import *


def query(msg):
    return not bool(input('[QUERY] ' + msg + ' [<ENTER> for yes] ').strip())


def info(msg):
    print('[INFO] ' + msg)


def warn(msg):
    print('[WARNING] ' + msg)


def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    # Iterate over the results and get back the text
    output_text = []
    for result in results:
        result = tf.strings.reduce_join(NUM_TO_CHAR(result)).numpy().decode("utf-8")
        output_text.append(result)
    return output_text

