import tensorflow as tf
from tensorflow import keras
import numpy as np
from .data_utils import *


def query(msg):
    return not bool(input('[QUERY] ' + msg + ' [<ENTER> for yes] ').strip())


def info(msg):
    print('[INFO] ' + msg)


def warn(msg):
    print('[WARNING] ' + msg)

