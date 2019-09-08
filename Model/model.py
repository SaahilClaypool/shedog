#!python3 

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd

import time
from datetime import timedelta

import math
import os
from os import makedirs
from os.path import join, exists, expanduser


import scipy.misc
from scipy.stats import itemfreq
from random import sample
import pickle

import PIL.Image

from IPython.display import display

from zipfile import ZipFile

from io import BytesIO


def setup_dirs():
    print("Setup directories")
    cache_dir = expanduser(join('~/.keras'))
    if not exists(cache_dir):
        makedirs(cache_dir)

    models_dir = join(cache_dir, 'models')
    if not exists(models_dir):
        makedirs(models_dir)

data_dir = '/home/sclaypool/gitProjects/DogApp/Data/'

SEED = 1987

def load_labs():
    return pd.read_csv(join(data_dir, 'labels.csv'))

if __name__ == "__main__":
    print("hello world")
