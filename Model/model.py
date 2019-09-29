#!python3 

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import json

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

DATA_DIR = '/home/sclaypool/gitProjects/DogApp/Data/'
SEED = 1987


def load_labs():
    return pd.read_csv(join(DATA_DIR, 'labels.csv'))

def setup_dirs():
    print("Setup directories")
    cache_dir = expanduser(join('~/.keras'))
    if not exists(cache_dir):
        makedirs(cache_dir)

    models_dir = join(cache_dir, 'models')
    if not exists(models_dir):
        makedirs(models_dir)

def count_per_breed(df):
    gr = df.groupby('breed')
    counts = {}
    for key in gr.groups.keys():
        counts[key] = len(gr.groups[key])

    f = lambda k: counts[k]
    # apply axis=1 would send the entire row to the function
    df['images'] = df['breed'].apply(f)
    return counts

def top_k(df, k=40):
    counts = count_per_breed(df)
    list_vals = [kv for kv in counts.items()]
    list_vals.sort(key=lambda v: v[1], reverse=True)
    return [i for i in map(lambda v: v[0], list_vals[:k])]

def filter_labels(labels, classes=40):
    tk = top_k(labels, k=classes)
    return pd.DataFrame(labels[labels['breed'].isin(tk)])

def char_columns():
    f = "../Data/traits.json"
    d = json.load(open(f, 'r'))

    cols = ['id']
    for k, v in d.items():
        if (len(v) == 0): # numeric
            cols.append(k)
        else: # categorical --> vector with key prefix
            for v1 in v:
                cols.append(f"{k}_{v1}")

    return cols, d

def open_trait_data():
    f = "../Data/breeds.json"
    return json.load(open(f, 'r'))

def breed_to_trait_vec(breed, data, col_options):
    """
    return dictionary with correct column labels
    """
    cols = {'breed': breed}
    for k, v in col_options.items():
        if (len(v) == 0): # numeric
            cols[k] = data[breed][k]
        else: # categorical --> vector with key prefix
            for v1 in v:
                if (v1 == data[breed][k]):
                    cols[f"{k}_{v1}"] = 1.0
                else:
                    cols[f"{k}_{v1}"] = 0.0

    return cols


def char_dataframe(labels):
    """
    Given the labels like "image_id": "breed",
    turn them into a dataframe like [id, c1, c2, c3]
    """
    cols, orig_cols = char_columns()
    data = open_trait_data()

    trait_dict = {}


    for breed in labels['breed'].unique():
        trait_dict[breed] = breed_to_trait_vec(breed, data, orig_cols)

    def f(row):
        d = trait_dict[row['breed']]
        d['id'] = row['id']
        return d

    df2 = pd.DataFrame(pd.DataFrame.from_records(list(labels.apply(f, axis = 1)))[cols])
    return df2


def split_train_characteristics(labels):
    """
    """
    rnd = np.random.random(len(labels))
    # Make training 80%, validation 80%
    # Could turn into K fold validation
    train_idx = rnd < 0.8
    valid_idx = rnd >= 0.8

    labels['target'] = 1

    id_to_breed_vec = labels.pivot('id', 'breed', 'target').reset_index().fillna(0)

    ytr_2 = id_to_breed_vec[train_idx]
    yv_2 = id_to_breed_vec[valid_idx]

    # make a matrix of id -> breed
    trait_df = char_dataframe(labels)
    ytr_1 = trait_df[train_idx]
    yv_1 = trait_df[valid_idx]

    return ytr_1, yv_1, ytr_2, yv_2

if __name__ == "__main__":
    print("hello world")

    labels = load_labs()
    count_per_breed(labels)

    # get top 10 breeds?
    l = labels
    l.sort_values(by='images', ascending=False)

    filt = filter_labels(l)
    filt
