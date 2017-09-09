
import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


N = 1000 # sample size


def diriter(root_dir):
    """ grab files from root/sub dirs having extension
    """
    return ((f, os.path.join(root, f))
            for root, dirs, files in os.walk(root_dir)
            for f in files)


def iscat(file_name):
    """ label each observation as cat or not
    """
    return 1 if 'cat' in file_name.lower() else 0


def make_key(dir_path):
    """ make dataframe of filename, filepath
    """
    return (pd.DataFrame.from_records([tup for tup in diriter(dir_path)],
                                    columns = ['file_name', 'file_path'])
                        .assign(label=lambda df: df.file_name.apply(iscat)))


def image_to_norm_vec(file_path):
    """ load image from file_path, down sample to 64x64
        flatten, and normalize
    """
    img = Image.open(file_path)
    img = img.resize((64, 64))
    return np.array(img).flatten()/255


def reshape(x):
    """ transpose and make ndarray
    """
    return x.reshape(x.shape[0], -1).T


def load_data(dir_path):
    """ load and preprocess all images in dirpath
    """
    np.random.seed(1)
    key = make_key(dir_path)
    key = (key.sample(N)
              .assign(image_vec=(
                lambda df: df.file_path.apply(image_to_norm_vec)))
          )
    X = np.vstack(key.image_vec)
    y = key.label.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return reshape(X_train), reshape(X_test), reshape(y_train), reshape(y_test)

