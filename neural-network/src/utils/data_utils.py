from builtins import range
from six.moves import cPickle as pickle
import numpy as np
import os
import platform

def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000,
                     subtract_mean=True):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for classifiers. These are the same steps as we used for the SVM, but
    condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'data/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    if subtract_mean:
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image

    # Transpose so that channels come first
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    # Package data into a dictionary
    return {
      'X_train': X_train, 'y_train': y_train,
      'X_val': X_val, 'y_val': y_val,
      'X_test': X_test, 'y_test': y_test,
    }


def load_FER_batch(filename):
    pass

def load_FER2013_data(ROOT):
    """ load all of fer2013 """

    training_images = []
    training_labels = []
    testing_images  = []
    testing_labels  = []

    with open(os.path.join(ROOT, "labels_public.txt")) as labels_file:

        # Skip the first line
        next(labels_file) 

        for line in labels_file:
            line = line.rstrip('\n')
            tokenised = line.split(',')
            file = tokenised[0]
            label = int(tokenised[1])
            im = cv2.imread(os.path.join(ROOT, file))

            if file.split('/')[0] == 'Train':
                training_images.append(im)
                training_labels.append(label)
            else:
                testing_images.append(im)
                testing_labels.append(label)
    return training_images, training_labels, testing_images, testing_labels

def get_FER2013_data(ROOT):

    trX, trY, teX, teY = load_fer2013_data(ROOT)
    vaX = []
    vaY = []

    # Use as many data items in test set for validation
    for _ in range(len(teX)):
        vaX.append(trX.pop())
        vaY.append(trY.pop())

    # Normalize the data: subtract the mean image
    mean_image = np.mean(trX, axis=0)
    trX -= mean_image
    vaX -= mean_image
    teX -= mean_image

    # Package data into a dictionary
    return {
        'X_train': np.array(trX).transpose(0,3,1,2), 'y_train': np.array(trY),
        'X_val': np.array(vaX).transpose(0,3,1,2), 'y_val': np.array(vaY),
        'X_test': np.array(teX).transpose(0,3,1,2), 'y_test': np.array(teY),
    }

