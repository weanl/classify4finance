
import argparse
import numpy as np


def get_args():

    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default=None,
        help='The Configuration file'
    )
    args = argparser.parse_args()

    return args


#
def split_index(instance_num, train_rate=0.8):
    train_num = int(instance_num*train_rate)
    Index = np.arange(instance_num)
    # np.random.seed(111)  #  设置随机种子
    np.random.shuffle(Index)
    train_index = Index[:train_num]
    test_index = Index[train_num:]
    return train_index, test_index


#
def split_index2(instance_num, train_rate=0.8):
    train_num = int(instance_num*train_rate)
    Index = []
    a, b = 0, instance_num-1
    while a<b:
        Index.append(a)
        Index.append(b)
        a += 1
        b -= 1
        pass
    if a==b:
        Index.append(a)
    else:
        pass
    Index = np.asarray(Index)
    print(Index.shape)
    train_index = Index[:train_num]
    test_index = Index[train_num:]
    return train_index, test_index


#
def normScalarZ(x, x_mean, x_std):
    x_norm = (x - x_mean) / x_std
    return x_norm


def resScalarZ(x_norm, x_mean, x_std):
    x = x_norm*x_std + x_mean
    return x


def accuracy_per_class(y_truth, y_pred, labels=[1,2,3,4,5]):

    idxs = {}
    for idx, c in enumerate(labels):
        idxs[c] = np.where(
            y_truth==c
        )[0]
    accuracy = []
    for idx, c in enumerate(labels):
        if idxs[c].shape[0]==0:
            acc = 0
        else:
            acc = np.where(y_pred[idxs[c]]==c)[0].shape[0] / idxs[c].shape[0]
        accuracy.append(
            acc
        )

    return accuracy


if __name__ == "__main__":

    A,B = split_index2(100)
    print(A)
