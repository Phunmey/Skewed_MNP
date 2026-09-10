import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


def _scale_keep_intercept(X_train, X_test=None):
    X_train = np.asarray(X_train, dtype=float)
    X_train_s = X_train.copy()
    scaler = RobustScaler()
    X_train_s[:, 1:] = scaler.fit_transform(X_train[:, 1:])

    if X_test is None:
        return X_train_s, None

    X_test = np.asarray(X_test, dtype=float)
    X_test_s = X_test.copy()
    X_test_s[:, 1:] = scaler.transform(X_test[:, 1:])
    return X_train_s, X_test_s


def split_data(df_num, df_y, train_size=0.6, random_state=123):
    xtrain, xtest, ytrain, ytest = train_test_split(
        df_num, df_y, train_size=train_size, random_state=random_state, stratify=df_y)

    #xtrain_scaled, xtest_scaled = _scale_keep_intercept(xtrain, xtest)

    minority_class_samples = pd.Series(ytrain).value_counts().min()
    k_neighbors = min(5, minority_class_samples - 1)

    print(f"xtrain_shape:{xtrain.shape}, ytrain_shape: {np.asarray(ytrain).shape}\n")
    print(f"Count of classes in the data: {np.unique(df_y, return_counts=True)}\n")
    print(f"Count of classes in ytrain: {np.unique(ytrain, return_counts=True)}\n")
    print(f"Count of classes in ytest: {np.unique(ytest, return_counts=True)}\n")

    return xtrain, xtest, np.asarray(ytrain), np.asarray(ytest), k_neighbors


def scale_full_data(df_num):
    X_scaled, _ = _scale_keep_intercept(df_num, None)
    return X_scaled
