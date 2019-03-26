import os
import pandas as pd
import numpy as np

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split
from config import n_jobs, DATA_DIR


def to_data_format(df):
    data = np.array(df.iloc[:, 2:], dtype=np.float32)
    labels = np.array(df.iloc[:, 1], dtype=np.int32)

    return data, labels


def get_data(force_reload=False, strategy='oversampling', test_size=0.15):
    train_data_file = os.path.join(DATA_DIR, 'train_data.{}.npy'.format(strategy))
    train_labels_file = os.path.join(DATA_DIR, 'train_labels.{}.npy'.format(strategy))
    val_data_file = os.path.join(DATA_DIR, 'val_data.{}.npy'.format(strategy))
    val_labels_file = os.path.join(DATA_DIR, 'val_labels.{}.npy'.format(strategy))

    training_files_exist = os.path.exists(train_data_file) and os.path.exists(train_labels_file)
    val_files_exist = os.path.exists(val_data_file) and os.path.exists(val_labels_file)

    if not force_reload and training_files_exist and val_files_exist:
        X_train = np.load(train_data_file)
        y_train = np.load(train_labels_file)

        X_val = np.load(val_data_file)
        y_val = np.load(val_labels_file)
    else:
        train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
        X, y = to_data_format(train_df)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size)

        print('Shapes before: {}, {}'.format(X_train.shape, y_train.shape))

        if strategy == 'oversampling':
            X_train, y_train = SMOTE(n_jobs=n_jobs).fit_resample(X_train, y_train)
        elif strategy == 'combine':
            smote = SMOTE(n_jobs=n_jobs)
            enn = EditedNearestNeighbours(n_jobs=n_jobs)
            X_train, y_train = SMOTEENN(smote=smote, enn=enn).fit_resample(X_train, y_train)
        elif strategy == 'undersampling':
            enn = EditedNearestNeighbours(n_jobs=n_jobs)
            X_train, y_train = enn.fit_resample(X_train, y_train)

        print('Shapes after: {}, {}'.format(X_train.shape, y_train.shape))

        np.save(train_data_file, X_train)
        np.save(train_labels_file, y_train)
        np.save(val_data_file, X_val)
        np.save(val_labels_file, y_val)

    return X_train, X_val, y_train, y_val
