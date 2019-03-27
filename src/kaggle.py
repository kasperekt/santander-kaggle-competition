import os
import pandas as pd
import numpy as np

from config import OUT_DIR, DATA_DIR
from utils import current_date_str

COMPETITION = 'santander-customer-transaction-prediction'


def kaggle_submit(filepath, message):
    if not os.path.exists(filepath):
        raise FileNotFoundError(filepath + ' doesn\'t exist.')

    os.system('kaggle c submit -f {} -m {} {}'.format(filepath, message, COMPETITION))


def submit(model):
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    X_test = np.array(test_df.iloc[:, 1:], dtype=np.float32)

    predictions = model.predict(X_test)
    submission = pd.DataFrame({'Id_code': test_df['ID_code'], 'target': predictions})

    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)

    current_date = current_date_str()
    submission_path = os.path.join(OUT_DIR, 'submission_{}.csv'.format(current_date))

    submission.to_csv(submission_path, index=False)
    kaggle_submit(submission_path, 'submission_' + current_date)