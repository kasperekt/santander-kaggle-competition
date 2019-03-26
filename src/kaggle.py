import os

COMPETITION = 'santander-customer-transaction-prediction'


def kaggle_submit(filepath, message):
    if not os.path.exists(filepath):
        raise FileNotFoundError(filepath + ' doesn\'t exist.')

    os.system('kaggle c submit -f {} -m {} {}'.format(filepath, message, COMPETITION))

