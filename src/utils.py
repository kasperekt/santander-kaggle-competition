import os
import datetime

from config import OUT_DIR


def current_date_str():
    now = datetime.datetime.now()
    return '{}-{}_{}-{}-{}'.format(now.day, now.month, now.hour, now.minute, now.second)


def get_params_filepath(explorer_type):
    return os.path.join(OUT_DIR, 'params.{}.pickle'.format(explorer_type))