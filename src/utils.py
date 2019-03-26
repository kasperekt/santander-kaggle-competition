import datetime


def current_date_str():
    now = datetime.datetime.now()
    return '{}-{}_{}-{}-{}'.format(now.day, now.month, now.hour, now.minute, now.second)