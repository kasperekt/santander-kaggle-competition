import os
import numpy as np
import pandas as pd

from data import get_data
from model.catboost import get_model, CatboostExplorer
from kaggle import kaggle_submit
from params import find_params, load_params
from argparse import ArgumentParser
from config import OUT_DIR, DATA_DIR
from utils import current_date_str


def submit(model):
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    X_test = np.array(test_df.iloc[:, 1:], dtype=np.float32)

    predictions = np.array(model.predict_proba(X_test))
    predictions = predictions[:, 1]
    submission = pd.DataFrame({'Id_code': test_df['ID_code'], 'target': predictions})

    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)

    current_date = current_date_str()
    submission_path = os.path.join(OUT_DIR, 'submission_{}.csv'.format(current_date))

    submission.to_csv(submission_path, index=False)
    kaggle_submit(submission_path, 'submission_' + current_date)


def main(args):
    # For now, there is only catboost explorer
    Explorer = CatboostExplorer

    X_train, X_val, y_train, y_val = get_data(force_reload=args.force_reload, strategy=args.strategy)

    if args.explore:
        params = find_params(X_train, y_train, Explorer, eval_set=(X_val, y_val))
    else:
        params = load_params(Explorer)
        params['iterations'] = 50000

    if args.fake_run:
        params['iterations'] = 1

    print('Using params: ', params)

    # For now, it's the "catboost" model
    model = get_model(**params)

    # train the model
    model.fit(X_train, y_train, eval_set=(X_val, y_val))

    # Submission
    if args.submit:
        submit(model)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--fake-run', action='store_true')
    parser.add_argument('--explore', action='store_true')
    parser.add_argument('--force-reload', action='store_true')
    parser.add_argument('--strategy', type=str, default='oversampling')
    parser.add_argument('--submit', action='store_true')

    args = parser.parse_args()
    main(args)
