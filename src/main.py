import warnings

from model.catboost import CatboostModel
from model.lightgbm import LightGBMModel
from data import Dataset
from params import find_params, load_params
from argparse import ArgumentParser
from kaggle import submit


def pick_model(args, **params):
    if args.model == 'catboost':
        return CatboostModel(**params)
    elif args.model == 'lightgbm':
        return LightGBMModel(**params)
    else:
        raise ValueError('{} should be either "catboost" or "lightgbm"'.format(args.model))


def main(args):
    dataset = Dataset(force_reload=args.force_reload,
                      strategy=args.strategy)

    if args.explore:
        params = find_params(dataset, args.model)
    else:
        # params = load_params(args.model)
        params = {'feature_fraction': 0.2,
                  'max_bin': 128,
                  'max_depth': 32,
                  'min_data_in_leaf': 500,
                  'num_leaves': 70,
                  'reg_alpha': 1.6,
                  'reg_lambda': 2.5}

    if args.fake_run:
        params['iterations'] = 1
    else:
        params['num_iterations'] = 100000
        params['iterations'] = 100000
        params['learning_rate'] = 0.001

    print('Using params: ', params)

    model = pick_model(args, **params)
    model.fit(dataset)

    # Submission
    if args.submit:
        submit(model)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--fake-run', action='store_true')
    parser.add_argument('--explore', action='store_true')
    parser.add_argument('--model', type=str, default='catboost')
    parser.add_argument('--force-reload', action='store_true')
    parser.add_argument('--strategy', type=str, default='oversampling')
    parser.add_argument('--submit', action='store_true')

    warnings.filterwarnings('ignore')

    args = parser.parse_args()
    main(args)
