from model.dataset import Dataset
from model.catboost import CatboostModel
from model.lightgbm import LightGBMModel
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
    dataset = Dataset(model=args.model,
                      force_reload=args.force_reload,
                      strategy=args.strategy)

    if args.explore:
        params = find_params(dataset, args.model)
    else:
        params = load_params(args.model)
        params['iterations'] = 50000

    if args.fake_run:
        params['iterations'] = 1

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

    args = parser.parse_args()
    main(args)
