from data import get_data
from model.catboost import get_model, CatboostExplorer
from params import find_params, load_params
from argparse import ArgumentParser
from kaggle import submit

def main(args):
    X_train, X_val, y_train, y_val = get_data(force_reload=args.force_reload, strategy=args.strategy)

    if args.explore:
        params = find_params(X_train, y_train, args.explorer_type, eval_set=(X_val, y_val))
    else:
        params = load_params(args.explorer_type)
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
    parser.add_argument('--explorer-type', type=str, default='catboost')
    parser.add_argument('--force-reload', action='store_true')
    parser.add_argument('--strategy', type=str, default='oversampling')
    parser.add_argument('--submit', action='store_true')

    args = parser.parse_args()
    main(args)
