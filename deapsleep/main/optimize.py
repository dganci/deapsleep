import os
import pickle
import argparse
from operator import itemgetter

from deapsleep.src.tester import test
from deapsleep.experiments.utils import load_yaml, load_internal, parse_extra_args, apply_overrides

def run_experiment(params):

    dirname, problem, n_runs, n_var, save_results, aggr_op, version, seed = itemgetter(
        'dirname', 'problem', 'n_runs', 'n_var',
        'save_results', 'aggregation_op', 'version', 'seed'
    )(params)

    # eventually, use a tuned configuration
    if params.get('use_opt_config', False):
        cfg_path = os.path.join(dirname, problem, f"{problem}_best_config.pkl")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(
                "Optimized configuration not found. "
                "Either create it or set use_opt_config=False."
            )
        with open(cfg_path, 'rb') as f:
            config = pickle.load(f)
    else:
        config = params['base']

    test(
        dirname,
        problem,
        config,
        n_runs,
        n_var=n_var,
        save_results=save_results,
        aggr_op=aggr_op,
        version=version,
        seed=seed
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run experiments (single- or multi-run) based on a YAML configuration."
    )
    parser.add_argument(
        '--config', type=str, required=True,
        help="Path to the YAML configuration file."
    )
    parser.add_argument(
        '-i', '--internal', action='store_true',
        help="Load an internal configuration instead of a YAML file."
    )
    args, remaining = parser.parse_known_args()

    if args.internal:
        params = load_internal(args.config)
    else:
        params = load_yaml(args.config)

    overrides = parse_extra_args(remaining)
    if overrides:
        apply_overrides(params, overrides)
        
    run_experiment(params)