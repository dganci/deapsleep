import os
import pickle
import argparse
import deapsleep.main as d_

def run_experiment(params):
    '''
    Run experiments based on the provided configuration parameters.
    '''

    dirname, problem, n_runs, n_var, save_results, aggr_op, version, seed = d_.extract_params(
        params, 
        ['dirname', 'problem'], # required keys
        {
            'n_runs': 30,
            'n_var': None, 
            'save_results': False, 
            'aggregation_op': 'median', 
            'version': '1.0', 
            'seed': 42
        } # optional keys with defaults
    )

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

    d_.test(
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

def run():
    parser = argparse.ArgumentParser(
        description="Run experiments (single- or multi-run) based on a YAML configuration."
    )
    # Configuration file argument (will load an internal configuration if --internal or --i is set)
    # Example: --config deapsleep.ackley --i
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
        params = d_.load_internal(args.config)
    else:
        params = d_.load_yaml(args.config)

    # Apply any command-line overrides
    overrides = d_.parse_extra_args(remaining)
    if overrides:
        d_.apply_overrides(params, overrides)
    
    run_experiment(params)

if __name__ == '__main__':
    run()
    