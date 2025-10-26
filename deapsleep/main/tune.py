import os
import yaml
import argparse
import optuna                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
import numpy as np
import deapsleep.main as d_
from operator import itemgetter

def get_trial_config(trial, trial_spec):
    '''
    Build a config (initparams, evolparams) for this Optuna trial:
    - Sample values for keys with {'type':..., 'low':..., 'high':...}
    - Copy fixed values unchanged.
    '''
    result = {'initparams': {}, 'evolparams': {}}

    for section in ('initparams', 'evolparams'):
        for key, val in trial_spec.get(section, {}).items():
            if isinstance(val, dict) and 'type' in val:
                t, low, high = val['type'], val['low'], val['high']
                if t == 'int':
                    result[section][key] = trial.suggest_int(key, low, high)
                elif t == 'float':
                    result[section][key] = trial.suggest_float(key, low, high)
                else:
                    raise ValueError(f"Unsupported type '{t}' for '{key}'")
            else:
                result[section][key] = val

    return result

def make_objective(base_conf, trial_spec):
    '''
    Return the Optuna objective function that:
    - Merges base_conf with parameters from trial_spec
    - Calls test(...) and returns mean of minima over n_runs
    '''
    def objective(trial):
        n_runs, problem, dirname, n_var, seed = itemgetter(
            'n_runs', 'problem', 'dirname', 'n_var', 'seed'
        )(base_conf)

        trial_conf = get_trial_config(trial, trial_spec)
        config = d_.mergeconfig(base_conf, trial_conf)
        config.setdefault('do_opt', True)

        minima = d_.test(
            n_runs=n_runs,
            probname=problem,
            config=config,
            dirname=dirname,
            n_var=n_var,
            seed=seed
        )
        return float(np.mean(minima))

    return objective

def find_best_params(base_conf, trial_spec, n_trials):
    '''
    Run Optuna optimization and return best parameters.
    '''
    problem = base_conf['problem']
    storage_url = f"sqlite:///optuna_{problem}.db"

    study = optuna.create_study(
        direction='minimize',
        study_name=problem,
        storage=storage_url,
        load_if_exists=True
    )
    objective = make_objective(base_conf, trial_spec)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params

def main(base_yaml, trial_yaml, n_trials=50):
    base_conf = d_.load_yaml(base_yaml)
    trial_spec = d_.load_yaml(trial_yaml)

    dirname, problem = itemgetter('dirname', 'problem')(base_conf)

    best_params = find_best_params(base_conf, trial_spec, n_trials)
    final_conf = d_.mergeconfig(base_conf, best_params)

    out_path = os.path.join(dirname, problem, f"{problem}_best_config.yaml")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        yaml.dump(final_conf, f)

    print(f"Best configuration saved to: {out_path}")

def run():
    parser = argparse.ArgumentParser(
        description="Optimize hyperparameters via Optuna using base + trial YAML configs."
    )
    parser.add_argument(
        '--base', required=True,
        help="Path to the base YAML configuration."
    )
    parser.add_argument(
        '--trial', required=True,
        help="Path to the trial-spec YAML describing search ranges."
    )
    parser.add_argument(
        '--n_trials', type=int, default=50,
        help="Number of Optuna trials to run (default: 50)."
    )
    args = parser.parse_args()

    main(args.base, args.trial, args.n_trials)

if __name__ == '__main__':
    run()