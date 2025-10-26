import os
import yaml
import random
import pickle
import numpy as np
from functools import partial

# Mapping types to (deap attribute, sampling function)
converter = {
    float: ('attr_float', random.uniform),
    int: ('attr_int', random.randint),
    bool: ('attr_bool', random.randint)
}

def mergeconfig(base, custom):
    '''
    Merges nested configurations
    '''
    merged = base.copy()
    for key, value in custom.items():
        if isinstance(value, dict) and key in merged:
            merged[key] = mergeconfig(merged[key], value)
        else:
            merged[key] = value
    return merged

def load_pickle(basepath: str, filename: str):
    '''Load a pickle file from basepath/filename'''
    with open(os.path.join(basepath, filename), 'rb') as f:
        return pickle.load(f)

def load_yaml(config_path):

    # Statistical functions mapping
    STAT_FUNCTIONS = {
        "min": np.min,
        "max": np.max,
        "mean": np.mean,
        "std": np.std,
        "med": np.median,
        "q1": partial(np.percentile, q=25),
        "q3": partial(np.percentile, q=75)
    }

    with open(config_path, 'r') as f:
        params = yaml.safe_load(f)
    
    if 'base' in params and 'evolparams' in params['base']:
        # Convert string types to actual types (e.g., 'int' -> int)
        import builtins
        init_params = params['base']['initparams']
        for key in ('ind_type', 'pop_type'):
            if key in init_params and isinstance(init_params[key], str):
                try:
                    init_params[key] = getattr(builtins, init_params[key])
                except AttributeError:
                    raise ValueError(f"Invalid type for '{key}': {init_params[key]}")
        # and map statistic names to functions
        statnames = params['base']['evolparams'].get('statparams', [])
        params['base']['evolparams']['statparams'] = {
            name: STAT_FUNCTIONS[name] 
            for name in statnames
            if name in STAT_FUNCTIONS
        }

    return params

def load_internal(probname, configtype='baseconfig'):
    '''Load an internal configuration file based on problem name and type'''
    folder = probname.replace('.', os.sep)
    path = os.path.join(
        f'deapsleep/experiments/configurations',
        folder, 
        f'{configtype}.yaml'
    )

    if not os.path.exists(path):
        raise FileNotFoundError(f"Can't find a configuration for: {probname}")

    return load_yaml(path)

def load_logbooks(probname, version, hof='hof.pkl', log_list='log_list.pkl', log_agg='log_median.pkl'):
    '''Load stored logbooks from internal "results" folder based on problem name (after optimization)'''
    folder = os.path.join(probname.replace('.', os.sep), version)
    hof_path = os.path.join(
        f'deapsleep/experiments/results',
        folder, 
        hof
    )
    log_list_path = os.path.join(
        f'deapsleep/experiments/results',
        folder,
        log_list
    )
    log_agg_path = os.path.join(
        f'deapsleep/experiments/results',
        folder,
        log_agg
    )
    if not os.path.exists(hof_path):
        raise FileNotFoundError(f"Can't find a Hall of Fame for: {probname}")
    if not os.path.exists(log_list_path):
        raise FileNotFoundError(f"Can't find a list of logbooks for: {probname}")
    if not os.path.exists(log_agg_path):
        raise FileNotFoundError(f"Can't find an aggregated logbook for: {probname}")

    return hof_path, log_list_path, log_agg_path

def parse_extra_args(extra_args):
    '''
    Parse extra command-line arguments in the form:
    --key value
    --key=value
    Returns a dictionary of overrides.
    '''
    overrides = {}
    key = None
    for arg in extra_args:
        if arg.startswith('--'):
            key_part = arg[2:]
            if '=' in key_part:
                k, v = key_part.split('=', 1)
                overrides[k] = v
            else:
                key = key_part
        else:
            if key is not None:
                overrides[key] = arg
                key = None
            else:
                raise ValueError(f"Unexpected argument: {arg}")
    if key is not None:
        raise ValueError(f"Missing value for argument --{key}")
    return overrides

def apply_overrides(params, overrides):
    '''
    Apply overrides to the params dictionary.
    Supports nested keys using dot notation.
    '''
    special_keys = {
        'indD_rate': ('base', 'evolparams'),
        'popD_rate': ('base', 'evolparams'),
        'indD_strg': ('base', 'evolparams')
    }

    for key_str, value_str in overrides.items():
        value = yaml.safe_load(value_str)

        if key_str in special_keys:
            section = params.setdefault(special_keys[key_str][0], {}).setdefault(special_keys[key_str][1], {})
            section[key_str] = value
            continue

        found = False

        if key_str in params:
            params[key_str] = value
            found = True
        elif 'base' in params:
            for sub in ('initparams', 'evolparams'):
                if sub in params['base'] and key_str in params['base'][sub]:
                    params['base'][sub][key_str] = value
                    found = True
                    break

        if not found:
            raise KeyError(f"'{key_str}' doesn't exist.")

def format_version(version):
    '''
    Format version string to a more readable form.
    '''
    version_lower = version.lower() 

    if 'base' in version_lower:
        return 'Baseline'
    pop = any(x in version_lower for x in ['pdrop', 'popd'])
    ind = any(x in version_lower for x in ['idrop', 'indd'])

    if pop and not ind:
        return 'Population Dropout'
    elif ind and not pop:
        return 'Individual Dropout'
    elif ind and pop:
        return 'Individual & Population Dropout'
    else:
        return version

def extract_params(params, required_keys, optional_keys):
    '''
    Extract required and optional parameters from a dictionary.
    Raises KeyError if any required key is missing.
    '''
    # Default empty lists/dicts if None
    if required_keys is None:
        required_keys = []
    if optional_keys is None:
        optional_keys = {}

    missing = [k for k in required_keys if k not in params]
    if missing:
        raise KeyError(f'Missing required key: {missing}')

    values = [params[k] for k in required_keys]
    values += [params.get(k, v) for k, v in optional_keys.items()]

    return tuple(values)

def get_distrib(loglist, stat, target):
    '''
    Extract a list of fitness values at 'target' index from each logbook,
    using the specified statistic (e.g., 'mean', 'min', etc.).
    '''
    return [logbook.select(stat)[target].item() for logbook in loglist]