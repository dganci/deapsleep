import os
import yaml
import random
import pickle
import numpy as np

from functools import partial

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
    """Ritorna l’oggetto pickle contenuto in basepath/filename."""
    with open(os.path.join(basepath, filename), 'rb') as f:
        return pickle.load(f)

def load_yaml(config_path):

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

        import builtins
        init_params = params['base']['initparams']
        for key in ('ind_type', 'pop_type'):
            if key in init_params and isinstance(init_params[key], str):
                try:
                    init_params[key] = getattr(builtins, init_params[key])
                except AttributeError:
                    raise ValueError(f"Invalid type for '{key}': {init_params[key]}")

        statnames = params['base']['evolparams'].get('statparams', [])
        params['base']['evolparams']['statparams'] = {
            name: STAT_FUNCTIONS[name] 
            for name in statnames
            if name in STAT_FUNCTIONS
        }

    return params

def load_internal(probname, configtype='baseconfig'):
    folder = probname.replace('.', os.sep)
    path = os.path.join(
        f'deapsleep/experiments/configurations',
        folder, 
        f'{configtype}.yaml'
    )

    if not os.path.exists(path):
        raise FileNotFoundError(f"Can't find a configuration for: {probname}")

    return load_yaml(path)

def parse_extra_args(extra_args):
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

    if 'base' in version:
        return 'Baseline'
    elif 'popD' in version and 'indD' not in version:
        return 'Population Dropout'
    elif 'indD' in version and 'popD' not in version:
        return 'Individual Dropout'
    elif 'indD' in version and 'popD' in version:
        return 'Individual & Population Dropout'
    else:
        return version