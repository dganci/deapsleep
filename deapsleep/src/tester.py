import random
import importlib
import numpy as np
from tqdm.auto import tqdm
from warnings import warn
from pymoo.problems import get_problem
from deapsleep.src.utils import Storer
from deapsleep.src.core import Initializer, Evolver
from utils.utils import converter
from utils.paths import RESULTS_DIR

def test(
    probname: str,
    config: dict,
    n_runs: int,
    n_var: int | None = None,
    additional: dict = {},
    save_results: bool = False,
    aggr_op: str = 'mean',
    version: str = 'std',
    _pymoo: bool = True,
    seed: int | None = None
):
    '''
    Run n_runs of an evolutionary experiment (single- or multi-objective),
    store logs, stats, and optionally save aggregated results.

    Parameters:
    -----------
    probname    : problem name (pymoo)
    config      : dict with 'initparams' and 'evolparams'
    n_runs      : number of independent runs
    n_var       : override for number of variables (None to use default)
    save_results: whether to save aggregated results at the end
    aggr_op     : aggregation operator for logs, e.g. 'mean'
    version     : label (e.g. 'baseline' or 'proposed')
    _pymoo      : if True, instantiate a pymoo problem; otherwise use custom setup
    seed        : random seed base (each run uses seed+i)
    '''
    def _set_seed(s: int):
        '''
        Set random seed for reproducibility.
        '''
        random.seed(s)
        np.random.seed(s)

    storer = Storer(probname, version)
    initparams = config['initparams']
    evolparams = config['evolparams']

    if _pymoo:

        if "generator" in additional: # then use a pymoo custom problem generator
            import inspect
            mdl, func = additional['generator'].rsplit('.', 1)
            generator = getattr(importlib.import_module(mdl), func)
            valids = inspect.signature(generator).parameters.keys()
            p = generator(
                n_var, # assume n_var is always the first argument
                **{
                    k: v 
                    for k, v in additional.items() 
                    if k in valids and k != 'generator'
                    }
                )
        else:
            try:
                p = get_problem(probname.lower(), n_var=n_var, **additional)
            except TypeError: # n_var override is not supported
                p = get_problem(probname.lower(), **additional)
                warn(f"\n{probname} does not support 'n_var'; using default n_var={p.n_var}.\n")

        # Eventually, ovveride additional parameters
        for param, value in additional.items():
            if param != "generator" and hasattr(p, param) and value is not None:
                if isinstance(value, (list, tuple)):
                    # lists will be converted to np.arrays
                    setattr(p, param, np.array(value))
                else:
                    setattr(p, param, value)

        # Build initializer attributes from problem bounds
        try:
            # Eventually convert bounds to integer if needed
            if p.vtype in (int, bool) \
                and not (
                    p.xl.dtype.kind in ("i", "b") 
                    and p.xu.dtype.kind in ("i", "b")):
                p.xl = np.array(p.xl, dtype=int)
                p.xu = np.array(p.xu, dtype=int)
            attr, func = converter[p.vtype]
        except KeyError: # then probably using a bbob problem
            attr, func = 'attr_float', random.uniform # assume a real-valued problem

        # Attach problem instance to params
        config['evolparams']['problem'] = p
        initattr = [[attr, func, lb, ub] for lb, ub in zip(p.xl, p.xu)]

    else:
        initattr = config['initattr']

    # Attach n_var into init parameters for problems that require it
    # (e.g. permutation problems)
    if probname in {
            'tsp'
            # ...
        }:
        initparams['n_var'] = n_var

    pbar = tqdm(total=n_runs, desc=f"Optimizing {probname}", unit="run")
    for i in range(n_runs):
        if seed is not None:
            _set_seed(seed + i)

        initializer = Initializer(*initattr, **initparams)
        evol = Evolver(initializer.toolbox, **evolparams)

        if p.n_obj > 1:
            logbook, archive = evol.moo()
            storer.add_pareto(archive)
        else:
            logbook, archive = evol.soo()

        # Store logs, stats, and evolution history
        storer.add_log(logbook)
        storer.add_evolution(logbook, archive)
        storer.add_stats(logbook, evolparams['statparams'].keys())

        pbar.update(1)
    pbar.close()

    if config.get('do_opt'):
        return storer.lastats['min']

    if save_results:
        # save results
        storer.save(op=aggr_op)
        # save problem internal state
        import os
        savepath = os.path.join(RESULTS_DIR, probname, version)
        from pprint import pformat
        with open(os.path.join(savepath, 'problem_dict.txt'), 'w') as f:
            f.write(pformat(p.__dict__, sort_dicts=False))
        # save problem instance
        import pickle
        with open(os.path.join(savepath, 'problem_instance.pkl'), 'wb') as f:
            pickle.dump(p, f)