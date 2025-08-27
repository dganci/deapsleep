import random
import numpy as np
from tqdm.auto import tqdm
from warnings import warn
from pymoo.problems import get_problem

from deapsleep.src.dropout import *
from deapsleep.src.utils.storer import Storer
from deapsleep.src.core.initializer import Initializer
from deapsleep.src.core.evolver import Evolver
from deapsleep.experiments.utils import converter

def test(
    dirname: str,
    probname: str,
    config: dict,
    n_runs: int,
    n_var: int | None = None,
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
    dirname     : base directory for results
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
        random.seed(s)
        np.random.seed(s)

    storer = Storer(dirname, probname, version)
    initparams = config['initparams']
    evolparams = config['evolparams']

    if _pymoo:
        # Instantiate pymoo problem (with optional n_var override)
        try:
            p = get_problem(probname, n_var=n_var) if n_var is not None else get_problem(probname)
        except TypeError:
            p = get_problem(probname)
            warn(f"\n{probname} does not support 'n_var'; using default n_var={p.n_var}.\n")

        # Save ideal point and Pareto front (for multi-objective)
        storer.add_targets(p.ideal_point())
        if p.n_obj > 1:
            storer.add_true_pareto(p.pareto_front())

        # Attach problem instance to evolparams
        config['evolparams']['problem'] = p

        # Build initializer attributes from problem bounds
        attr, func = converter[p.vtype]
        initattr = [[attr, func, lb, ub] for lb, ub in zip(p.xl, p.xu)]

    else:
        initattr = config['initattr']

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
        storer.save(n_obj=p.n_obj, op=aggr_op)