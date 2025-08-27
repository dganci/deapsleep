import os
import pickle
import numpy as np
import pandas as pd

from collections import defaultdict
from collections.abc import Iterable

from deap.tools import Logbook

from deapsleep.src.utils.visualizer import Visualizer

class Storer(Visualizer):
    '''
    Extend Visualizer to store multiple runs:
      - loglist: raw Logbook objects per run
      - evolres: best individuals and eval counts per run
      - statres: time series of stats per run
      - lastats: final stat value per run
      - paretos: Pareto front per run (for multi-objective)
    Provides methods to save aggregated logs, best individuals, and plots.
    '''

    def __init__(self, dirname: str, problem: str, version: str):
        super().__init__(dirname, problem, version)
        self.path = os.path.join(dirname, problem, version)
        self.loglist = []
        self.paretos = []
        self.evolres = {'best': [], 'evals': []}
        self.statres = defaultdict(list)
        self.lastats = defaultdict(list)

    def add_log(self, log: Logbook):
        '''
        Append a run's Logbook.
        '''
        self.loglist.append(log)

    def add_evolution(self, log: Logbook, archive: Iterable):
        '''
        Record:
          - 'best': list of all individuals in the final archive
          - 'evals': total evaluations (sum of 'evals' column)
        '''
        self.evolres['best'].append(archive)
        self.evolres['evals'].append(int(sum(log.select('evals'))))

    def add_stats(self, log: Logbook, keys: Iterable[str]):
        '''
        For each stat key, save the entire series (statres)
        and the last value (lastats).
        '''
        for k in keys:
            series = log.select(k)
            self.statres[k].append(series)
            self.lastats[k].append(series[-1])

    def add_pareto(self, archive: Iterable):
        '''
        Store the final Pareto front as a list of fitness tuples.
        '''
        self.paretos.append([tuple(ind.fitness.values) for ind in archive])

    def add_targets(self, targets):
        '''
        Store ideal/target values for plotting.
        '''
        self.targets = targets

    def add_true_pareto(self, true_pareto):
        '''
        Store true Pareto front for comparison.
        '''
        self.true_pareto = true_pareto

    def _aggregate(self, op: str, fmt: str = '{:.2e} \u00B1 {:.2e}'):
        '''
        Aggregate logbooks across runs using 'op' ('mean' or 'median'):
          - num_agg: numerical Logbook of center values per generation
          - fmt_agg: DataFrame of formatted "center ± dispersion"
        '''
        n_gen = len(self.loglist[0])
        header = self.loglist[0].header

        num_log = Logbook()
        num_log.header = header  # type: ignore
        fmt_rows = []

        for i in range(n_gen):
            row_center = {}
            row_fmt = {}
            for key in header:
                vals = np.array([lb[i][key] for lb in self.loglist], dtype=object)
                # If entries are iterables (multi-objective), stack them
                arr = np.stack(vals) if isinstance(vals[0], Iterable) else vals[:, None]

                center = getattr(np, op)(arr, axis=0)
                disp = np.std(arr, axis=0) if op == 'mean' else np.median(np.abs(arr - center), axis=0)

                # store numeric (scalar or list)
                row_center[key] = center.item() if center.size == 1 else center.tolist()
                # format string cell
                texts = [fmt.format(c, d) for c, d in zip(center.flatten(), disp.flatten())]
                row_fmt[key] = '; '.join(texts)

            num_log.record(**row_center)
            fmt_rows.append(row_fmt)

        self.num_agg = num_log
        self.fmt_agg = pd.DataFrame(fmt_rows, index=pd.Index(range(n_gen), name='Generation'))

    def save(self, stat: str = 'min', n_obj: int = 1, op: str = 'median'):
        '''
        Create output directory and save:
          - raw loglist as 'log_list.pkl'
          - aggregated logbook (pickle + CSV)
          - best individuals per run (pickle + CSV)
          - last statistics per run (CSV + boxplot)
          - evolution plot (single-objective) or Pareto plot (multi-objective)
        '''
        os.makedirs(self.path, exist_ok=True)
        if not self.loglist:
            raise ValueError('No logbooks to save.')

        ngen = len(self.statres[stat][0])
        nvar = len(self.evolres['best'][0][0])

        # 1. Save raw loglist
        with open(os.path.join(self.path, 'log_list.pkl'), 'wb') as f:
            pickle.dump(self.loglist, f)

        # 2. Aggregate logs
        self._aggregate(op)
        with open(os.path.join(self.path, f'log_{op}.pkl'), 'wb') as f:
            pickle.dump(self.num_agg, f)
        self.saveCSV(self.fmt_agg, filename=f'log_{op}', index_name='Generation')

        # 3. Save best individuals
        with open(os.path.join(self.path, 'hof.pkl'), 'wb') as f:
            pickle.dump(self.evolres, f)
        self.saveCSV(self.evolres, filename='best_individuals', index_name='Run')

        # 4. Save last statistics
        self.saveCSV(self.lastats, filename='last_stats', index_name='Run')
        self.lastatbox(self.lastats, targets=getattr(self, 'targets', None), ngen=ngen, nvar=nvar)

        # 5. Plot evolution or Pareto
        if n_obj == 1:
            self.plotEvolution(self.statres, stat, targets=getattr(self, 'targets', None), agg_op=op, agg_log=self.num_agg, ngen=ngen, nvar=nvar)
        else:
            n_runs = len(self.paretos)
            self.plot2DPareto(self.paretos, targets=getattr(self, 'targets', None), true_pareto=getattr(self, 'true_pareto', None), n_runs=n_runs, nvar=nvar, ngen=ngen)