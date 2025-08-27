import os
import math
import warnings

import numpy as np
import pandas as pd

class SOPerformanceMetrics:
    '''
    Compute single-objective metrics (ACR, FTRT, FBP, ERT) given:
      - agg_logs: aggregated DataFrame or list of dicts per version
      - loglists: per-version list of DataFrames or dict-lists
      - stat: which statistic to use (e.g. 'mean')
      - optional parameters: target, max_gen, thr (threshold), budget
    '''

    def __init__(self, agg_logs, loglists, stat, **kwargs):
        self.agg_logs = agg_logs
        self.loglists = loglists
        self.stat = stat
        self.target = kwargs.get('target')
        self.max_gen = kwargs.get('max_gen')
        self.thr = kwargs.get('thr')
        self.budget = kwargs.get('budget')
        self._compute_results()

    def acr(self):
        '''
        Average Convergence Rate: requires target and max_gen.
        Returns (ACR_value, improvements_count) or None if insufficient data.
        '''
        if self.target is None or self.max_gen is None:
            return None

        gen_means = np.asarray(self.logbook.get('mean', []), float)
        if len(gen_means) < 2:
            warnings.warn('Not enough generations for ACR.')
            return None

        arr = gen_means[: self.max_gen]
        e_prev = abs(self.target - arr[0])
        logsum = 0.0
        improvements = 0

        for m in arr[1:]:
            e_curr = abs(self.target - m)
            if e_curr < e_prev:
                logsum += math.log((e_prev - e_curr) / e_prev)
                improvements += 1
                e_prev = e_curr

        if improvements == 0:
            return 0.0, 0
        return math.exp(logsum / improvements), improvements

    def ftrt(self):
        '''
        Fixed-Target Running Time: first generation where stat <= thr.
        Returns (cum_evals, generation_index) or None if never reached.
        '''
        if self.thr is None:
            return None

        values = np.asarray(self.logbook.get(self.stat, []), float)
        mask = values <= self.thr
        if not mask.any():
            return None

        idx = int(np.argmax(mask))
        evals = np.asarray(self.logbook.get('evals', []), int)
        cumevals = np.cumsum(evals)
        return int(cumevals[idx]), idx

    def fbp(self):
        '''
        Fixed-Budget Performance: value of stat at last gen within budget.
        Returns (stat_value, generation_index) or None if budget too small.
        '''
        if self.budget is None:
            return None

        evals = np.asarray(self.logbook.get('evals', []), int)
        cumevals = np.cumsum(evals)
        mask = cumevals <= self.budget
        if not mask.any():
            return None

        idx = int(np.where(mask)[0][-1])
        values = np.asarray(self.logbook.get(self.stat, []), float)
        if idx >= len(values):
            return None
        return float(values[idx]), idx

    @staticmethod
    def ert(metrics_runs, thr, budget):
        '''
        Expected Running Time over multiple runs:
          ERT = (sum(successes) + failures * budget) / num_successes
        Returns (ERT_value, num_successes) or None if thr/budget missing.
        '''
        if thr is None or budget is None:
            return None

        R = len(metrics_runs)
        successes = []
        for m in metrics_runs:
            res = m.ftrt()
            if res and np.isfinite(res[0]):
                successes.append(res[0])

        s = len(successes)
        if s == 0:
            return float('inf'), 0
        ert = (sum(successes) + (R - s) * budget) / s
        return int(ert), s

    def _compute_results(self):
        '''
        For each version in agg_logs:
          1. Build logbook DataFrame
          2. Compute ACR, FTRT, FBP
          3. Build per-run SOPerformanceMetrics to compute ERT
          4. Store results in self.df (DataFrame indexed by version)
        '''
        records = {}

        for label, log in self.agg_logs.items():
            # Ensure DataFrame
            self.logbook = pd.DataFrame(log) if not isinstance(log, pd.DataFrame) else log.copy()
            record = {}

            # ACR
            acr_res = self.acr()
            if acr_res:
                acr_val, acr_imp = acr_res
                record[f'ACR (target={self.target})'] = f'Value={acr_val:.2e}, Imp={acr_imp}'

            # FTRT
            ftrt_res = self.ftrt()
            if ftrt_res:
                fevals, fgen = ftrt_res
                record[f'FTRT ({self.stat}≤{self.thr})'] = f'Evals={fevals}, Gen={fgen}'

            # FBP
            fbp_res = self.fbp()
            if fbp_res:
                fbp_val, fbp_gen = fbp_res
                record[f'FBP( budget={self.budget})'] = f'Value={fbp_val:.2e}, Gen={fbp_gen}'
            else:
                warnings.warn(f'Insufficient budget for {label}.')

            # ERT: build metrics for each run in loglists[label]
            runs = self.loglists.get(label, [])
            metrics_runs = []
            for run_log in runs:
                df_run = pd.DataFrame(run_log) if not isinstance(run_log, pd.DataFrame) else run_log
                mr = SOPerformanceMetrics(
                    {label: df_run}, {},
                    self.stat,
                    target=self.target,
                    max_gen=self.max_gen,
                    thr=self.thr,
                    budget=self.budget
                )
                metrics_runs.append(mr)

            ert_res = self.ert(metrics_runs, self.thr, self.budget)
            if ert_res:
                ert_evals, ert_succ = ert_res
                record[f'ERT(thr={self.thr}, bud={self.budget})'] = f'Evals={ert_evals}, Succ={ert_succ}'

            records[label] = record

        self.df = pd.DataFrame.from_dict(records, orient='index')

    def save_csv(self, path: str, filename: str):
        '''
        Save metrics table (transposed) to CSV.
        '''
        self.df.T.to_csv(os.path.join(path, filename))

    def save_markdown(self, path: str, filename: str):
        '''
        Save metrics table (transposed) in Markdown format.
        '''
        with open(os.path.join(path, filename), 'w') as f:
            f.write(self.df.T.to_markdown(tablefmt='grid'))