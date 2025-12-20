import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
from collections import defaultdict
from deapsleep.src.utils import Visualizer
from utils.utils import load_logbooks
from utils.paths import RESULTS_DIR

class Plotter(Visualizer):
    '''
    Extend Visualizer to plot stored logbooks.
    '''
    def __init__(self, problem: str, version: str):
        super().__init__(problem, version)
        self.path = os.path.join(RESULTS_DIR, problem.replace('.', os.sep), version)

    def getdata(self, hof_path: str, log_list_path: str, log_agg_path: str):

        # Load .pkl files
        with open(hof_path, "rb") as f:
            self.hof = pickle.load(f)
        with open(log_list_path, "rb") as f:
            self.loglist = pickle.load(f)
        with open(log_agg_path, "rb") as f:
            self.log_agg = pickle.load(f)

        self.statres = defaultdict(list)
        self.lastats = defaultdict(list)

        # logbook keys
        stat_keys = self.loglist[0].header

        for key in stat_keys:
            self.statres[key] = [log.select(key) for log in self.loglist]
            self.lastats[key] = [series[-1] for series in self.statres[key]]
        self.ngen = len(self.loglist[0])

        ffirst = self.hof['best'][0][0]
        self.nvar = len(ffirst)
        self.nobj = len(ffirst.fitness.values)
        self.paretos = [
            [tuple(ind.fitness.values) for ind in archive] 
            for archive in self.hof['best']
        ]

    def plot(self, stat: str = 'min', op: str = 'median', plot_tsp=False):
        '''
        Plot results:
            - lastatbox: boxplot of last statistics across runs
            - plotEvolution: evolution of 'stat' across generations (single-objective)
            - plot2DPareto: observed vs true Pareto fronts (multi-objective)
        Parameters:
        -----------
        stat    : statistic to plot (e.g., 'min', 'avg', 'std')
        op      : aggregation operator for evolution plots ('mean' or 'median')
        '''
        
        try:
            with open(os.path.join(self.path, 'problem_instance.pkl'), 'rb') as f:
                p = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                "Problem instance file not found. "
                "Ensure that 'problem_instance.pkl' is present in the results folder."
            )

        # Last statistic boxplot
        self.lastatbox(
            self.lastats, 
            targets=p.ideal_point(), 
            ngen=self.ngen, 
            nvar=self.nvar
        )

        # Plot evolution trajectories for single objective...
        if self.nobj == 1:
            self.plotEvolution(
                self.statres, 
                stat, 
                targets=p.ideal_point(), 
                agg_op=op, 
                agg_log=self.log_agg, 
                ngen=self.ngen, 
                nvar=self.nvar
            )
        else: # ...or theoretical vs observed Pareto fronts for multi-objective
            n_runs = len(self.paretos)
            self.plot2DPareto(
                self.paretos, 
                targets=p.ideal_point(), 
                true_pareto=p.pareto_front(), 
                n_runs=n_runs, 
                nvar=self.nvar, 
                ngen=self.ngen
            )
        
        if hasattr(p, 'cities'): # then TSP
            from pymoo.problems.single.traveling_salesman import visualize
            bests = [
                r[0] # best individual per run
                for r in self.hof['best']
            ]
            bestfits = np.array([
                ind.fitness.values[0] 
                for ind in bests]
            )
            # individual with fitness closest to the median fitness across all best individuals
            idx = np.argmin(np.abs(bestfits - np.median(bestfits)))
            visualize(p, bests[idx], show=False)

            plt.savefig(os.path.join(self.path, 'tsp_median_best.png'), dpi=300, bbox_inches='tight')
            plt.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Plot stored logbooks from experiments.")

    # required
    parser.add_argument('--problem', type=str, required=True,
                        help="Problem name (e.g., single.ackley).")
    
    # required
    parser.add_argument('--version', type=str, required=True,
                        help="Experiment version (default: base).")
    
    parser.add_argument('--hof', type=str, help="Path to Hall of Fame file (pickle).")
    parser.add_argument('--logs', type=str, help="Path to list of logbooks file (pickle).")
    parser.add_argument('--agg', type=str, help="Path to aggregated logbook file (pickle).")

    parser.add_argument('--stat', type=str, default='min',
                        help="Statistic to plot (default: min).")
    parser.add_argument('--op', type=str, default='median',
                        help="Aggregation operator (mean or median).")

    parser.add_argument('-i', '--internal', action='store_true',
                        help="Load internal result files automatically.")
    
    args = parser.parse_args()

    def build_path(filename):
        folder = args.problem.replace('.', os.sep)
        return os.path.join(RESULTS_DIR, folder, args.version, filename)

    # Load internal files if flag is used
    if args.internal:
        hof_path, log_list_path, log_agg_path = load_logbooks(args.problem, args.version)
    else:
        # Otherwise use explicit paths (or default file names)
        hof_path = args.hof or build_path('hof.pkl')
        log_list_path = args.logs or build_path('log_list.pkl')
        log_agg_path = args.agg or build_path('log_median.pkl')

    # Instantiate and run the plotter
    plotter = Plotter(args.problem, args.version)
    plotter.getdata(hof_path, log_list_path, log_agg_path)
    plotter.plot(args.stat, args.op)