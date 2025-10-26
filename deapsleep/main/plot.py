import os
import pickle
import argparse
from pymoo.problems import get_problem
from collections import defaultdict
from deapsleep.src.utils import Visualizer
from deapsleep.experiments.utils import load_logbooks

class Plotter(Visualizer):
    '''
    Extend Visualizer to plot stored logbooks.
    '''
    def __init__(self, dirname: str, problem: str, version: str):
        super().__init__(dirname, problem, version)
        self.path = os.path.join(dirname, problem.replace('.', os.sep), version)

    def getdata(self, hof_path: str, log_list_path: str, log_agg_path: str):
        '''
        Load stored results from .pkl files:
            - hall-of-fame (best individuals and Pareto fronts)
            - loglist (all runs' logbooks)
        Parameters:
        -----------
        hof_filename        : filename of the hall-of-fame .pkl (without .pkl)
        log_list_filename   : filename of the loglist .pkl (without .pkl)
        '''

        # Load .pkl files
        with open(hof_path, "rb") as f:
            hof = pickle.load(f)
        with open(log_list_path, "rb") as f:
            loglist = pickle.load(f)
        with open(log_agg_path, "rb") as f:
            self.log_agg = pickle.load(f)

        self.statres = defaultdict(list)
        self.lastats = defaultdict(list)

        # logbook keys
        stat_keys = loglist[0].header

        for key in stat_keys:
            self.statres[key] = [log.select(key) for log in loglist]
            self.lastats[key] = [series[-1] for series in self.statres[key]]
        self.ngen = len(loglist[0])

        first = hof['best'][0][0]
        self.nvar = len(first)
        self.nobj = len(first.fitness.values)
        self.paretos = [
            [tuple(ind.fitness.values) for ind in archive] 
            for archive in hof['best']
        ]

    def plot(self, stat: str = 'min', op: str = 'median'):
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

        p = get_problem(self.probname.lower(), n_var=self.nvar) \
            if self.nvar \
            else get_problem(self.probname.lower())

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

def run():

    parser = argparse.ArgumentParser(description="Plot stored logbooks from experiments.")

    parser.add_argument('--dirname', type=str, default='deapsleep/experiments/results',
                        help="Base directory for experiment results.")
    
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
        return os.path.join(args.dirname, folder, args.version, filename)

    # Load internal files if flag is used
    if args.internal:
        hof_path, log_list_path, log_agg_path = load_logbooks(args.problem, args.version)
    else:
        # Otherwise use explicit paths (or default file names)
        hof_path = args.hof or build_path('hof.pkl')
        log_list_path = args.logs or build_path('log_list.pkl')
        log_agg_path = args.agg or build_path('log_median.pkl')

    # Instantiate and run the plotter
    plotter = Plotter(args.dirname, args.problem, args.version)
    plotter.getdata(hof_path, log_list_path, log_agg_path)
    plotter.plot(args.stat, args.op)

if __name__ == '__main__':
    run()