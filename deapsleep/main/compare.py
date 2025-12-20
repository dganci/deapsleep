import os
import pickle
import argparse
import deapsleep.main as d_
from scipy.stats import wilcoxon
from deap.tools import uniform_reference_points
from utils.paths import RESULTS_DIR

class Compare:

    def __init__(self, config):
        self.__dict__.update(config)

    def _getdata(self):

        # bbox formatting
        if hasattr(self, 'instance') and 'bbob' in self.problem:
            self.problem += f"-{self.instance}"

        self.basepath = os.path.join(RESULTS_DIR, self.problem)
        path1, path2 = map(
            lambda v: os.path.join(self.basepath, v),
            (self.version1, self.version2)
        )

        self.v1 = d_.format_version(self.version1)
        self.v2 = d_.format_version(self.version2)

        # aggregated logbooks
        agg_logs = {
            self.v1: d_.load_pickle(path1, f'log_{self.aggregation_op}.pkl'),
            self.v2: d_.load_pickle(path2, f'log_{self.aggregation_op}.pkl')
        }

        # lobgook lists (per run)
        loglists = {
            self.v1: d_.load_pickle(path1, 'log_list.pkl'),
            self.v2: d_.load_pickle(path2, 'log_list.pkl')
        }

        # halls of fame
        hofs = {
            self.v1: d_.load_pickle(path1, 'hof.pkl'),
            self.v2: d_.load_pickle(path2, 'hof.pkl')
        }

        try:
            with open(os.path.join(path1, 'problem_instance.pkl'), 'rb') as f:
                p1 = pickle.load(f)
            with open(os.path.join(path2, 'problem_instance.pkl'), 'rb') as f:
                p2 = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                "Problem instance file not found. "
                "Ensure that 'problem_instance.pkl' is present in the results folder."
            )

        if p1.n_var != p2.n_var:
            raise ValueError(
                f"Cannot compare versions with different number of variables: "
                f"{self.v1} has {p1.n_var}, {self.v2} has {p2.n_var}."
            )
        self.n_var = p1.n_var
        if len(agg_logs[self.v1]) != len(agg_logs[self.v2]):
            raise ValueError(
                f"Cannot compare versions with different number of generations: "
                f"{self.v1} has {len(agg_logs[self.v1])}, {self.v2} has {len(agg_logs[self.v2])}."
            )
        self.n_gen = len(agg_logs[self.v1])
        self.evaluate = p1.evaluate
        self.targets = p1.ideal_point() if hasattr(p1, 'ideal_point') else None

        if self.probtype == 'multi':
            if not hasattr(self, 'n_obj'):
                raise ValueError("Number of objectives 'n_obj' must be specified for multi-objective problems.")
            if not hasattr(self, 'P'):
                raise ValueError("Reference points parameter 'P' must be specified for multi-objective problems.")
            try:
                self.ref_front = p1.pareto_front()
            except ValueError as e:
                raise ValueError(
                    f"Failed to get Pareto front for problem '{self.problem}': {e}"
                )
            try:
                ref_points = uniform_reference_points(self.n_obj, self.P)
            except Exception as e:
                raise ValueError(
                    f"Failed to generate reference points for n_obj={self.n_obj} and P={self.P}: {e}"
                )

        return agg_logs, loglists, hofs

    def compare(self):

        agg_logs, loglists, hofs = self._getdata()

        # final evaluation testing using best individuals per run
        fe = d_.PostHocAnalysis(
                {self.v1: hofs[self.v1], self.v2: hofs[self.v2]}, 
                probname=self.problem,
                v1=self.v1,
                v2=self.v2,
                ngen=self.n_gen,
                nvar=self.n_var,
                targets=self.targets
        )
        
        if self.probtype == 'single':
            
            fe.compute_grouped_stats()
            if self.save_full:
                fe.save_markdown(self.basepath, f'{self.v1}_{self.v2}_final_evaluation_table.txt')
            fe.show_grouped(self.basepath, f'{self.v1}_{self.v2}_boxplot.png')
            fe.save_summary(self.basepath, f'{self.v1}_{self.v2}_summary.txt')

            # fitness distributions at last generation
            distrib1 = d_.get_distrib(loglists[self.v1], self.stat, -1)
            distrib2 = d_.get_distrib(loglists[self.v2], self.stat, -1)

            M, p_val = wilcoxon(distrib1, distrib2)
            text = f"\n--- {'wilcoxon signed-rank test'.capitalize()} ---\nMetric = {M}\np-value = {p_val}\n"
            with open(os.path.join(self.basepath, f'{self.v1}_{self.v2}_summary.txt'), "a") as f:
                f.write(text)

        elif self.probtype == 'multi':

            fronts = fe.extract_fronts()

            metrics = d_.MOPerformanceMetrics(
                fronts,
                self.ref_front,
                ref_points=self.ref_points
            )

            distribs1 = metrics.get_distributions(self.v1)
            distribs2 = metrics.get_distributions(self.v2)

            metrics.save_markdown(
                self.basepath, 
                f'{self.v1}_{self.v2}_summary.txt'
            )

            fe.plot_scatterplot(
                distribs1, 
                distribs2, 
                self.basepath, 
                f'{self.v1}_{self.v2}_scatter.png'
            )

            for metric in distribs1.keys():
                distrib1 = distribs1[metric]
                distrib2 = distribs2[metric]

                M, p_val = wilcoxon(distrib1, distrib2)
                text = f"\n--- {f'wilcoxon signed-rank test - {metric}'.capitalize()} ---\nMetric = {M}\np-value = {p_val}\n"
                with open(os.path.join(self.basepath, f'{self.v1}_{self.v2}_summary.txt'), "a") as f:
                    f.write(text)

        else:
            raise ValueError(f"Unsupported run type: {self.probtype}")        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Compare two versions of an optimization problem'
        )
    parser.add_argument('--config', type=str, required=True,
                        help='YAML configuration file for the experiment.')
    parser.add_argument('-i', '--internal', action='store_true',
                        help='Load internal configuration instead of YAML file.')

    args, remaining = parser.parse_known_args()
    if args.internal:
        params = d_.load_internal(args.config, configtype='evalconfig')
    else:
        params = d_.load_yaml(args.config)

    overrides = d_.parse_extra_args(remaining)
    if overrides:
        d_.apply_overrides(params, overrides)

    Compare(params).compare()