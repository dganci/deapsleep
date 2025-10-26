import os
import argparse
import deapsleep.main as d_
from operator import itemgetter
from scipy.stats import wilcoxon
from pymoo.problems import get_problem
from deap.tools import uniform_reference_points

def main(config):
   
    dirname, probname, v1, v2, probtype, \
    aggr_op, save_full, stat = d_.extract_params(
        config, 
        ['dirname', 'problem', 'version1', 'version2', 'probtype'], # required keys
        {
            'aggregation_op': 'median',
            'save_full': False,
            'stat': 'min'
        } # optional keys with defaults
    )

    basepath = os.path.join(dirname, probname)
    path1, path2 = map(lambda v: os.path.join(basepath, v), (v1, v2))

    v1 = d_.format_version(v1)
    v2 = d_.format_version(v2)

    # aggregated logbooks
    agg_logs = {
        v1: d_.load_pickle(path1, f'log_{aggr_op}.pkl'),
        v2: d_.load_pickle(path2, f'log_{aggr_op}.pkl')
    }

    # lobgook lists (per run)
    loglists = {
        v1: d_.load_pickle(path1, 'log_list.pkl'),
        v2: d_.load_pickle(path2, 'log_list.pkl')
    }

    # halls of fame
    hofs = {
        v1: d_.load_pickle(path1, 'hof.pkl'),
        v2: d_.load_pickle(path2, 'hof.pkl')
    }

    n_var = len(hofs[v1]['best'][0][0])
    ngen = len(agg_logs[v1])

     # final evaluation testing using best individuals per run
    try:
        problem = get_problem(probname.lower(), n_var=n_var)
    except TypeError:
        problem = get_problem(probname.lower())
    evalfunc = problem.evaluate
    targets = problem.ideal_point() if hasattr(problem, 'ideal_point') else None

    fe = d_.FinalEvaluation(
            {v1: hofs[v1], v2: hofs[v2]}, 
            evalfunc,
            probname=probname,
            v1=v1,
            v2=v2,
            ngen=ngen,
            nvar=n_var,
            targets=targets
    )
    
    if probtype == 'single':
        
        fe.compute_grouped_stats()
        if save_full:
            fe.save_markdown(basepath, f'{v1}_{v2}_final_evaluation_table.txt')
        fe.show_grouped(basepath, f'{v1}_{v2}_boxplot.png')
        fe.save_summary(basepath, f'{v1}_{v2}_summary.txt')

        # fitness distributions at last generation
        distrib1 = d_.get_distrib(loglists[v1], stat, -1)
        distrib2 = d_.get_distrib(loglists[v2], stat, -1)

        M, p_val = wilcoxon(distrib1, distrib2)
        text = f"\n--- {'wilcoxon signed-rank test'.capitalize()} ---\nMetric = {M}\np-value = {p_val}\n"
        with open(os.path.join(basepath, f'{v1}_{v2}_summary.txt'), "a") as f:
            f.write(text)

    elif probtype == 'multi':
        n_obj, P, stat = itemgetter(
            'n_obj', 'P', 'stat'
        )(config)
        try:
            ref_front = get_problem(probname.lower()).pareto_front()
        except ValueError as e:
            raise ValueError(
                f"Failed to get Pareto front for problem '{probname}': {e}"
            )
        
        fronts = fe.extract_fronts()

        if n_obj is None or P is None:
            raise ValueError("Number of objectives or P must be provided for computing the reference points.")
        ref_points = uniform_reference_points(n_obj, P)
        
        metrics = d_.MOPerformanceMetrics(
            fronts,
            ref_front,
            ref_points=ref_points
        )
        distribs1 = metrics.get_distributions(v1)
        distribs2 = metrics.get_distributions(v2)
        metrics.save_markdown(basepath, f'{v1}_{v2}_summary.txt')

        fe.plot_scatterplot(distribs1, distribs2, basepath, f'{v1}_{v2}_scatter.png')

        name_metrics =  distribs1.keys()
        for metric in name_metrics:
            distrib1 = distribs1[metric]
            distrib2 = distribs2[metric]

            M, p_val = wilcoxon(distrib1, distrib2)
            text = f"\n--- {f'wilcoxon signed-rank test - {metric}'.capitalize()} ---\nMetric = {M}\np-value = {p_val}\n"
            with open(os.path.join(basepath, f'{v1}_{v2}_summary.txt'), "a") as f:
                f.write(text)

    else:
        raise ValueError(f"Unsupported run type: {probtype}")

def run():
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

    main(params)

if __name__ == '__main__':
    run()