import os
import argparse
from operator import itemgetter

from pymoo.problems import get_problem
from deap.tools import uniform_reference_points

from deapsleep.src.metrics.final_evaluation import FinalEvaluation
from deapsleep.src.metrics.so_metrics import SOPerformanceMetrics
from deapsleep.src.metrics.mo_metrics import MOPerformanceMetrics
from deapsleep.src.utils.statistical_tests import get_distrib, normtest, perform_test
from deapsleep.experiments.utils import load_pickle, load_yaml, load_internal, parse_extra_args, apply_overrides, format_version

def main(config):
   
    dirname, probname, v1, v2, aggr_op, probtype, stat, \
    symlog_thresh, y_upper_range, y_lower_range = itemgetter(
        'dirname', 'problem', 'version1', 'version2', 'aggregation_op', 'probtype', 'stat',
        'symlog_thresh', 'y_upper_range', 'y_lower_range'
    )(config)

    if os.path.normpath(dirname).endswith(probtype):
        basepath = os.path.join(dirname, probname)
    else:
        basepath = os.path.join(dirname, probtype, probname)
    path1, path2 = map(lambda v: os.path.join(basepath, v), (v1, v2))

    v1 = format_version(v1)
    v2 = format_version(v2)

    # aggregated logbooks
    agg_logs = {
        v1: load_pickle(path1, f'log_{aggr_op}.pkl'),
        v2: load_pickle(path2, f'log_{aggr_op}.pkl')
    }

    # lobgook lists (per run)
    loglists = {
        v1: load_pickle(path1, 'log_list.pkl'),
        v2: load_pickle(path2, 'log_list.pkl')
    }

    # halls of fame
    hofs = {
        v1: load_pickle(path1, 'hof.pkl'),
        v2: load_pickle(path2, 'hof.pkl')
    }

    n_var = len(hofs[v1]['best'][0][0])
    ngen = len(agg_logs[v1])

     # final evaluation testing using best individuals per run
    try:
        problem = get_problem(probname, n_var=n_var)
    except TypeError:
        problem = get_problem(probname)
    evalfunc = problem.evaluate
    targets = problem.ideal_point() if hasattr(problem, 'ideal_point') else None
    
    if probtype == 'single':
        target, max_gen, thr, budget, gen_target = itemgetter(
            'target', 'max_gen', 'thr', 'budget', 'gen_target'
        )(config)
        
        # evolution metrics
        metrics = SOPerformanceMetrics(
            agg_logs=agg_logs,
            loglists=loglists,
            target=target,
            max_gen=max_gen,
            thr=thr,
            stat=stat,
            budget=budget
        )

        metrics.save_markdown(basepath, f'{v1}_{v2}_comparison.txt')
        
        final_eval = FinalEvaluation(
            {v1: hofs[v1], v2: hofs[v2]}, 
            evalfunc,
            probname=probname,
            v1=v1,
            v2=v2,
            ngen=ngen,
            nvar=n_var,
            targets=targets,
            symlog_thresh=symlog_thresh,
            y_lower_range=y_lower_range,
            y_upper_range=y_upper_range
        )
        final_eval.compute_grouped_stats()
        final_eval.save_markdown(basepath, f'{v1}_{v2}_final_evaluation.txt')
        final_eval.show_grouped(basepath, f'{v1}_{v2}_final_evaluation_grouped.png')
        final_eval.save_summary(basepath, f'{v1}_{v2}_summary.txt')

        # fitness distributions at target generation
        distrib1 = get_distrib(loglists[v1], stat, gen_target)
        distrib2 = get_distrib(loglists[v2], stat, gen_target)

        # normality tests
        pval1 = normtest(path1, distrib1, shapiro=True, probplot=True, histplot=True)
        pval2 = normtest(path2, distrib2, shapiro=True, probplot=True, histplot=True)

        perform_test(basepath, f'{v1}_{v2}', pval1, pval2, distrib1, distrib2)

    elif probtype == 'multi':
        n_obj, P, stat = itemgetter(
            'n_obj', 'P', 'stat'
        )(config)
        try:
            ref_front = get_problem(probname).pareto_front()
        except ValueError as e:
            raise ValueError(
                f"Failed to get Pareto front for problem '{probname}': {e}"
            )
        
        final_eval = FinalEvaluation(
            {v1: hofs[v1], v2: hofs[v2]}, 
            evalfunc,
            probname=probname,
            v1=v1,
            v2=v2,
            ngen=ngen,
            nvar=n_var
            )
        fronts = final_eval.extract_fronts()

        if n_obj is None or P is None:
            raise ValueError("Number of objectives or P must be provided for computing the reference points.")
        ref_points = uniform_reference_points(n_obj, P)
        
        metrics = MOPerformanceMetrics(
            fronts,
            ref_front,
            ref_points=ref_points
        )
        distribs1 = metrics.get_distributions(v1)
        distribs2 = metrics.get_distributions(v2)
        metrics.save_markdown(basepath, f'{v1}_{v2}_final_eval.txt')

        final_eval.plot_scatterplot(distribs1, distribs2, basepath, f'{v1}_{v2}_final_evaluation_grouped.png')

        name_metrics =  distribs1.keys()
        for metric in name_metrics:
            distrib1 = distribs1[metric]
            distrib2 = distribs2[metric]
            
            pval1 = normtest(path1, distrib1, metric=f'_{metric}', shapiro=True)
            pval2 = normtest(path2, distrib2, metric=f'_{metric}', shapiro=True)

            perform_test(basepath, f'{v1}_{v2}', pval1, pval2, distrib1, distrib2, metric=f'_{metric}', corr=len(name_metrics))

    else:
        raise ValueError(f"Unsupported run type: {probtype}")

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
        params = load_internal(args.config, configtype='evalconfig')
    else:
        params = load_yaml(args.config)

    overrides = parse_extra_args(remaining)
    if overrides:
        apply_overrides(params, overrides)

    main(params)
