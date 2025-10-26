import os
import warnings
import numpy as np
import pandas as pd
from math import hypot, sqrt
try:
    from scipy.spatial.distance import cdist
    _scipy_available = True
except ImportError:
    _scipy_available = False
try:
    # try importing the C version
    from deap.tools._hypervolume import hv
except ImportError:
    # fallback on python version
    from deap.tools._hypervolume import pyhv as hv

def diversity(front, extremes):
    '''
    Diversity (spread) metric as in NSGA-II.

    Parameters
    ----------
    front : array-like, shape (N, M)
        Pareto front points sorted by one objective.
    extremes : tuple of array-like
        (min_point, max_point) of true Pareto front extremes.

    Returns
    -------
    float
        Diversity metric (lower is better).
    '''
    first_ext, last_ext = extremes
    df = hypot(front[0][0] - first_ext[0], front[0][1] - first_ext[1])
    dl = hypot(front[-1][0] - last_ext[0], front[-1][1] - last_ext[1])

    if front.shape[0] < 2:
        return df + dl

    # Pairwise distances along the front
    distances = [hypot(a[0] - b[0], a[1] - b[1])
                 for a, b in zip(front[:-1], front[1:])]
    mean_d = np.mean(distances)
    deviation = sum(abs(d - mean_d) for d in distances)

    return (df + dl + deviation) / (df + dl + len(distances) * mean_d)


def convergence(front, optimal):
    '''
    Convergence: average nearest Euclidean distance from front to optimal Pareto front.
    '''
    dists = []
    for p in front:
        sq = np.sum((optimal - p) ** 2, axis=1)
        dists.append(sqrt(np.min(sq)))
    return float(np.mean(dists))

def hypervolume(front, ref_points=None):
    '''
    Hypervolume indicator for maximization problems,
    assuming minimization of all objectives.
    
    Parameters
    ----------
    front : array-like, shape (n_points, n_obj)
    ref_points : None or array-like
        Reference points for hypervolume calculation:
        - None: base = max(front, axis=0).
        - 1D array (n_obj,): base = ref_points.
        - 2D array (k, n_obj): base = max(ref_points, axis=0).
    '''

    arr = np.asarray(front)
    wobj = -arr # assuming minimization

    if ref_points is None:
        base = np.max(wobj, axis=0)
    else:
        rp = np.asarray(ref_points)
        if rp.ndim == 1:
            base = rp
        elif rp.ndim == 2:
            base = np.max(rp, axis=0)
        else:
            raise ValueError(f"reference point has invalid shape: {rp.shape}")

    ref_point = base * 1.01 # 1% margin

    return hv.hypervolume(wobj, ref_point)

def igd(obtained, optimal):
    '''
    Inverted Generational Distance (IGD).
    '''
    if not _scipy_available:
        raise ImportError("IGD requires scipy.spatial.distance.cdist.")
    A = np.asarray(obtained)
    Z = np.asarray(optimal)
    dist_matrix = cdist(A, Z)
    # For each Z point, its nearest neighbor in A
    return float(np.mean(np.min(dist_matrix, axis=0)))


class MOPerformanceMetrics:
    '''
    Compute and store multi-objective performance metrics

    Parameters
    ----------
    fronts : dict
        {label: {'best': [array_run1, array_run2, ...]}, ...}
    ref_front : array-like, shape (K, M)
    ref_point : array-like or None
    '''

    def __init__(self, fronts, ref_front, ref_points=None):
        self.fronts = fronts
        self.ref_front = np.asarray(ref_front, float)
        self.ref_points = ref_points
        self._compute_results()

    def _compute_results(self):
        records = {}  # stores mean and std dev of each metric, for all the versions
        self.all_metrics = {}  # stores the full list of metric values per version

        for version, runs in self.fronts.items():
            metrics = {
                'Diversity': [], 
                'Convergence': [], 
                'Hypervolume': [], 
                'IGD': []
            }

            for run, front in enumerate(runs):
                front = np.asarray(front, float)

                # Diversity
                try:
                    metrics['Diversity'].append(
                        diversity(front, (self.ref_front[0], self.ref_front[-1]))
                    )
                except Exception as e:
                    warnings.warn(f"Diversity failed for {version}_{run}: {e}")

                # Convergence
                try:
                    metrics['Convergence'].append(convergence(front, self.ref_front))
                except Exception as e:
                    warnings.warn(f"Convergence failed for {version}_{run}: {e}")

                # Hypervolume
                if self.ref_points is not None:
                    try:
                        metrics['Hypervolume'].append(
                            hypervolume(front, self.ref_points)
                        )
                    except Exception as e:
                        warnings.warn(f"Hypervolume failed for {version}_{run}: {e}")

                # IGD
                try:
                    metrics['IGD'].append(igd(front, self.ref_front))
                except Exception as e:
                    warnings.warn(f"IGD failed for {version}_{run}: {e}")

            self.all_metrics[version] = metrics

            rec = {}
            for m, vals in metrics.items():
                arr = np.asarray(vals, float)
                rec[m] = float(np.nanmean(arr))
                rec[f"{m}_std"] = float(np.nanstd(arr))
            records[version] = rec

        self.df = pd.DataFrame.from_dict(records, orient='index')
        self.df.index.name = 'Version'

    def get_distributions(self, version: str):
        '''
        Return the full distributions of metrics for a given version.
        If a metric is missing, returns an empty list for it.
        '''
        metrics_names = ['Diversity', 'Convergence', 'Hypervolume', 'IGD']
        distribs = {}
        for m in metrics_names:
            try:
                distribs[m] = self.all_metrics[version][m]
            except KeyError:
                distribs[m] = []
        return distribs

    def save_csv(self, path: str, filename: str):
        '''
        Save the aggregated metrics (mean e std) per version to CSV,
        with version labels as index.
        '''
        full_path = os.path.join(path, filename)
        self.df.to_csv(full_path, index=True)

    def save_markdown(self, path: str, filename: str):
        '''
        Save the aggregated metrics (mean e std) per version in Markdown.
        '''
        full_path = os.path.join(path, filename)
        markdown = self.df.to_markdown(tablefmt='grid', index=True)
        with open(full_path, 'w') as f:
            f.write(markdown)

