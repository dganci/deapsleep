import os
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from deapsleep.experiments.utils import format_version

class Visualizer:
    '''
    Base class for plotting and saving CSVs in experiment directories.
    '''

    def __init__(self, dirname: str, probname: str, version: str):
        '''
        Create output directory structure for this problem/version.
        '''
        self.dirname = dirname
        self.probname = probname
        self.version = format_version(version)
        self.path = os.path.join(dirname, probname, version)
        os.makedirs(self.path, exist_ok=True)

    def saveCSV(self, res: dict, filename: str = '', index_name: str = '') -> None:
        '''
        Save a dictionary or DataFrame to CSV under self.path.
        If 'res' is not a DataFrame, convert it first.
        '''
        if not isinstance(res, pd.DataFrame):
            res = pd.DataFrame(res)
        res.index.name = index_name
        res.to_csv(os.path.join(self.path, f'{filename}.csv'), sep=',')

    def plotEvolution(self, evols: dict, stat: str,
                      targets: list[float] | None = None, agg_op: str = None,
                      agg_log: any = None, xstep: float = 0.1, ngen=0, nvar=0) -> None:
        '''
        Plots and saves multiple objective function values evolutions over generations.
        '''
        fig, ax = plt.subplots(figsize=(12, 8))

        legend_handles = []
        n_runs = len(evols[stat])
        cmap = plt.get_cmap('viridis')
        colors = cmap(np.linspace(0, 1, n_runs))

        # Plot each run’s series
        for i, series in enumerate(evols[stat]):
            ax.plot(
                range(len(series)), 
                series,
                color=colors[i],
                linestyle='-', 
                linewidth=0.4,
                marker='.', 
                alpha=0.25
            )

        # Overlay aggregated trend if provided
        if agg_log is not None:
            trend = np.array(agg_log.select(stat))
            ax.plot(range(len(trend)), trend, color='black', alpha=0.8)
            legend_handles.append(
                Line2D([0], [0], color='black', lw=1, label=f'Tendency ({agg_op})')
            )

        # Configure x-axis ticks
        last_len = len(evols[stat][-1])
        xticks = list(range(0, last_len, int(last_len * xstep)))
        if xticks[-1] != last_len - 1:
            xticks.append(last_len - 1)
        ax.set_xlabel('Generation')
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks)

        ax.set_ylabel('Objective Value')

        # Draw target lines if provided
        if targets is not None:
            for idx, tval in enumerate(targets, start=1):
                ax.axhline(y=tval, color='red', linestyle='--', linewidth=1)
                legend_handles.append(
                    Line2D([0], [0], color='red', lw=1, linestyle='--', label=f'Target = {tval}')
                )

        ax.grid(True, axis='both', linestyle='--', alpha=0.3, color='lightgray')
        ax.legend(handles=legend_handles, loc='best')
        titlepos = ax.get_position().x0 + ax.get_position().width * 0.5

        fig.suptitle(
            rf'$\bf{{{self.probname.capitalize()}}}$ - {self.version} - Objective Value Evolutions',
            fontsize=18,
            x=titlepos,      
            y=1.0,     
            ha='center'
        )

        fig.text(
            titlepos,        
            1.02,        
            f'{n_runs} runs | {ngen} generations | {nvar} variables',
            transform=ax.transAxes,
            ha='center',
            va='center', 
            fontsize=12
        )
        
        fig.tight_layout()
        fig.savefig(os.path.join(self.path, 'evolution.png'))
        plt.close(fig)

    def lastatbox(self, lastats: dict, targets: list[float] | None = None, seed: int = 123, ngen=0, nvar=0) -> None:
        '''
        Create and save boxplots of final-generation objective function value for each run and objective.
        '''
        mean = np.array(lastats['mean']).T
        med = np.array(lastats['med']).T
        n_obj, n_runs = mean.shape

        fig, ax = plt.subplots(figsize=(12, 8))
        offset = 0.2
        box_width = 0.3
        legend_handles = [
            Line2D([0], [0], color='orange', lw=1.5, label='Median'),
            Line2D([0], [0], color='green', lw=1.5, linestyle='--', label='Mean')
        ]

        random.seed(seed)
        # Generate a pastel-like random color per objective
        colors = [
            f'#{"".join(random.choices("89ABCDEF", k=3))}{"".join(random.choices("89ABCD", k=3))}'
            for _ in range(n_obj)
        ]

        # Plot box for each run and objective
        for run_idx in range(n_runs):
            for obj_idx in range(n_obj):
                pos = run_idx + 1 + (obj_idx - (n_obj - 1)/2) * offset
                stats = {
                    'whislo': lastats['min'][run_idx][obj_idx],
                    'q1': lastats['q1'][run_idx][obj_idx],
                    'med': lastats['med'][run_idx][obj_idx],
                    'q3': lastats['q3'][run_idx][obj_idx],
                    'whishi': lastats['max'][run_idx][obj_idx],
                    'mean': lastats['mean'][run_idx][obj_idx],
                }
                ax.bxp(
                    [stats],
                    positions=[pos],
                    widths=box_width,
                    patch_artist=True,
                    boxprops={'facecolor': colors[obj_idx % len(colors)], 'alpha': 0.7},
                    showfliers=False,
                    showmeans=True,
                    meanline=True
                )

        x = np.arange(1, n_runs + 1)
        for obj_idx in range(n_obj):
            ax.plot(x, mean[obj_idx], color='green', linestyle='--', linewidth=1.5, alpha=0.5)
            ax.plot(x, med[obj_idx], color='orange', linestyle='-', linewidth=1.5, alpha=0.5)

        xticks = np.arange(0, n_runs + 1, step=5)
        ax.set_xlabel('Run')
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks)

        ax.set_ylabel('Objective Function Value')

        # Draw target lines if provided
        if targets is not None:
            for idx, tval in enumerate(targets, start=1):
                ax.axhline(y=tval, color='red', linestyle='--', linewidth=1)
                legend_handles.append(
                    Line2D([0], [0], color='red', lw=1, linestyle='--', label=f'Target = {tval}')
                )

        ax.grid(True, axis='both', linestyle='--', alpha=0.3, color='lightgray')
        ax.legend(handles=legend_handles, loc='best')
        titlepos = ax.get_position().x0 + ax.get_position().width * 0.5

        fig.suptitle(
            rf'$\bf{{{self.probname.capitalize()}}}$ - {self.version} - Last Generation Objective Value Distributions',
            fontsize=18,
            x=titlepos,      
            y=0.995,     
            ha='center'
        )

        fig.text(
            titlepos,        
            0.945,        
            f'{n_runs} runs | {ngen} generations | {nvar} variables',
            ha='center',
            va='center', 
            fontsize=12
        )

        plt.tight_layout()
        fig_path = os.path.join(self.path, 'last_stats_boxplot.png')
        plt.savefig(fig_path)
        plt.close(fig)

    def histplot(self, values: list[float], name: str = '') -> None:
        '''
        Plot histogram (with KDE overlay) of 'values',
        and save as '<name>_histplot.png' in self.path.
        '''
        plt.figure(figsize=(8, 6))
        plt.hist(values, bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel('Values')
        plt.ylabel('Absolute Frequency')
        plt.title(f'{name} Values Distribution')
        plt.grid(axis='y', linestyle='--', alpha=0.4)
        plt.tight_layout()

        plt.savefig(os.path.join(self.path, f'{name}_histplot.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot2DPareto(
        self,
        paretos: list[tuple],
        true_pareto: np.ndarray | None = None,
        targets: tuple[float, float] | None = None,
        n_runs: int = None,
        ngen: int = None,
        nvar: int = None
    ) -> None:
        '''
        Plot observed Pareto fronts (scatter + line) for each run in 'paretos',
        overlay true Pareto front if given, and mark 'targets'.
        Saves figure as 'pareto.png'.
        '''

        fig, ax = plt.subplots(figsize=(12, 8))

        for front in paretos:
            x, y = zip(*sorted(front, key=lambda tup: tup[0]))
            ax.scatter(x, y, alpha=0.3, marker='.')
            ax.plot(x, y, alpha=0.3, linestyle='-', linewidth=1)
        ax.scatter([], [], c='gray', marker='.', label='Observed Pareto Front')

        if true_pareto is not None:
            ax.plot(
                true_pareto[:, 0], true_pareto[:, 1],
                c='black', marker='2', label='Theoretical Pareto Front'
            )

        if targets is not None:
            ax.scatter(
                targets[0], targets[1],
                c='red', marker='*', s=200, label='Target'
            )

        ax.set_xlabel('1st Objective')
        ax.set_ylabel('2nd Objective')
        ax.grid(True, axis='both', linestyle='--', alpha=0.3, color='lightgray')
        ax.legend(loc='best')

        titlepos = ax.get_position().x0 + ax.get_position().width * 0.5

        fig.suptitle(
            rf'$\bf{{{self.probname.capitalize()}}}$ - {self.version} - Pareto Fronts',
            fontsize=18,
            x=titlepos,
            y=1.0,
            ha='center'
        )

        if n_runs is not None and ngen is not None and nvar is not None:
            fig.text(
                titlepos,
                1.02,
                f'{n_runs} runs | {ngen} generations | {nvar} variables',
                transform=ax.transAxes,
                ha='center',
                va='center',
                fontsize=12
            )

        fig.tight_layout()
        fig.savefig(os.path.join(self.path, 'pareto.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

    # def plot3DPareto(self, directory: str, name: str, bestinds, true_pareto=None, targets=None) -> None:
    #     '''
    #     3D Pareto front plot (commented out):
    #       - 'bestinds': list of nondominated sets per run (each a list of tuples)
    #       - 'true_pareto': np.ndarray of true Pareto points
    #       - 'targets': 3-element tuple for 3D target
    #     '''
    #     fig = plt.figure(figsize=(10, 8))
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.set_title('Pareto Fronts (3D)', fontweight='bold')
    #
    #     for nondom in bestinds:
    #         arr = np.array(nondom)
    #         ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], alpha=0.3, marker='.', color='gray')
    #     ax.scatter([], [], [], c='gray', marker='.', label='Observed Pareto Front')
    #
    #     if true_pareto is not None:
    #         ax.plot(true_pareto[:, 0], true_pareto[:, 1], true_pareto[:, 2],
    #                 c='black', marker='2', label='Theoretical Pareto Front')
    #
    #     if targets is not None:
    #         ax.scatter(targets[0], targets[1], targets[2], c='red', marker='*', s=200, label='Target')
    #
    #     ax.set_xlabel('1st Objective')
    #     ax.set_ylabel('2nd Objective')
    #     ax.set_zlabel('3rd Objective')
    #     ax.grid(True, linestyle='--', alpha=0.3, color='lightgray')
    #     ax.legend(loc='best')
    #     plt.tight_layout()
    #
    #     plt.savefig(os.path.join(directory, f'{name}_pareto3D.png'), dpi=300, bbox_inches='tight')
    #     plt.close()