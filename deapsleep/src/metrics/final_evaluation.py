import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from functools import partial

class FinalEvaluation:
    '''
    Compare two sets of Hall-of-Fame solutions (Baseline vs Proposed):
      1. Evaluate each solution's true objective value via evalfunc.
      2. Compute summary statistics per run/version.
      3. Plot side-by-side boxplots and output head-to-head win counts.
    '''

    def __init__(
        self, 
        hofs: dict[str, dict], 
        evalfunc: callable, 
        probname: str = None, 
        v1: str = 'Baseline', 
        v2: str = 'Individual Dropout',
        ngen: int = 0,
        nvar: int = 0,
        targets: list[float] = None,
        symlog_thresh: float = None,
        y_lower_range: tuple[float, float] = None,
        y_upper_range: tuple[float, float] = None
        ):
        self.hofs = hofs
        self.evalfunc = evalfunc
        self.probname = probname
        self.v1 = v1
        self.v2 = v2
        self.ngen = ngen
        self.nvar = nvar
        self.targets = targets
        self.symlog_thresh = symlog_thresh
        self._assemble_dataframe()
        self.y_lower_range = y_lower_range
        self.y_upper_range = y_upper_range

    def _assemble_dataframe(self):
        '''
        Evaluate every solution in each HOF and store in self.df
        '''
        records = []
        for version, data in self.hofs.items():
            for run_idx, solutions in enumerate(data['best'], start=1):
                for sol_idx, indiv in enumerate(solutions, start=1):
                    X = np.asarray(indiv, dtype=float).reshape(1, -1)
                    out = self.evalfunc(X)

                    # Estrai i valori di fitness (multi-obiettivo)
                    if isinstance(out, dict):
                        fit = out['F']
                    elif isinstance(out, (tuple, list)):
                        fit = out[0]
                    else:
                        fit = out

                    fit = np.atleast_1d(fit).flatten() 

                    record = {
                        'Run': run_idx,
                        'ID': sol_idx,
                        'Version': version,
                        'Variables': indiv
                    }

                    for i, f in enumerate(fit, start=1):
                        record[f'Fitness_{i}'] = float(f)

                    records.append(record)

        self.df = pd.DataFrame(records)
        self.fitness_cols = [col for col in self.df.columns if "Fitness" in col]

    def extract_fronts(self):
        '''
        Extract Pareto fronts per version and run from self.df.
        Returns a dict: {version: [array of fronts per run]}
        '''
        fronts = {}
        # Primo livello: Version
        for version, df_ver in self.df.groupby('Version'):
            # Lista dei fronti, uno per run
            run_fronts = []
            for _, df_run in df_ver.groupby('Run'):
                arr = df_run[self.fitness_cols].to_numpy(dtype=float)
                run_fronts.append(arr)
            fronts[version] = run_fronts

        return fronts
    
    def _combine_distrib(self):

        df1 = pd.DataFrame(self.d1)
        df1['Run'] = df1.index + 1
        df1['Version'] = self.v1

        df2 = pd.DataFrame(self.d2)
        df2['Run'] = df2.index + 1
        df2['Version'] = self.v2

        return pd.concat(
            [df1, df2], 
            ignore_index=True
        ).melt(
            id_vars=['Run', 'Version'], 
            var_name='Metric', 
            value_name='Value'
        )

    def plot_scatterplot(self, d1, d2, path: str = None, filename: str = None):

        self.d1 = d1
        self.d2 = d2
        df = self._combine_distrib()

        colors = {
            'Baseline': 'skyblue',
            'Population Dropout': 'salmon',
            'Individual Dropout': 'lightgreen',
            'Individual & Population Dropout': 'plum'
        }

        sns.set_theme(style="whitegrid")
        g = sns.FacetGrid(df, col="Metric", hue="Version", sharey=False,
                        col_wrap=2, height=4, palette=colors)

        g.map_dataframe(sns.scatterplot, x="Run", y="Value", alpha=0.8, s=60)

        for ax, metric in zip(g.axes.flat, df['Metric'].unique()):
            ax.set_title(rf"$\bf{{{metric}}}$")
            ax.xaxis.get_major_locator().set_params(integer=True)

            if self.symlog_thresh is not None:
                ax.set_yscale('symlog', linthresh=self.symlog_thresh)

            if self.y_lower_range and self.y_upper_range:
                vals = df[df['Metric'] == metric]['Value'].values
                ymin = min(vals.min(), self.y_lower_range[0])
                ymax = max(vals.max(), self.y_upper_range[1])
                ax.set_ylim(ymin, ymax)

        g.set_axis_labels("Run", "Value")

        fig = g.figure
        fig.suptitle(
            rf"$\bf{{{self.probname.capitalize()}}}$ — {self.v1} vs. {self.v2} — Final Re-evaluation",
            fontsize=18, x=0.5, y=0.99, ha='center'
        )
        fig.text(
            0.5, 0.945,
            f'Scatterplot of final evaluation metrics per run — {self.ngen} generations | {self.nvar} variables',
            ha='center', va='center', fontsize=12
        )

        handles, labels = g.axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels,
            loc='lower center', bbox_to_anchor=(0.5, -0.02),
            ncol=len(labels), fontsize=10, title_fontsize=11
        )

        fig.subplots_adjust(top=0.88, bottom=0.12)

        os.makedirs(path, exist_ok=True)
        fig.savefig(os.path.join(path, filename), dpi=300, bbox_inches='tight')

    def compute_grouped_stats(self):
        '''
        Group by ('Run', 'Version') and compute summary stats for one or more fitness columns.

        Parameters:
        - fitness_cols: list of column names to aggregate. If None, infer all columns starting with 'Fitness'.

        Returns:
        - DataFrame with one row per (Run, Version) and one column per statistic per fitness column.
        '''

        # Define functions for quartiles
        q1 = partial(np.nanpercentile, q=25)
        q3 = partial(np.nanpercentile, q=75)

        # Define statistics
        agg_funcs = {
            'Mean': 'mean',
            'Std': 'std',
            'Min': 'min',
            'Q1': lambda x: q1(x),
            'Median': 'median',
            'Q3': lambda x: q3(x),
            'Max': 'max',
            'Count': 'count'
        }

        # Build named aggregation dict
        named_aggs = {}
        for col in self.fitness_cols:
            for stat_name, func in agg_funcs.items():
                out_col = f"{col}_{stat_name}"
                named_aggs[out_col] = pd.NamedAgg(column=col, aggfunc=func)

        # Perform groupby with named aggregations
        grouped = (
            self.df
                .groupby(['Run', 'Version'])
                .agg(**named_aggs)
                .reset_index()
        )

        self.grouped = grouped

    def plot_boxplots(self, path: str, filename: str, obj: int = 0):
        runs = sorted(self.grouped['Run'].unique())
        n_best = int(self.grouped[f"{self.fitness_cols[obj]}_Count"].unique()[0])
        group_centers = np.arange(len(runs))

        colors = {
            'Baseline': 'skyblue',
            'Population Dropout': 'salmon',
            'Individual Dropout': 'lightgreen',
            'Individual & Population Dropout': 'plum'
        }
        box_width = 0.4

        use_broken = (
            isinstance(self.y_lower_range, (list, tuple)) and
            isinstance(self.y_upper_range, (list, tuple))
        )

        if use_broken:
            fig = plt.figure(figsize=(12, 8))
            fig.tight_layout()
            gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3], hspace=0.05)
            ax_top = fig.add_subplot(gs[0])
            ax_bot = fig.add_subplot(gs[1], sharex=ax_top)
            axes = (ax_top, ax_bot)
        else:
            fig, ax_main = plt.subplots(figsize=(12, 8))
            axes = (ax_main,)

        for ax in axes:
            for i, run in enumerate(runs):
                for j, version in enumerate([self.v1, self.v2]):
                    subset = self.grouped.query("Run == @run and Version == @version")
                    if subset.empty:
                        continue
                    row = subset.iloc[0]
                    stats = {
                        'label': f'Run {run} {version}',
                        'whislo':  row[f"{self.fitness_cols[obj]}_Min"],
                        'q1':      row[f"{self.fitness_cols[obj]}_Q1"],
                        'q3':      row[f"{self.fitness_cols[obj]}_Q3"],
                        'med':     row[f"{self.fitness_cols[obj]}_Median"],
                        'whishi':  row[f"{self.fitness_cols[obj]}_Max"],
                        'mean':    row[f"{self.fitness_cols[obj]}_Mean"],
                        'fliers':  np.array([])
                    }
                    pos = group_centers[i] + (j - 0.5) * box_width
                    ax.bxp([stats],
                        positions=[pos],
                        widths=box_width,
                        patch_artist=True,
                        boxprops=dict(facecolor=colors.get(version), alpha=0.7),
                        showfliers=False,
                        showmeans=True,
                        meanline=True,
                        meanprops={'color': 'black', 'linestyle': '--', 'linewidth': 1.5},
                        medianprops={'color': 'black', 'linewidth': 2})

        for ax in axes:
            # (a) target lines
            for tval in (self.targets or []):
                ax.axhline(y=float(tval), color='red', linestyle='--', linewidth=1)

            # (b) symlog (if not broken)
            if not use_broken and self.symlog_thresh is not None:
                ax.set_yscale('symlog', linthresh=float(self.symlog_thresh))

            '''# (c) y-axis limits
            ymin, ymax = ax.get_ylim()
            for tval in (self.targets or []):
                ymin = min(ymin, float(tval))
                ymax = max(ymax, float(tval))
            ax.set_ylim(ymin, ymax)'''

            # (d) grid
            ax.grid(axis='y', linestyle='--', alpha=0.7)

        bottom_ax = axes[-1]
        bottom_ax.set_xticks(group_centers)
        bottom_ax.set_xticklabels([str(r) if r in (1, len(runs)) or r % 5 == 0 else ''
                                for r in runs])
        for x in range(1, len(runs)):
            bottom_ax.axvline(x=x - 0.5, color='gray', linestyle=':', alpha=0.5)

        handles = [
            Patch(facecolor=colors[self.v1], alpha=0.7, label=self.v1),
            Patch(facecolor=colors[self.v2], alpha=0.7, label=self.v2),
            Line2D([0], [0], color='black', linestyle='--', label='Mean'),
            Line2D([0], [0], color='black', linewidth=2, label='Median')
        ]
        for tval in (self.targets or []):
            handles.append(Line2D([0], [0], color='red', lw=1, linestyle='--',
                                label=f'Target = {tval}'))
        bottom_ax.legend(handles=handles, loc='best')

        titlepos = bottom_ax.get_position().x0 + bottom_ax.get_position().width * 0.5
        fig.suptitle(
            rf"$\bf{{{self.probname.capitalize()}}}$ — {self.v1} vs. {self.v2} — Final Re-evaluation",
            fontsize=18, x=titlepos, y=0.995, ha='center'
        )
        fig.text(
            0.5, 0.945,
            f'Boxplot of the best {n_best} objective values per run - '
            f'{len(runs)} runs | {self.ngen} gens | {self.nvar} variables',
            ha='center', va='center', fontsize=12
        )
        bottom_ax.set_xlabel('Run', fontsize=12)
        bottom_ax.set_ylabel('Objective Value', fontsize=12)

        fig.savefig(os.path.join(path, filename), dpi=300)
        plt.close(fig)

    def count_head2head_wins(self, obj: int = 0) -> pd.DataFrame:
        '''
        For each run, compare Min objective value of Version 1 vs Version 2.
        Return DataFrame with columns: Run, MinV1, MinV2, Winner.
        '''
        df = self.grouped
        fc = self.fitness_cols[obj]
        base = df[df['Version'] == self.v1][['Run', f'{fc}_Min']].rename(columns={f'{fc}_Min': 'MinV1'})
        prop = df[df['Version'] == self.v2][['Run', f'{fc}_Min']].rename(columns={f'{fc}_Min': 'MinV2'})
        merged = pd.merge(base, prop, on='Run', how='outer')

        def decide(row):
            b, p = row['MinV1'], row['MinV2']
            if pd.isna(b) or pd.isna(p):
                return np.nan
            if p < b:
                return 'v2'
            if p > b:
                return 'v1'
            return 'Tie'

        merged['Winner'] = merged.apply(decide, axis=1)
        return merged

    def save_summary(self, path: str, filename: str, objs: tuple = (0,)):
        '''
        Count how many runs each version wins and save percentages as JSON.
        '''
        for obj in objs:
            if len(objs) > 1:
                filename += f'_obj_{obj}'
            head2head = self.count_head2head_wins(obj=obj)
            counts = head2head['Winner'].value_counts(dropna=True)
            total = counts.sum()

            mean_best_v1 = head2head['MinV1'].mean()
            mean_best_v2 = head2head['MinV2'].mean()
            std_best_v1 = head2head['MinV1'].std()
            std_best_v2 = head2head['MinV2'].std()

            summary = {
                'Total Runs': int(total),
                f'{self.v1} Wins': int(counts.get('v1', 0)),
                f'{self.v2} Wins': int(counts.get('v2', 0)),
                'Ties': int(counts.get('Tie', 0)),
                f'Mean best objective value across runs - {self.v1}': mean_best_v1,
                f'Std best objective value across runs - {self.v1}': std_best_v1,
                f'Mean best objective value across runs - {self.v2}': mean_best_v2,
                f'Std best objective value across runs - {self.v2}': std_best_v2
            }
            if total > 0:
                summary.update({
                    f'{self.v1} Wins (%)': counts.get('v1', 0) / total * 100,
                    f'{self.v2} Wins (%)': counts.get('v2', 0) / total * 100,
                    'Ties (%)': counts.get('Tie', 0) / total * 100
                })

            with open(os.path.join(path, filename), 'w') as f:
                    json.dump(summary, f, indent=4)

    def save_csv(self, path: str, filename: str):
        '''
        Save the raw DataFrame of all solutions to CSV.
        '''
        self.df.to_csv(os.path.join(path, filename), index=False)

    def save_grouped_csv(self, path: str, filename: str):
        '''
        Save the grouped-stats DataFrame to CSV.
        '''
        self.grouped.to_csv(os.path.join(path, filename), index=False)

    def save_markdown(self, path: str, filename: str):
        '''
        Save the raw DataFrame to a Markdown-formatted table.
        '''
        with open(os.path.join(path, filename), 'w') as f:
            f.write(self.df.to_markdown(tablefmt='grid'))

    def save_grouped_markdown(self, path: str, filename: str):
        '''
        Save the grouped stats to a Markdown-formatted table.
        '''
        with open(os.path.join(path, filename), 'w') as f:
            f.write(self.grouped.to_markdown(tablefmt='grid'))

    def show_grouped(self, path: str, filename: str, objs: tuple = (0,)):
        '''
        Alias for plotting side-by-side boxplots.
        '''
        for obj in objs:
            if len(objs) > 1:
                filename += f'_obj_{obj}'
            self.plot_boxplots(path, filename, obj)