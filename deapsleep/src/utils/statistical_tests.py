import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kstest, wilcoxon, shapiro, normaltest, anderson, probplot, ttest_rel

plt.switch_backend("Agg")

def get_distrib(loglist, stat, target):
    '''
    Extract a list of fitness values at 'target' index from each logbook,
    using the specified statistic (e.g., 'mean', 'min', etc.).
    '''
    return [logbook.select(stat)[target].item() for logbook in loglist]


def normtest(path, distrib, metric='', **kwargs):
    '''
    Perform normality tests on 'distrib' and save results and plots.
    Accepted kwargs:
      - shapiro=True      → Shapiro-Wilk
      - dagostino=True    → D'Agostino-Pearson
      - anderson=True     → Anderson-Darling
      - kstest=True       → Kolmogorov-Smirnov
      - probplot=True     → QQ plot saved as 'probability_plot.png'
      - histplot=True     → Histogram + KDE saved as 'histogram_plot.png'
    Results of tests are written to 'normal_test.txt' in 'path'.
    '''
    os.makedirs(path, exist_ok=True)
    lines = []

    if kwargs.get("shapiro", False):
        W, p = shapiro(distrib)
        lines.append(f"--- Shapiro-Wilk ---\nW = {W}\np = {p}\n")

    if kwargs.get("dagostino", False):
        K, p = normaltest(distrib)
        lines.append(f"--- D'Agostino-Pearson ---\nK² = {K}\np = {p}\n")

    if kwargs.get("anderson", False):
        stats = anderson(distrib, dist="norm")
        lines.append(
            f"--- Anderson-Darling ---\nA² = {stats.statistic}\n"
            f"Crit. values = {stats.critical_values}\n"
            f"Significance levels = {stats.significance_level}\n"
        )

    if kwargs.get("kstest", False):
        mu, sigma = np.mean(distrib), np.std(distrib, ddof=1)
        D, p = kstest(distrib, "norm", args=(mu, sigma))
        lines.append(f"--- Kolmogorov-Smirnov ---\nD = {D}\np = {p}\n")

    # Write test results
    with open(os.path.join(path, f"normal_test{metric}.txt"), "w") as f:
        f.write("\n".join(lines))

    # QQ plot
    if kwargs.get("probplot", False):
        plt.figure()
        probplot(distrib, plot=plt)
        plt.savefig(os.path.join(path, "probability_plot.png"))
        plt.close()

    # Histogram + KDE
    if kwargs.get("histplot", False):
        plt.figure()
        plt.hist(distrib, bins=20, density=True, alpha=0.5)
        sns.kdeplot(distrib, label="KDE")
        plt.legend()
        plt.savefig(os.path.join(path, "histogram_plot.png"))
        plt.close()

    return p

def perform_test(path, filename, normtest_p1, normtest_p2, distrib1, distrib2, metric='', corr=None):

    os.makedirs(path, exist_ok=True)

    if normtest_p1 > 0.05 and normtest_p2 > 0.05: # then, both data are normally distributed
        func = ttest_rel
        name = 'paired t-test'  
    else:
        func = wilcoxon
        name = 'wilcoxon signed-rank'

    M, p = func(distrib1, distrib2)

    if corr is not None: # then, apply Bonferroni correction
        if not isinstance(corr, int) or corr <= 0:
            raise ValueError("Invalid value for the Bonferroni correction factor: must be a positive integer.")
        p *= corr
        if p > 1.0:
            p = 1.0

    text = f"--- {name.capitalize()} ---\nMetric = {M}\np-value = {p}\n"
    with open(os.path.join(path, f"{filename}_{name}{metric}.txt"), "w") as f:
        f.write(text)