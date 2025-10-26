# plotting GUI
import tkinter as tk
from tkinter import ttk, messagebox
import threading
from utils import run, open_extra_params, Tooltip
from extra_params import plot_params, plot_tooltips

extra_params = {"args": []}

def window():
    '''
    GUI window to plot results.
    '''
    win = tk.Toplevel()
    win.title("Plot results")

    # Input fields
    tk.Label(win, text="Problem name*:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
    problem_entry = ttk.Entry(win, width=15)
    problem_entry.grid(row=0, column=1, sticky='we', padx=5, pady=5)

    tk.Label(win, text='Experiment name (e.g. "baseline")*').grid(row=1, column=0, sticky="w", padx=5, pady=5)
    version_entry = ttk.Entry(win, width=15)
    version_entry.grid(row=1, column=1, sticky='we', padx=5, pady=5)

    Tooltip(problem_entry, 
        "Accepted: 'ackley', 'griewank', 'rastrigin', 'rosenbrock', 'schwefel',\n'sphere', 'zakharov', 'zdt1', 'zdt2', 'zdt3', 'zdt4', 'zdt6'.")
    Tooltip(version_entry, 
        "Accepted: any version containing 'base', 'idrop',\n'indd', 'pdrop', 'popd' (case insensitive).")

    ttk.Button(
        win,
        text="Extra parameters...",
        command=lambda: open_extra_params(win, plot_params, extra_params, gui_type='plot', tooltips=plot_tooltips)
    ).grid(row=2, column=1, padx=5, pady=5)

    # Run plotting
    def start_run():
        '''
        Starts the plotting process in a separate thread.
        '''
        problem = problem_entry.get().strip()
        version = version_entry.get().strip()

        if not problem:
            messagebox.showwarning("Warning", "Please fill in the problem field.")
            return

        progress_win = tk.Toplevel()
        progress_win.title("Plot Progress")
        ttk.Label(progress_win, text=f"Plotting {problem} ...").pack(padx=10, pady=10)
        progress_var = tk.DoubleVar(value=0)
        ttk.Progressbar(progress_win, variable=progress_var, maximum=100, length=300).pack(padx=10, pady=10)
        time_label = ttk.Label(progress_win, text="Estimated time: --:--:--")
        time_label.pack(padx=10, pady=5)

        args = [
            "--problem", problem,
            "--version", version,
            f"--dirname=/app/results"
        ]
        args.extend(extra_params["args"])

        threading.Thread(
            target=run,
            args=("plot", args, progress_var, progress_win, time_label),
            daemon=True
        ).start()
    
    ttk.Button(win, text="Run plot", command=start_run).grid(row=3, column=0, columnspan=2, pady=10)
