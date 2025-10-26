# optimization GUI
import tkinter as tk
from tkinter import ttk, messagebox
import threading
from utils import run, open_extra_params, Tooltip
from extra_params import opt_params, opt_tooltips

extra_params = {"args": []}

def window():
    win = tk.Toplevel()
    win.title("Run optimization")

    labels = [
        ("Problem name*", 0),
        ("Problem type (single or multi)*", 1),
        ('Experiment name (e.g. "baseline")*', 2),
        ("Number of runs*", 3),
        ("Number of generations*", 4),
        ("Number of variables*", 5),
        ("Individual dropout rate (optional):", 6),
        ("Population dropout rate (optional):", 7),
    ]
    entries = {}

    for text, row in labels:
        tk.Label(win, text=text).grid(row=row, column=0, sticky="w", padx=5, pady=5)
        entry = ttk.Entry(win, width=20)
        entry.grid(row=row, column=1, sticky="we", padx=5, pady=5)
        entries[text] = entry
    
    Tooltip(entries['Experiment name (e.g. "baseline")*'], 
        "Accepted: any version containing 'base', 'idrop',\n'indd', 'pdrop', 'popd' (case insensitive).")

    Tooltip(entries['Problem name*'], 
        "Accepted: 'ackley', 'griewank', 'rastrigin', 'rosenbrock', 'schwefel',\n'sphere', 'zakharov', 'zdt1', 'zdt2', 'zdt3', 'zdt4', 'zdt6'.")

    Tooltip(entries['Problem type (single or multi)*'], 
        "Accepted: 'single', 'multi'.")
        
    Tooltip(entries["Individual dropout rate (optional):"],
        "Accepted: float in [0, 1] (e.g., 0.3 for 30% dropout).")
        
    Tooltip(entries["Population dropout rate (optional):"],
        "Accepted: float in [0, 1] (e.g., 0.3 for 30% dropout).")

    ttk.Button(
        win,
        text="Extra parameters...",
        command=lambda: open_extra_params(win, opt_params, extra_params, gui_type='optimize', tooltips=opt_tooltips)
    ).grid(row=8, column=1, padx=5, pady=5)

    def start_run():
        probname = entries["Problem name*"].get()
        probtype = entries["Problem type (single or multi)*"].get()
        version = entries['Experiment name (e.g. "baseline")*'].get()
        n_runs = entries["Number of runs*"].get()
        n_gen = entries["Number of generations*"].get()
        n_var = entries["Number of variables*"].get()
        indD_rate = entries["Individual dropout rate (optional):"].get()
        popD_rate = entries["Population dropout rate (optional):"].get()

        if not all([probname, probtype, version, n_runs, n_gen, n_var]):
            messagebox.showwarning("Warning", "Please fill in all required fields.")
            return

        progress_win = tk.Toplevel()
        progress_win.title("Optimization Progress")
        ttk.Label(progress_win, text="Running optimization...").pack(padx=10, pady=10)
        progress_var = tk.DoubleVar(value=0)
        ttk.Progressbar(progress_win, variable=progress_var, maximum=100, length=300).pack(padx=10, pady=10)
        time_label = ttk.Label(progress_win, text="Estimated time: --:--:--")
        time_label.pack(padx=10, pady=5)

        args = [
            "--config", f"{probtype}.{probname}",
            "-i",
            f"--version='{version}'",
            f"--n_runs={n_runs}",
            f"--ngen={n_gen}",
            f"--n_var={n_var}",
            f"--dirname=/app/results",
        ]
        if indD_rate: args.append(f"--indD_rate={indD_rate}")
        if popD_rate: args.append(f"--popD_rate={popD_rate}")
        args.extend(extra_params["args"])

        threading.Thread(
            target=run,
            args=("optimize", args, progress_var, progress_win, time_label),
            daemon=True
        ).start()

    ttk.Button(win, text="Run optimization", command=start_run).grid(row=8, column=0, padx=5, pady=5)
