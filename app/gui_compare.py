# comparison GUI
import tkinter as tk
from tkinter import ttk, messagebox
import threading
from utils import run, open_extra_params, Tooltip
from extra_params import comp_params, comp_tooltips

extra_params = {"args": []}

def window():
    win = tk.Toplevel()
    win.title("Run Comparison")

    # Input fields
    tk.Label(win, text="Problem name*").grid(row=0, column=0, sticky="w", padx=5, pady=5)
    probname_entry = ttk.Entry(win, width=15)
    probname_entry.grid(row=0, column=1, sticky='we', padx=5, pady=5)

    tk.Label(win, text="Problem type (single or multi)*").grid(row=1, column=0, sticky="w", padx=5, pady=5)
    probtype_entry = ttk.Entry(win, width=15)
    probtype_entry.grid(row=1, column=1, sticky='we', padx=5, pady=5)

    tk.Label(win, text='1st experiment name (e.g., "baseline")*').grid(row=2, column=0, sticky="w", padx=5, pady=5)
    version1_entry = ttk.Entry(win, width=15)
    version1_entry.grid(row=2, column=1, sticky='we', padx=5, pady=5)

    tk.Label(win, text='2nd experiment name (e.g., "IDrop")*').grid(row=3, column=0, sticky="w", padx=5, pady=5)
    version2_entry = ttk.Entry(win, width=15)
    version2_entry.grid(row=3, column=1, sticky='we', padx=5, pady=5)

    Tooltip(version1_entry, 
        "Accepted: any version containing 'base', 'idrop',\n'indd', 'pdrop', 'popd' (case insensitive).")

    Tooltip(version2_entry,
        "Accepted: any version containing 'base', 'idrop',\n'indd', 'pdrop', 'popd' (case insensitive).")

    Tooltip(probname_entry, 
        "Accepted: 'ackley', 'griewank', 'rastrigin', 'rosenbrock', 'schwefel',\n'sphere', 'zakharov', 'zdt1', 'zdt2', 'zdt3', 'zdt4', 'zdt6'.")

    Tooltip(probtype_entry, 
        "Accepted: 'single', 'multi'.")
    
    ttk.Button(
        win,
        text="Extra parameters...",
        command=lambda: open_extra_params(win, comp_params, extra_params, gui_type='compare', tooltips=comp_tooltips)
    ).grid(row=4, column=1, padx=5, pady=5)

    # Run comparison
    def start_run():
        probname = probname_entry.get().strip()
        probtype = probtype_entry.get().strip()
        version1 = version1_entry.get().strip()
        version2 = version2_entry.get().strip()

        if not all([probname, probtype, version1, version2]):
            messagebox.showwarning("Warning", "Please fill in all required fields.")
            return

        progress_win = tk.Toplevel()
        progress_win.title("Comparison Progress")
        ttk.Label(progress_win, text=f"Comparing {version1} vs {version2} ...").pack(padx=10, pady=10)
        progress_var = tk.DoubleVar(value=0)
        ttk.Progressbar(progress_win, variable=progress_var, maximum=100, length=300).pack(padx=10, pady=10)
        time_label = ttk.Label(progress_win, text="Estimated time: --:--:--")
        time_label.pack(padx=10, pady=5)

        args = [
            "--config", f"{probtype}.{probname}",
            "-i",
            f"--version1='{version1}'",
            f"--version2='{version2}'",
            f"--dirname=/app/results"
        ]
        args.extend(extra_params["args"])

        threading.Thread(
            target=run,
            args=("compare", args, progress_var, progress_win, time_label),
            daemon=True
        ).start()
    
    ttk.Button(win, text="Run comparison", command=start_run).grid(row=5, column=0, columnspan=2, pady=10)
