# plotting GUI
import tkinter as tk
from tkinter import ttk, messagebox
import threading
from common import run, Tooltip
from _fields import plot_FIELDS as FIELDS
from utils.paths import RESULTS_DIR

extra_params = {"args": []}

def window():
    '''
    GUI window to plot results.
    '''
    win = tk.Toplevel()
    win.title("Plot results")

    entries = {}

    for i, field in enumerate(FIELDS):
        tk.Label(win, text=field["label"]).grid(row=i, column=0, sticky="w", padx=5, pady=5)
        entry = ttk.Entry(win, width=20)
        entry.grid(row=i, column=1, sticky='we', padx=5, pady=5)
        entries[field["key"]] = entry

    if field.get("tooltip"):
        Tooltip(entry, field["tooltip"])

    def start_run():
        values = {k: e.get().strip() for k, e in entries.items()}

        if not all(values.values()):
            messagebox.showwarning("Warning", "Please fill in all required fields.")
            return

        progress_win = tk.Toplevel()
        progress_win.title("Plot Progress")
        ttk.Label(progress_win, text=f"Plotting {values.get('problem', '?')} ...").pack(padx=10, pady=10)
        progress_var = tk.DoubleVar(value=0)
        ttk.Progressbar(progress_win, variable=progress_var, maximum=100, length=300).pack(padx=10, pady=10)
        time_label = ttk.Label(progress_win, text="Estimated time: --:--:--")
        time_label.pack(padx=10, pady=5)

        args = [f"--{k}={v}" for k, v in values.items() if v]
        #args.append(f"--dirname={RESULTS_DIR}")
        args.extend(extra_params["args"])

        threading.Thread(
            target=run,
            args=("plot", args, progress_var, progress_win, time_label),
            daemon=True
        ).start()

    ttk.Button(win, text="Run plot", command=start_run).grid(row=len(FIELDS)+1, column=0, columnspan=2, pady=10)