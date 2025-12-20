# optimization GUI
import tkinter as tk
from tkinter import ttk, messagebox
import threading
from common import run, open_extra_params, Tooltip
from _loader import config_to_gui
from _fields import opt_FIELDS as FIELDS, opt_EXTRA_FIELDS as EXTRA_FIELDS
from utils.paths import RESULTS_DIR

extra_params = {"args": []}

def window():
    win = tk.Toplevel()
    win.title("Run optimization")

    # First window
    labels = [
        (f["label"], i) 
        for i, f in enumerate(FIELDS)
    ]
    field_map = {
        f["label"]: f["key"] 
        for f in FIELDS
    }
    tooltips = {
        f["label"]: f["tooltip"] 
        for f in FIELDS
    }

    entries = {}

    for text, row in labels:
        tk.Label(win, text=text).grid(row=row, column=0, sticky="w", padx=5, pady=5)
        entry = ttk.Entry(win, width=20)
        entry.grid(row=row, column=1, sticky="we", padx=5, pady=5)
        entries[text] = entry
        if tooltips[text]:
            Tooltip(entries[text], tooltips[text])

    ttk.Button(
        win,
        text="Load basic configuration",
        command=lambda: (
            config_to_gui(
                entries["Problem name*"].get().strip(),
                entries["Problem type (single or multi)*"].get().strip(),
                entries,
                extra_params,
                field_map,
                EXTRA_FIELDS,
                configtype='baseconfig'
            )
        )
    ).grid(row=1, column=2, padx=5, pady=5)

    ttk.Button(
        win,
        text="Extra parameters...",
        command=lambda: open_extra_params(
            win, {}, extra_params, 
            gui_type='optimize',
            tooltips={}
        )
    ).grid(row=8, column=1, padx=5, pady=5)

    def start_run():
        values = {f["key"]: entries[f["label"]].get() for f in FIELDS}

        required = ["problem_name", "problem_type", "version", "n_runs", "ngen", "n_var"]
        missing = [k for k in required if not values[k]]
        if missing:
            messagebox.showwarning("Warning", f"Missing required fields: {', '.join(missing)}")
            return

        progress_win = tk.Toplevel()
        progress_win.title("Optimization Progress")
        ttk.Label(progress_win, text="Running optimization...").pack(padx=10, pady=10)
        progress_var = tk.DoubleVar(value=0)
        ttk.Progressbar(progress_win, variable=progress_var, maximum=100, length=300).pack(padx=10, pady=10)
        time_label = ttk.Label(progress_win, text="Estimated time: --:--:--")
        time_label.pack(padx=10, pady=5)

        args = [
            "--config", f"{values['problem_type']}.{values['problem_name']}",
            "-i",
            f"--version='{values['version']}'",
            f"--n_runs={values['n_runs']}",
            f"--ngen={values['ngen']}",
            f"--n_var={values['n_var']}",
            #f"--dirname={RESULTS_DIR}"
        ]
        if values.get("indD_rate"): args.append(f"--indD_rate={values['indD_rate']}")
        if values.get("popD_rate"): args.append(f"--popD_rate={values['popD_rate']}")
        args.extend(extra_params["args"])

        threading.Thread(
            target=run,
            args=("optimize", args, progress_var, progress_win, time_label),
            daemon=True
        ).start()

    ttk.Button(win, text="Run optimization", command=start_run).grid(row=8, column=0, padx=5, pady=5)
