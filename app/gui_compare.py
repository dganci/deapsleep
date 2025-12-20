# comparison GUI
import tkinter as tk
from tkinter import ttk, messagebox
import threading
from _loader import config_to_gui
from common import run, open_extra_params, Tooltip
from _fields import comp_FIELDS as FIELDS, comp_EXTRA_FIELDS as EXTRA_FIELDS
from utils.paths import RESULTS_DIR

extra_params = {"args": []}

def window():
    win = tk.Toplevel()
    win.title("Run Comparison")

    labels = [(f["label"], i) for i, f in enumerate(FIELDS)]
    tooltips = {f["label"]: f["tooltip"] for f in FIELDS}
    field_map = {f["label"]: f["key"] for f in FIELDS}
    entries = {}

    for text, row in labels:
        tk.Label(win, text=text).grid(row=row, column=0, sticky="w", padx=5, pady=5)
        entry = ttk.Entry(win, width=20)
        entry.grid(row=row, column=1, sticky="we", padx=5, pady=5)
        entries[text] = entry
        if tooltips.get(text):
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
                configtype='evalconfig'
            )
        )
    ).grid(row=1, column=2, padx=5, pady=5)

    ttk.Button(
        win,
        text="Extra parameters...",
        command=lambda: open_extra_params(
            win, {}, extra_params, 
            gui_type='compare',
            tooltips={}
        )
    ).grid(row=len(FIELDS), column=1, padx=5, pady=5)

    # Run comparison
    def start_run():
        values = {f["key"]: entries[f["label"]].get().strip() for f in FIELDS}

        required = ["problem_name", "problem_type", "version1", "version2"]
        missing = [k for k in required if not values[k]]
        if missing:
            messagebox.showwarning("Warning", f"Missing required fields: {', '.join(missing)}")
            return

        progress_win = tk.Toplevel()
        progress_win.title("Comparison Progress")
        ttk.Label(progress_win, text=f"Comparing {values['version1']} vs {values['version2']}...").pack(padx=10, pady=10)
        progress_var = tk.DoubleVar(value=0)
        ttk.Progressbar(progress_win, variable=progress_var, maximum=100, length=300).pack(padx=10, pady=10)
        time_label = ttk.Label(progress_win, text="Estimated time: --:--:--")
        time_label.pack(padx=10, pady=5)

        args = [
            "--config", f"{values['problem_type']}.{values['problem_name']}",
            "-i",
            f"--version1='{values['version1']}'",
            f"--version2='{values['version2']}'",
            #f"--dirname={RESULTS_DIR}"
        ]
        args.extend(extra_params["args"])

        threading.Thread(
            target=run,
            args=("compare", args, progress_var, progress_win, time_label),
            daemon=True
        ).start()

    ttk.Button(win, text="Run comparison", command=start_run).grid(row=len(FIELDS)+1, column=0, columnspan=2, pady=10)