# utils for the GUI
import traceback
import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import re
import os
from _tooltip import Tooltip

def show_traceback(title, text):
    win = tk.Toplevel()
    win.title(title)
    win.geometry("900x550")

    win.rowconfigure(0, weight=1)
    win.columnconfigure(0, weight=1)

    frame = ttk.Frame(win)
    frame.grid(row=0, column=0, sticky="nsew")

    frame.rowconfigure(0, weight=1)
    frame.columnconfigure(0, weight=1)

    yscroll = ttk.Scrollbar(frame, orient="vertical")
    xscroll = ttk.Scrollbar(frame, orient="horizontal")

    txt = tk.Text(
        frame,
        wrap="none",
        font=("Courier New", 10),
        yscrollcommand=yscroll.set,
        xscrollcommand=xscroll.set
    )

    yscroll.config(command=txt.yview)
    xscroll.config(command=txt.xview)

    txt.grid(row=0, column=0, sticky="nsew")
    yscroll.grid(row=0, column=1, sticky="ns")
    xscroll.grid(row=1, column=0, sticky="ew")

    txt.insert("1.0", text)
    txt.config(state="disabled")

    btn_frame = ttk.Frame(win)
    btn_frame.grid(row=1, column=0, sticky="ew", pady=5)

    ttk.Button(btn_frame, text="Close", command=win.destroy).pack(pady=2)

def run(
        # module to run
        module_name, 
        # parameters
        args, 
        # progress bar
        progress_var, 
        progress_win, 
        time_label):
    '''
    Executes a deapsleep module as a subprocess, updating the progress bar and ETA label.
    '''
    out = []
    try:
        cmd = ["python3", "-m", f"deapsleep.main.{module_name}", *args]
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=os.environ
        )

        # Monitor progress and ETA (retrieved from stdout)
        for line in process.stdout:
            line = line.strip()
            out.append(line)
            match = re.search(r'(\d+)%', line)
            if match:
                progress_var.set(int(match.group(1)))

            match_eta = re.search(r'<(\d+):(\d+)(?::(\d+))?', line)
            if match_eta:
                hrs, mins, secs = map(int, match_eta.groups(default='0'))
                time_label.config(text=f"Estimated time: {hrs:02d}:{mins:02d}:{secs:02d}")
                progress_win.update_idletasks()

        process.wait()

        names = {
            'optimize': 'Optimization',
            'compare': 'Comparison',
            'plot': 'Plotting'
        }

        if process.returncode == 0:
            messagebox.showinfo("Done", f"{names[module_name]} finished successfully!")
        else:
            out = "\n".join(out[-300:])
            show_traceback(
                f"{names[module_name]} failed",
                out
            )
    except Exception as e:
        tb = traceback.format_exc()
        messagebox.showerror(
            "Error", 
            f"{names.get(module_name, module_name)} failed:\n{e}\n\nTraceback:\n{tb}"
        )
    finally:
        progress_win.destroy()

_last_extra_win_geometry = {
    'optimize': "620x400",
    'compare': "620x215",
    'plot': "620x120",
}

def open_extra_params(parent, fixed_params, extra_params, gui_type=None, tooltips=None):
    """
    Opens a window to input extra parameters.
    - Extra parameters from config: fixed name, editable value (readonly name).
    - User-added parameters: editable name/value, removable.
    """
    if gui_type is None:
        raise ValueError("Specify gui_type as 'optimize', 'compare', or 'plot'.")

    global _last_extra_win_geometry
    try: parent.grab_release()
    except tk.TclError: pass

    extra_win = tk.Toplevel(parent)
    extra_win.title("Extra Parameters")

    geom = _last_extra_win_geometry[gui_type]
    min_w, min_h = map(int, geom.split('x'))
    extra_win.minsize(min_w, min_h)
    extra_win.transient(parent)
    extra_win.update_idletasks()
    extra_win.geometry(_last_extra_win_geometry[gui_type])

    # Scrollable frame
    canvas = tk.Canvas(extra_win)
    scrollbar = ttk.Scrollbar(extra_win, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)
    scroll_frame = ttk.Frame(canvas)
    scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0,0), window=scroll_frame, anchor="nw")
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    fixed_entries = {}
    row_idx = 1

    # Fixed parameters
    if fixed_params:
        for label, (default, _) in fixed_params.items():
            tk.Label(scroll_frame, text=label).grid(row=row_idx, column=0, sticky="w", padx=5, pady=3)
            entry = ttk.Entry(scroll_frame, width=25)
            entry.grid(row=row_idx, column=1, padx=5, pady=3)
            entry.insert(0, default)
            fixed_entries[label] = entry
            if tooltips and label in tooltips:
                Tooltip(entry, tooltips[label], delay=600)
            row_idx += 1


    param_entries = []

    def add_param_row(name_default="", value_default="", fixed_name=False):
        nonlocal row_idx
        name_entry = ttk.Entry(scroll_frame, width=20)
        if fixed_name:
            name_entry.insert(0, name_default)
            name_entry.config(state="readonly")
        else:
            name_entry.insert(0, name_default)
        value_entry = ttk.Entry(scroll_frame, width=20)
        value_entry.insert(0, value_default)
        remove_btn = ttk.Button(scroll_frame, text="✕", width=2, command=lambda: remove_row((name_entry, value_entry, remove_btn)))
        name_entry.grid(row=row_idx, column=0, padx=5, pady=3, sticky="we")
        value_entry.grid(row=row_idx, column=1, padx=5, pady=3, sticky="we")
        remove_btn.grid(row=row_idx, column=2, padx=5, pady=3)
        param_entries.append((name_entry, value_entry, remove_btn))
        row_idx += 1

    def remove_row(entry_tuple):
        for w in entry_tuple:
            w.destroy()
        param_entries.remove(entry_tuple)

    # ❌ Non aggiungere nulla all'apertura se extra_params è vuoto
    if extra_params.get("args"):
        for arg in extra_params["args"]:
            if "=" in arg:
                name, value = arg.lstrip("-").split("=", 1)
            else:
                name, value = arg.lstrip("-"), ""
            add_param_row(name, value, fixed_name=True)

    ttk.Button(scroll_frame, text="Add extra parameter", command=lambda: add_param_row()).grid(row=999, column=0, pady=5)

    # Buttons
    def save_and_close():
        extras = []

        # Fixed parameters
        for label, entry in fixed_entries.items():
            value = entry.get().strip()
            short_name = fixed_params[label][1]
            fixed_params[label] = (value, short_name)
            if value:
                extras.append(f"--{short_name}={value}")

        # Extra parameters (user-added + config-added)
        for name_entry, value_entry, _ in param_entries:
            name = name_entry.get().strip()
            value = value_entry.get().strip()
            if name:
                extras.append(f"--{name}={value}" if value else f"--{name}")

        extra_params["args"].clear()
        extra_params["args"].extend(extras)

        extra_win.grab_release()
        extra_win.destroy()
        try: parent.grab_set()
        except tk.TclError: pass

    def cancel_and_close():
        extra_win.grab_release()
        extra_win.destroy()
        try: parent.grab_set()
        except tk.TclError: pass

    button_frame = ttk.Frame(extra_win)
    button_frame.pack(side="bottom", fill="x", padx=5, pady=10)
    ttk.Button(button_frame, text="Back (Save)", command=save_and_close).pack(fill='x', pady=2)
    ttk.Button(button_frame, text="Cancel", command=cancel_and_close).pack(fill='x', pady=2)
    extra_win.protocol("WM_DELETE_WINDOW", cancel_and_close)

    extra_win.update_idletasks()
    extra_win.grab_set()
    extra_win.lift()
    extra_win.focus_force()

