# utils for the GUI
import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import re

class Tooltip:
    '''A simple tooltip for Tkinter widgets'''
    def __init__(self, widget, text, delay=600):
        self.widget = widget
        self.text = text
        self.tipwindow = None
        self.after_id = None
        self.delay = delay
        widget.bind("<Enter>", self.schedule)
        widget.bind("<Leave>", self.hide)

    def schedule(self, event=None):
        '''
        Program the display of the tooltip after a delay
        '''
        self.unschedule()
        self.after_id = self.widget.after(self.delay, self.show)

    def unschedule(self):
        '''
        Cancel the scheduled display of the tooltip
        '''
        if self.after_id:
            self.widget.after_cancel(self.after_id)
            self.after_id = None

    def show(self, event=None):
        if self.tipwindow or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + 20
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)  # no title bar
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw, text=self.text, justify='left',
            background="#ffffc0", relief='solid', borderwidth=1,
            font=("arial", "9", "normal"),
            padx=6, pady=4
        )
        label.pack(ipadx=1)

    def hide(self, event=None):
        self.unschedule()
        if self.tipwindow:
            self.tipwindow.destroy()
            self.tipwindow = None

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

    try:
        cmd = ["python3", "-m", f"deapsleep.main.{module_name}", *args]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        # Monitor progress and ETA (retrieved from stdout)
        for line in process.stdout:
            line = line.strip()
            print(line) 
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
            messagebox.showerror("Error", f"{names[module_name]} failed. Check logs.")

    except Exception as e:
        messagebox.showerror("Error", f"{names[module_name]} failed:\n{e}")
    finally:
        progress_win.destroy()

_last_extra_win_geometry = {
    'optimize': "620x400",
    'compare': "620x215",
    'plot': "620x120",
}

def open_extra_params(
        parent, 
        fixed_params, 
        extra_params,
        gui_type=None,
        tooltips=None
    ):
    '''
    Opens a window to input extra parameters for the deapsleep modules.
    '''

    if gui_type is None:
        raise ValueError("Specify gui_type as 'optimize', 'compare', or 'plot'.")
    
    global _last_extra_win_geometry

    try:
        parent.grab_release()
    except tk.TclError:
        pass 

    extra_win = tk.Toplevel(parent)
    extra_win.title("Extra Parameters")
    
    geom = _last_extra_win_geometry[gui_type]
    min_w, min_h = map(int, geom.split('x'))
    extra_win.minsize(min_w, min_h)
    
    extra_win.transient(parent)
    
    extra_win.update_idletasks()
    extra_win.geometry(_last_extra_win_geometry[gui_type])

    fixed_entries = {}
    row_idx = 1

    # Fixed parameters
    for label, (default, _) in fixed_params.items():
        tk.Label(extra_win, text=label).grid(row=row_idx, column=0, sticky="w", padx=5, pady=3)
        entry = ttk.Entry(extra_win, width=25)
        entry.grid(row=row_idx, column=1, padx=5, pady=3)
        entry.insert(0, default)
        fixed_entries[label] = entry
        if tooltips and label in tooltips:
            Tooltip(entry, tooltips[label], delay=600)
        row_idx += 1

    param_entries = []
    initial_extra_rows = 0

    def adjust_window_height(delta):
        '''
        Adjust the height of the extra parameters window.
        '''
        curr = extra_win.geometry()
        geom = curr.split("+")[0]
        w, h = map(int, geom.split("x"))
        new_h = max(min_h, h + delta)
        pos = "+".join(curr.split("+")[1:])
        extra_win.geometry(f"{w}x{new_h}+{pos}")

    def add_param_row(name_default="", value_default=""):
        '''
        Add a new row for an extra parameter.
        '''
        nonlocal row_idx
        name_entry = ttk.Entry(extra_win, width=20)
        value_entry = ttk.Entry(extra_win, width=20)
        remove_btn = ttk.Button(extra_win, text="✕", width=2)

        name_entry.grid(row=row_idx, column=0, padx=5, pady=5, sticky="we")
        value_entry.grid(row=row_idx, column=1, padx=5, pady=5, sticky="we")
        remove_btn.grid(row=row_idx, column=2, padx=5, pady=5)

        name_entry.insert(0, name_default)
        value_entry.insert(0, value_default)

        if tooltips and name_default in tooltips:
            Tooltip(name_entry, tooltips[name_default])

        entry_tuple = (name_entry, value_entry, remove_btn)
        param_entries.append(entry_tuple)

        def remove_row():
            for w in entry_tuple:
                w.destroy()
            param_entries.remove(entry_tuple)
            adjust_window_height(-40)
        
        remove_btn.config(command=remove_row)
        row_idx += 1
        adjust_window_height(+40)

    # Pre-fill existing extra parameters
    for arg in extra_params.get("args", []):
        if "=" in arg:
            name, value = arg.lstrip("-").split("=", 1)
        else:
            name, value = arg.lstrip("-"), ""
        if not any(value_key == name for _, (_, value_key) in fixed_params.items()):
            add_param_row(name, value)
            initial_extra_rows += 1

    ttk.Button(extra_win, text="Add extra parameter", command=add_param_row).grid(
        row=999, column=0, pady=5
    )

    def save_and_close():
        '''
        Save the extra parameters and close the window.
        '''
        extras = []

        # Update fixed parameters
        for label, entry in fixed_entries.items():
            value = entry.get().strip()
            short_name = fixed_params[label][1]
            fixed_params[label] = (value, short_name)
            if value:
                extras.append(f"--{short_name}={value}")

        final_extra_rows = len(param_entries)

        # Check extra parameters
        for name_entry, value_entry, _ in param_entries:
            name = name_entry.get().strip()
            value = value_entry.get().strip()
            if name:
                extras.append(f"--{name}={value}" if value else f"--{name}")
        
        if final_extra_rows > initial_extra_rows:
            global _last_extra_win_geometry
            current_geom = extra_win.geometry().split("+")[0]
            _last_extra_win_geometry[gui_type] = current_geom

        '''global _last_extra_win_geometry
        current_geom = extra_win.geometry().split("+")[0]
        old_geom = _last_extra_win_geometry[gui_type]
        
        if current_geom != old_geom:
            old_w, old_h = map(int, old_geom.split("x"))
            w, h = map(int, current_geom.split("x"))
            if h > old_h:
                _last_extra_win_geometry[gui_type] = current_geom'''

        extra_params["args"] = extras

        extra_win.grab_release()
        extra_win.destroy()

        try:
            parent.grab_set()
        except tk.TclError:
            pass
    
    def cancel_and_close():
        '''
        Discard changes and close the extra parameters window.
        '''
        extra_win.grab_release()
        extra_win.destroy()

        try:
            parent.grab_set()
        except tk.TclError:
            pass

    ttk.Frame(extra_win).grid(row=998, column=0)
    button_frame = ttk.Frame(extra_win)
    button_frame.grid(row=999, column=1, padx=5, pady=10, sticky="we")
    ttk.Button(button_frame, text="Back (Save)", command=save_and_close).pack(fill='x', pady=2)
    ttk.Button(button_frame, text="Cancel", command=cancel_and_close).pack(fill='x', pady=2)
    
    extra_win.protocol("WM_DELETE_WINDOW", cancel_and_close)

    extra_win.update_idletasks()
    extra_win.grab_set()
    extra_win.lift()
    extra_win.focus_force()