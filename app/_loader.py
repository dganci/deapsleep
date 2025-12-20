# config_loader.py
import tkinter as tk
from tkinter import messagebox
from utils.utils import load_internal

EXCLUDED = {"ind_type", "pop_type", "statparams", "weights", "ngen"}

def config_to_gui(probname, probtype, entries, extra_params, field_map, extra_fields, configtype):

    if not probname and not probtype:
        messagebox.showwarning("Warning", "Please enter a problem name or type first.")
        return

    try:
        params = load_internal(f'{probtype}.{probname}', configtype=configtype)
        base = params.get("base", {})
        evol = base.get("evolparams", {})
        init = base.get("initparams", {})
        guivals = {
            k: v for k, v in base.items()
            if not isinstance(v, dict)
        }

        additional = params.get('additional', {})
        extras = {**init, **evol, **additional}

        extra_fields[:] = [{"key": k} for k in additional.keys()]

        if "ngen" in evol:
            guivals["ngen"] = evol["ngen"]
            del extras["ngen"]

        for key in ["n_runs", "n_var", "version", "aggregation_op", "instance", "seed"]:
            if key in params:
                guivals[key] = params[key]

    except FileNotFoundError as e:
        messagebox.showerror("Error", f"No configuration found for problem '{probname}'")
        return
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load config:\n{e}")
        return
    
    for label, key in field_map.items():
        if key in guivals and label in entries:
            entry = entries[label]
            entry.delete(0, tk.END)
            entry.insert(0, str(guivals[key]))
        
    updated_extras = []
    known_keys = {ef["key"] for ef in extra_fields}

    for ef in extra_fields:
        k = ef["key"]
        if k in EXCLUDED:
            continue
        default = ef.get("default")
        val = extras.get(k, default)
        updated_extras.append(f"--{k}={val}")

    for k, v in extras.items():
        if k not in known_keys and k not in EXCLUDED and v not in (None, ""):
            updated_extras.append(f"--{k}={v}")

    extra_params["args"].clear()
    extra_params["args"].extend(sorted(updated_extras))