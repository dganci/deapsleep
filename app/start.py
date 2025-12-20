# Main application file to start the GUI
import tkinter as tk
from tkinter import ttk
import gui_optimize, gui_compare, gui_plot
from _gif import AnimatedGIF
from utils.paths import ROOT
import os

def main_app():
    '''
    Main application window for deapsleep GUI.
    '''
    # Create the main application window
    root = tk.Tk()
    root.title("deapsleep GUI")
    root.geometry("800x400")

    # Title
    ttk.Label(
        root,
        text="DeapSleep",
        font=("Arial", 20, "bold")
    ).pack(pady=(10, 0)) 

    # Subtitle
    ttk.Label(
        root,
        text="A DEAP-based toolkit for testing dropout in genetic algorithms.",
        font=("Arial", 12)
    ).pack(pady=(0, 15)) 

    # Buttons
    frame = ttk.Frame(root)
    frame.pack(pady=20)

    style = ttk.Style()
    style.configure("Big.TButton", font=("Arial", 12), padding=10)

    ttk.Button(
        frame, 
        text="1. Run optimization", 
        width=40,   
        command=gui_optimize.window,
        style="Big.TButton"
    ).grid(row=0, column=0, pady=10)

    ttk.Button(
        frame, 
        text="2. Plot results", 
        width=40,
        command=gui_plot.window,
        style="Big.TButton"
    ).grid(row=1, column=0, pady=10)

    ttk.Button(
        frame, 
        text="3. Compare two versions", 
        width=40,
        command=gui_compare.window,
        style="Big.TButton"
    ).grid(row=2, column=0, pady=10)

    try:
        gif = AnimatedGIF(root, os.path.join(ROOT, "app/opt.gif"), delay=100, scale=0.3)
        gif.place(relx=0.0, rely=1.0, anchor="sw", x=5, y=-5)
    except Exception as e:
        print(f"[WARN] Could not load GIF: {e}")

    credit = tk.Label(root, text="© Daniele Ganci", font=("Arial", 8))
    credit.place(relx=1.0, rely=1.0, anchor="se", x=-5, y=-5)

    # Start the main event loop
    root.mainloop()

if __name__ == "__main__":
    main_app()
