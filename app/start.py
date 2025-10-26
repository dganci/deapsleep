# Main application file to start the GUI
import tkinter as tk
from tkinter import ttk
import gui_optimize, gui_compare, gui_plot

def main_app():
    '''
    Main application window for deapsleep GUI.
    '''
    # Create the main application window
    root = tk.Tk()
    root.title("deapsleep GUI")
    root.geometry("600x300")

    # Title
    ttk.Label(
        root,
        text="DeapSleep",
        font=("Arial", 18, "bold")
    ).pack(pady=(10, 0)) 

    # Subtitle
    ttk.Label(
        root,
        text="A DEAP-based framework for testing dropout in genetic algorithms.",
        font=("Arial", 11)
    ).pack(pady=(0, 15)) 

    # Buttons
    frame = ttk.Frame(root)
    frame.pack(pady=20)

    ttk.Button(frame, text="1. Run optimization", width=35, command=gui_optimize.window).grid(
        row=0, column=0, pady=10
    )
    ttk.Button(frame, text="2. Plot results", width=35, command=gui_plot.window).grid(
        row=1, column=0, pady=10
    )
    ttk.Button(frame, text="3. Compare two versions", width=35, command=gui_compare.window).grid(
        row=2, column=0, pady=10
    )

    credit = tk.Label(root, text="© Daniele Ganci", font=("Arial", 8))
    credit.place(relx=1.0, rely=1.0, anchor="se", x=-5, y=-5)

    # Start the main event loop
    root.mainloop()

if __name__ == "__main__":
    main_app()
