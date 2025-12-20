import tkinter as tk

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