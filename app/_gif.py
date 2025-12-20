import tkinter as tk
from PIL import Image, ImageTk

class AnimatedGIF(tk.Label):
    def __init__(self, parent, gif_path, delay=100, scale=1.0):
        super().__init__(parent, borderwidth=0, bg="white")
        self.frames = []
        self.delay = delay
        self.frame_index = 0

        img = Image.open(gif_path)
        try:
            while True:
                frame = img.copy().convert("RGBA")
                if scale != 1.0:
                    w, h = frame.size
                    frame = frame.resize((int(w*scale), int(h*scale)), Image.Resampling.LANCZOS)
                self.frames.append(ImageTk.PhotoImage(frame))
                img.seek(img.tell() + 1)
        except EOFError:
            pass

        if not self.frames:
            raise ValueError(f"No frames found in {gif_path}")

        self.configure(image=self.frames[0])
        self.after(self.delay, self.animate)

    def animate(self):
        self.frame_index = (self.frame_index + 1) % len(self.frames)
        self.configure(image=self.frames[self.frame_index])
        self.after(self.delay, self.animate)
