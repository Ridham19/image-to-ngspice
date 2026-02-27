import tkinter as tk
from editor import CircuitEditor

if __name__ == "__main__":
    root = tk.Tk()
    app = CircuitEditor(root)
    root.mainloop()