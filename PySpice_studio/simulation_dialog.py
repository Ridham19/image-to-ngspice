import tkinter as tk
from tkinter import ttk

class SimulationDialog:
    def __init__(self, parent, available_nodes, available_sources, available_sweeps, current_config, on_save):
        self.top = tk.Toplevel(parent)
        self.top.title("Simulation Setup")
        self.top.geometry("650x750")
        
        # --- THEME COLORS ---
        BG_MAIN = "#1E1E1E"       # Window Background
        BG_PANEL = "#252526"      # Tab/Form Background
        BG_INPUT = "#333333"      # Entry/Combo Background
        FG_TEXT = "white"
        BTN_PRIMARY = "#0078D7"   # Blue
        BTN_DANGER = "#C62828"    # Red
        BTN_SECONDARY = "#444444" # Gray
        
        self.top.configure(bg=BG_MAIN)
        
        # Configure Styles
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TCombobox", fieldbackground=BG_INPUT, background=BG_INPUT, foreground="white", arrowcolor="white", borderwidth=0)
        style.configure("TNotebook", background=BG_MAIN, borderwidth=0)
        style.configure("TNotebook.Tab", background=BG_INPUT, foreground="#AAA", padding=[15, 5])
        style.map("TNotebook.Tab", background=[("selected", BG_PANEL)], foreground=[("selected", "white")])
        
        self.on_save = on_save
        self.current_config = current_config
        self.nodes = available_nodes if available_nodes else ["1"]
        self.sources = available_sources if available_sources else ["V1"]
        self.sweeps = available_sweeps if available_sweeps else ["V1"]
        self.colors = ["white", "black", "red", "blue", "green", "orange", "magenta", "cyan", "yellow", "gray", "brown"]
        
        # --- LAYOUT STRATEGY ---
        
        # 1. Footer (Pinned to Bottom)
        footer = tk.Frame(self.top, bg=BG_MAIN, pady=10)
        footer.pack(side="bottom", fill="x")
        
        tk.Button(footer, text="Save & Close", command=self.save, 
                  bg=BTN_PRIMARY, fg="white", font=("Segoe UI", 10, "bold"), 
                  bd=0, padx=20, pady=5).pack(side="right", padx=20)
        
        tk.Button(footer, text="Cancel", command=self.top.destroy, 
                  bg=BTN_SECONDARY, fg="white", font=("Segoe UI", 9), 
                  bd=0, padx=15, pady=5).pack(side="right", padx=5)

        # 2. Tabs (Fills Remaining Space)
        tabs = ttk.Notebook(self.top)
        tabs.pack(side="top", fill="both", expand=True, padx=10, pady=10)
        
        self.tab_analysis = tk.Frame(tabs, bg=BG_PANEL)
        self.tab_plot = tk.Frame(tabs, bg=BG_PANEL)
        
        tabs.add(self.tab_analysis, text="Analysis Config")
        tabs.add(self.tab_plot, text="Plot & Colors")
        
        self.build_analysis_ui(BG_PANEL, BG_INPUT)
        self.build_plot_ui(BG_PANEL, BG_INPUT, BTN_SECONDARY, BTN_DANGER)

    def lbl(self, parent, text, bg): 
        return tk.Label(parent, text=text, bg=bg, fg="white", font=("Segoe UI", 9))
    
    def build_analysis_ui(self, bg, input_bg):
        f = self.tab_analysis
        
        # Header
        header_frame = tk.Frame(f, bg=bg, pady=15)
        header_frame.pack(fill="x", padx=15)
        self.lbl(header_frame, "Simulation Mode:", bg).pack(side="left")
        
        prev_cmd = self.current_config.get('cmd', '.op')
        start_mode = "Operating Point"
        if ".tran" in prev_cmd: start_mode = "Transient"
        elif ".dc" in prev_cmd: start_mode = "DC Sweep"
        elif ".ac" in prev_cmd: start_mode = "AC Analysis"
        
        self.mode_var = tk.StringVar(value=start_mode)
        modes = ["Transient", "DC Sweep", "AC Analysis", "Operating Point"]
        cb = ttk.Combobox(header_frame, textvariable=self.mode_var, values=modes, state="readonly", width=25)
        cb.pack(side="left", padx=10)
        cb.bind("<<ComboboxSelected>>", lambda e: self.rebuild_form(bg, input_bg))
        
        # Dynamic Form
        self.form_frame = tk.Frame(f, bg=bg, pady=10)
        self.form_frame.pack(fill="both", expand=True, padx=15)
        self.entries = {}
        self.rebuild_form(bg, input_bg)

    def build_plot_ui(self, bg, input_bg, btn_bg, danger_bg):
        f = self.tab_plot
        
        # --- Graph Settings ---
        bg_frame = tk.LabelFrame(f, text="Graph Appearance", bg=bg, fg="#AAA", padx=10, pady=10)
        bg_frame.pack(fill="x", padx=15, pady=10)
        
        self.lbl(bg_frame, "Background Color:", bg).pack(side="left")
        self.cb_bg = ttk.Combobox(bg_frame, values=self.colors, state="readonly", width=12)
        saved_colors = self.current_config.get('colors', {})
        self.cb_bg.set(saved_colors.get('0', 'white'))
        self.cb_bg.pack(side="left", padx=10)

        # --- Active Signals List ---
        list_frame = tk.LabelFrame(f, text="Active Signals", bg=bg, fg="#AAA", padx=10, pady=10)
        list_frame.pack(fill="x", padx=15, pady=5)
        
        self.plot_data = [] 
        self.sig_listbox = tk.Listbox(list_frame, height=6, bg=input_bg, fg="white", 
                                      bd=0, highlightthickness=0, selectbackground="#0078D7")
        self.sig_listbox.pack(fill="x", pady=(0, 5))
        
        saved_plots = self.current_config.get('plots', {})
        idx_counter = 2
        for win_id, signals in saved_plots.items():
            for sig in signals:
                col = saved_colors.get(str(idx_counter), 'red')
                self.add_to_list(sig, col, win_id)
                idx_counter += 1

        tk.Button(list_frame, text="Remove Selected", command=self.remove_sig, 
                  bg=danger_bg, fg="white", bd=0, padx=10, pady=2).pack(anchor="e")
        
        # --- Add New Signal (Using Grid Layout for Stability) ---
        add_frame = tk.LabelFrame(f, text="Add New Signal", bg=bg, fg="#AAA", padx=10, pady=10)
        add_frame.pack(fill="x", padx=15, pady=10)
        
        # Configure Grid Column 1 to expand (Inputs)
        add_frame.columnconfigure(1, weight=1)
        
        # Row 0: Global Config
        cfg_frame = tk.Frame(add_frame, bg=bg)
        cfg_frame.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 10))
        
        self.lbl(cfg_frame, "Color:", bg).pack(side="left")
        self.cb_sig_color = ttk.Combobox(cfg_frame, values=self.colors, state="readonly", width=8)
        self.cb_sig_color.set("red")
        self.cb_sig_color.pack(side="left", padx=5)
        
        self.lbl(cfg_frame, "Window:", bg).pack(side="left", padx=(15,0))
        self.cb_window = ttk.Combobox(cfg_frame, values=["1", "2", "3", "4"], state="readonly", width=3)
        self.cb_window.current(0)
        self.cb_window.pack(side="left", padx=5)

        # Row 1: Voltage
        self.lbl(add_frame, "Voltage v(node):", bg).grid(row=1, column=0, sticky="w", pady=5)
        
        self.cb_node = ttk.Combobox(add_frame, values=self.nodes, state="readonly")
        if self.nodes: self.cb_node.current(0)
        self.cb_node.grid(row=1, column=1, sticky="ew", padx=10, pady=5)
        
        tk.Button(add_frame, text="Add", command=self.add_voltage, 
                  bg=btn_bg, fg="white", bd=0, width=8).grid(row=1, column=2, sticky="e", pady=5)

        # Row 2: Current
        self.lbl(add_frame, "Current i(source):", bg).grid(row=2, column=0, sticky="w", pady=5)
        
        self.cb_source = ttk.Combobox(add_frame, values=self.sources, state="readonly")
        if self.sources: self.cb_source.current(0)
        self.cb_source.grid(row=2, column=1, sticky="ew", padx=10, pady=5)
        
        tk.Button(add_frame, text="Add", command=self.add_current, 
                  bg=btn_bg, fg="white", bd=0, width=8).grid(row=2, column=2, sticky="e", pady=5)

    # --- LOGIC ---
    def add_voltage(self):
        n = self.cb_node.get()
        if n: self.add_to_list(f"v({n})", self.cb_sig_color.get(), self.cb_window.get())
        
    def add_current(self):
        s = self.cb_source.get()
        if s: self.add_to_list(f"i({s})", self.cb_sig_color.get(), self.cb_window.get())

    def add_to_list(self, sig, col, win):
        display = f"Win {win}  |  {sig}  [{col}]"
        self.sig_listbox.insert(tk.END, display)
        self.plot_data.append((sig, col, win))

    def remove_sig(self):
        sel = self.sig_listbox.curselection()
        if sel:
            idx = sel[0]
            self.sig_listbox.delete(idx)
            del self.plot_data[idx]

    def rebuild_form(self, bg, input_bg):
        for w in self.form_frame.winfo_children(): w.destroy()
        self.entries = {}
        m = self.mode_var.get()
        
        if m == "Transient": 
            self.add("Step Time", "0.1m", bg, input_bg)
            self.add("Stop Time", "80m", bg, input_bg)
            self.add("Start Time", "0", bg, input_bg)
        elif m == "DC Sweep": 
            self.add_c("Source 1", self.sweeps, bg, input_bg)
            self.add("Start", "0", bg, input_bg); self.add("Stop", "5", bg, input_bg); self.add("Incr", "0.1", bg, input_bg)
            
            sep = tk.Label(self.form_frame, text="--- Secondary Sweep (Optional) ---", bg=bg, fg="#888")
            sep.pack(pady=10)
            
            self.add_c("Source 2", ["None"] + self.sweeps, bg, input_bg)
            self.add("Start 2", "0", bg, input_bg); self.add("Stop 2", "5", bg, input_bg); self.add("Incr 2", "1", bg, input_bg)
        elif m == "AC Analysis": 
            self.add_c("Type", ["DEC", "LIN"], bg, input_bg)
            self.add("Points", "10", bg, input_bg)
            self.add("Start Freq", "1", bg, input_bg)
            self.add("Stop Freq", "10meg", bg, input_bg)

    def add(self, lbl, default, bg, input_bg):
        row = tk.Frame(self.form_frame, bg=bg)
        row.pack(fill="x", pady=2)
        tk.Label(row, text=lbl, width=20, anchor="w", bg=bg, fg="white").pack(side="left")
        e = tk.Entry(row, bg=input_bg, fg="white", insertbackground="white", bd=0, highlightthickness=1, highlightbackground="#555")
        e.insert(0, default)
        e.pack(side="right", expand=True, fill="x", padx=5, ipady=3)
        self.entries[lbl] = e

    def add_c(self, lbl, vals, bg, input_bg):
        row = tk.Frame(self.form_frame, bg=bg)
        row.pack(fill="x", pady=2)
        tk.Label(row, text=lbl, width=20, anchor="w", bg=bg, fg="white").pack(side="left")
        c = ttk.Combobox(row, values=vals, state="readonly")
        if vals: c.current(0)
        c.pack(side="right", expand=True, fill="x", padx=5)
        self.entries[lbl] = c

    def get_val(self, key): return self.entries[key].get()

    def save(self):
        m = self.mode_var.get()
        cmd = ""
        g = self.get_val
        
        if m == "Transient": cmd = f".tran {g('Step Time')} {g('Stop Time')} {g('Start Time')}"
        elif m == "DC Sweep": 
            s1 = g('Source 1'); s2 = g('Source 2')
            part1 = f"{s1} {g('Start')} {g('Stop')} {g('Incr')}"
            part2 = f" {s2} {g('Start 2')} {g('Stop 2')} {g('Incr 2')}" if s2 != "None" else ""
            cmd = f".dc {part1}{part2}"
        elif m == "AC Analysis": cmd = f".ac {g('Type')} {g('Points')} {g('Start Freq')} {g('Stop Freq')}"
        elif m == "Operating Point": cmd = ".op"

        plots = {}
        colors = {}
        colors['0'] = self.cb_bg.get()
        colors['1'] = 'white' if colors['0'] == 'black' else 'black'
        for i, (sig, col, win) in enumerate(self.plot_data):
            if win not in plots: plots[win] = []
            plots[win].append(sig)
            colors[str(i+2)] = col 
            
        self.on_save({'cmd': cmd, 'plots': plots, 'colors': colors})
        self.top.destroy()