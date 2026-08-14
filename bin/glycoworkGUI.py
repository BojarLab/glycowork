import os
import re
import sys
import time
import base64
import warnings
import threading
import subprocess
import pandas as pd
import tkinter as tk
from tkinter import simpledialog, filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore', message = '.*non-interactive.*')
from glycowork.glycan_data.loader import motif_list
from glycowork.motif.draw import GlycoDraw, plot_glycans_excel
from glycowork.motif.analysis import get_differential_expression, get_heatmap, get_lectin_array, get_volcano, get_ma, get_coverage, get_pca
from glycowork.motif.processing import canonicalize_iupac

def create_tooltip(widget, text):
  tl = widget.winfo_toplevel()
  tip = ttk.Label(tl, text=text, background="#FFFBE6", relief='solid', borderwidth=1, wraplength=220,
                  font=('Helvetica', 9), padding=(4, 2))
  def show(e):
    tip.lift()
    tip.place(x=widget.winfo_rootx()-tl.winfo_rootx()+widget.winfo_width()+6,
              y=widget.winfo_rooty()-tl.winfo_rooty())
  def hide(e):
    tip.place_forget()
  widget.bind('<Enter>', show, add='+')
  widget.bind('<Leave>', hide, add='+')
  tip.bind('<Enter>', hide, add='+')


class BaseDialog(simpledialog.Dialog):
    def __init__(self, parent, title = None):
        super().__init__(parent, title)

    def add_file_input(self, master, row, label_text, help_text=None, filetypes=None):
        if filetypes is None:
            filetypes = [("Data Files", "*.csv *.tsv *.xlsx"), ("CSV Files", "*.csv"),
                         ("TSV Files", "*.tsv"), ("Excel Files", "*.xlsx")]
        frame = ttk.Frame(master)
        frame.grid(row=row, column=0, columnspan=3, sticky='ew', pady=5)
        frame.grid_columnconfigure(1, weight=1)
        ttk.Label(frame, text=label_text).grid(row=0, column=0, sticky='w', padx=(0,5))
        entry_var = tk.StringVar(frame)
        entry = ttk.Entry(frame, textvariable=entry_var, state='readonly')
        entry.grid(row=0, column=1, sticky='ew', padx=5)
        browse_btn = ttk.Button(frame, text="Browse",
                              command=lambda: self.browse_file(entry_var, filetypes),
                              style='Modern.TButton')
        browse_btn.grid(row=0, column=2, padx=(5,0))
        if help_text:
            help_btn = ttk.Label(frame, text="?", cursor="question_arrow")
            help_btn.grid(row=0, column=3, padx=(5,0))
            create_tooltip(help_btn, help_text)
        return entry_var

    def add_folder_input(self, master, row, label_text):
        frame = ttk.Frame(master)
        frame.grid(row=row, column=0, columnspan=3, sticky='ew', pady=5)
        frame.grid_columnconfigure(1, weight=1)
        ttk.Label(frame, text=label_text).grid(row=0, column=0, sticky='w', padx=(0,5))
        entry_var = tk.StringVar(frame)
        entry = ttk.Entry(frame, textvariable=entry_var, state='readonly')
        entry.grid(row=0, column=1, sticky='ew', padx=5)
        browse_btn = ttk.Button(frame, text="Browse",
                              command=lambda: self.browse_folder(entry_var),
                              style='Modern.TButton')
        browse_btn.grid(row=0, column=2, padx=(5,0))
        return entry_var

    def browse_file(self, entry_var, filetypes=None):
        if filetypes is None:
            filetypes = [("CSV Files", "*.csv"), ("Excel Files", "*.xlsx")]
        file_path = filedialog.askopenfilename(filetypes=filetypes)
        if file_path:
            entry_var.set(file_path)

    def browse_folder(self, entry_var):
        folder_path = filedialog.askdirectory()
        if folder_path:
            entry_var.set(folder_path)

    def buttonbox(self):
        box = ttk.Frame(self)
        ok_btn = ttk.Button(box, text="OK", command=self.ok, style='Modern.TButton')
        ok_btn.pack(side=tk.LEFT, padx=5, pady=5)
        cancel_btn = ttk.Button(box, text="Cancel", command=self.cancel,
                              style='Modern.TButton')
        cancel_btn.pack(side=tk.LEFT, padx=5, pady=5)
        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)
        box.pack(pady=5)

    def add_group_selector(self, master, row, label_text, help_text = None):
        frame = ttk.Frame(master)
        frame.grid(row = row, column = 0, columnspan = 3, sticky = 'ew', pady = 5)
        frame.grid_columnconfigure(1, weight = 1)
        ttk.Label(frame, text = label_text).grid(row = 0, column = 0, sticky = 'nw', padx = (0, 5))
        box = tk.Listbox(frame, selectmode = tk.EXTENDED, height = 5, exportselection = False,
                         relief = 'flat', highlightthickness = 1, activestyle = 'none')
        box.grid(row = 0, column = 1, sticky = 'ew', padx = 5)
        if help_text:
            lbl = ttk.Label(frame, text = "?", cursor = "question_arrow")
            lbl.grid(row = 0, column = 2, padx = (5, 0))
            create_tooltip(lbl, help_text)
        return box

    def populate_groups(self, path, by_rows, *boxes):
        if not path:
            return
        try:
            suffix = Path(path).suffix.lower()
            head = pd.read_csv(path, nrows = 500) if suffix == '.csv' else pd.read_csv(path, sep = '\t', nrows = 500) if suffix == '.tsv' else pd.read_excel(path, nrows = 500)
            labels = head.iloc[:, 0].astype(str).tolist() if by_rows else head.columns[1:].tolist()
        except Exception as e:
            messagebox.showerror("Error", f"Could not read {os.path.basename(path)}:\n{e}", parent = self)
            return
        for box in boxes:
            box.delete(0, tk.END)
            for i, name in enumerate(labels, start = 1):
                box.insert(tk.END, f"{i:>3}  {name}")

    def selected(self, box):
        return [i + 1 for i in box.curselection()]


class GlycoDrawDialog(BaseDialog):
    _recent = []
    def body(self, master):
        self.title("Draw Glycan")
        self._job, self._img = None, None
        seq_frame = ttk.LabelFrame(master, text = "Glycan Sequence", padding = 10)
        seq_frame.pack(fill = tk.X, padx = 10, pady = 5)
        self.sequence_entry = ttk.Entry(seq_frame, width = 70)
        self.sequence_entry.pack(fill = tk.X, padx = 5, pady = 5)
        self.sequence_entry.bind('<KeyRelease>', lambda e: self.schedule_preview())
        ttk.Label(seq_frame, style = 'Sub.TLabel',
                  text = "IUPAC-condensed, WURCS, GlycoCT, Oxford, GLYCAM, LinearCode or a composition are all accepted").pack(anchor = 'w', padx = 5)
        opt = ttk.LabelFrame(master, text = "Style", padding = 10)
        opt.pack(fill = tk.X, padx = 10, pady = 5)
        self.compact_var, self.vertical_var = tk.BooleanVar(), tk.BooleanVar()
        self.linkage_var = tk.BooleanVar(value = True)
        for text, var, tip in (("Compact", self.compact_var, "Drop linkage spacing so large structures stay readable"),
                               ("Vertical", self.vertical_var, "Rotate the structure 90 degrees"),
                               ("Show linkages", self.linkage_var, "Print linkage labels such as b1-4 on the bonds")):
            cb = ttk.Checkbutton(opt, text = text, variable = var, command = self.render_preview)
            cb.pack(side = tk.LEFT, padx = 6)
            create_tooltip(cb, tip)
        row = ttk.Frame(opt)
        row.pack(fill = tk.X, pady = (8, 0))
        ttk.Label(row, text = "Highlight motif:").pack(side = tk.LEFT)
        self.highlight_var = tk.StringVar()
        hl = ttk.Combobox(row, textvariable = self.highlight_var, width = 28,
                          values = [''] + sorted(motif_list.motif_name.tolist()))
        hl.pack(side = tk.LEFT, padx = 5)
        hl.bind('<<ComboboxSelected>>', lambda e: self.render_preview())
        ttk.Label(row, text = "Save as:").pack(side = tk.LEFT, padx = (12, 0))
        self.format_var = tk.StringVar(value = 'pdf')
        ttk.Combobox(row, textvariable = self.format_var, values = ['pdf', 'svg', 'png'], width = 5,
                     state = 'readonly').pack(side = tk.LEFT, padx = 5)
        self.preview = ttk.Label(master, anchor = 'center', background = '#FFFFFF', relief = 'solid',
                                 borderwidth = 1, text = "Preview appears here as you type")
        self.preview.pack(fill = tk.BOTH, expand = True, padx = 10, pady = 5)
        if GlycoDrawDialog._recent:
            hist = ttk.Combobox(master, values = GlycoDrawDialog._recent, state = 'readonly')
            hist.pack(fill = tk.X, padx = 10, pady = (0, 5))
            hist.bind('<<ComboboxSelected>>', lambda e: (self.sequence_entry.delete(0, tk.END),
                                                         self.sequence_entry.insert(0, hist.get()), self.render_preview()))
        return self.sequence_entry

    def schedule_preview(self):
        if self._job:
            self.after_cancel(self._job)
        self._job = self.after(400, self.render_preview)

    def render_preview(self):
        self._job = None
        seq = self.sequence_entry.get().strip()
        if not seq:
            self.preview.configure(image = '', text = "Preview appears here as you type")
            return
        try:
            png = GlycoDraw(seq, compact = self.compact_var.get(), vertical = self.vertical_var.get(),
                            show_linkage = self.linkage_var.get(), suppress = True,
                            highlight_motif = self.highlight_var.get() or None)._repr_png_()
            img = tk.PhotoImage(data = base64.b64encode(png).decode())
            shrink = max(1, -(-img.width() // 620), -(-img.height() // 300))
            self._img = img.subsample(shrink) if shrink > 1 else img
            self.preview.configure(image = self._img, text = '')
        except Exception as e:
            self._img = None
            self.preview.configure(image = '', text = f"Cannot draw this sequence:\n{e}")

    def validate(self):
        if not self.sequence_entry.get().strip():
            messagebox.showerror("Error", "Please enter a glycan sequence", parent = self)
            return 0
        return 1

    def apply(self):
        sequence = self.sequence_entry.get().strip()
        if sequence not in GlycoDrawDialog._recent:
            GlycoDrawDialog._recent = (GlycoDrawDialog._recent + [sequence])[-10:]
        self.result = (sequence, self.compact_var.get(), self.vertical_var.get(), self.linkage_var.get(),
                       self.highlight_var.get() or None, self.format_var.get())


class GlycoDrawExcelDialog(BaseDialog):
    def body(self, master):
        self.title("Batch Draw Glycans")
        # Input frame
        input_frame = ttk.LabelFrame(master, text="Input Data", padding=10)
        input_frame.pack(fill=tk.X, padx=10, pady=5)
        help_text = ("CSV Format Help:\n\n"
                    "Glycans should be in first column (ideally in IUPAC-condensed)\n"
                    "Other columns (e.g., abundances or intensities) are permitted")
        self.csv_var = self.add_file_input(input_frame, 0, "Select CSV/Excel:", help_text)
        # Output frame
        output_frame = ttk.LabelFrame(master, text="Output Options", padding=10)
        output_frame.pack(fill=tk.X, padx=10, pady=5)
        self.folder_var = self.add_folder_input(output_frame, 0, "Save Images To:")
        # Display options
        self.compact_var = tk.BooleanVar()
        ttk.Checkbutton(output_frame, text="Compact Display", variable=self.compact_var).grid(row=1, column=0, pady=5, sticky="w")
        return None

    def validate(self):
        for value, msg in ((self.csv_var.get(), "Please select an input file"),
                           (self.folder_var.get(), "Please select an output folder")):
            if not value:
                messagebox.showerror("Error", msg, parent = self)
                return 0
        return 1

    def apply(self):
        self.result = (self.csv_var.get(), self.folder_var.get(), self.compact_var.get())


class ProgressDialog(tk.Toplevel):
    def __init__(self, parent, title="Processing", determinate=False):
        super().__init__(parent)
        self.title(title)
        self.geometry("400x150")
        self.transient(parent)
        self.wait_visibility()
        self.grab_set()
        style = ttk.Style()
        style.configure("Modern.Horizontal.TProgressbar", thickness=20, troughcolor='#E0E0E0', background='#4CAF50')
        # Main frame
        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        # Status message
        self.status_var = tk.StringVar(value="Initializing...")
        status_label = ttk.Label(main_frame, textvariable=self.status_var, font=("Helvetica", 10))
        status_label.pack(pady=(0, 15))
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, orient="horizontal", length=300,
            mode='determinate' if determinate else 'indeterminate', style="Modern.Horizontal.TProgressbar")
        self.progress.pack(fill=tk.X, pady=10)
        # Time elapsed
        self.time_var = tk.StringVar(value="Time elapsed: 0:00")
        time_label = ttk.Label(main_frame, textvariable=self.time_var)
        time_label.pack(pady=(10, 15))
        # Cancel button
        self.cancel_btn = ttk.Button(main_frame, text="Cancel", command=self.request_cancel, style='Modern.TButton')
        self.cancel_btn.pack()
        self.cancelled = False
        self.start_time = time.time()
        self.update_time()
        if not determinate:
            self.progress.start(10)
        self.protocol("WM_DELETE_WINDOW", self.request_cancel)

    def update_time(self):
        if self.cancelled or not self.winfo_exists():
            return
        elapsed = int(time.time() - self.start_time)
        self.time_var.set(f"Time elapsed: {elapsed // 60}:{elapsed % 60:02d}")
        self.after(1000, self.update_time)

    def request_cancel(self):
        if messagebox.askyesno("Hide Progress", "The analysis cannot be interrupted once started.\nHide this window and let it finish in the background?", parent = self):
            self.cancelled = True
            self.grab_release()
            self.withdraw()

    def update_status(self, message, progress_value = None):
        if not self.winfo_exists():
            return
        self.status_var.set(message)
        if progress_value is not None and self.progress['mode'] == 'determinate':
            self.progress['value'] = progress_value

    def end(self):
        if self.winfo_exists():
            self.progress.stop()
            self.destroy()

    def finish(self, message = "Operation completed successfully"):
        if not self.winfo_exists():
            return
        self.progress.stop()
        self.status_var.set(message)
        self.cancel_btn.configure(text = "Close", command = self.end)
        self.after(2000, self.end)


class DifferentialExpressionDialog(BaseDialog):
    def body(self, master):
        self.title("Differential Expression Analysis")
        # Input frame
        input_frame = ttk.LabelFrame(master, text="Input Data", padding=10)
        input_frame.pack(fill=tk.X, padx=10, pady=5)
        help_text = ("CSV Format Help:\n\n"
                    "Glycans should be in first column (ideally in IUPAC-condensed)\n"
                    "If you do NOT analyze motifs, the glycan format does not matter\n"
                    "Other columns should be the abundances (each sample one column)")
        self.csv_var = self.add_file_input(input_frame, 0, "CSV/Excel File:", help_text)
        # Groups frame
        groups_frame = ttk.LabelFrame(master, text="Sample Groups", padding=10)
        groups_frame.pack(fill=tk.X, padx=10, pady=5)
        groups_help = "Sample columns are listed once you pick a file. Ctrl-click or Shift-click to select several."
        self.treatment_box = self.add_group_selector(groups_frame, 0, "Treatment Samples:", groups_help)
        self.control_box = self.add_group_selector(groups_frame, 1, "Control Samples:", groups_help)
        self.csv_var.trace_add('write', lambda *a: self.populate_groups(self.csv_var.get(), False, self.treatment_box,
                                                                        self.control_box))
        # Analysis options
        options_frame = ttk.LabelFrame(master, text="Analysis Options", padding=10)
        options_frame.pack(fill=tk.X, padx=10, pady=5)
        self.motifs_var = tk.BooleanVar()
        motif_cb = ttk.Checkbutton(options_frame, text = "Perform Motif-based Analysis", variable = self.motifs_var)
        motif_cb.pack(anchor = 'w', pady = 2)
        create_tooltip(motif_cb,
                       "Test known and exhaustive substructures instead of whole glycans. Requires IUPAC-condensed sequences.")
        self.plots_var = tk.BooleanVar(value = True)
        plots_cb = ttk.Checkbutton(options_frame, text = "Also save volcano and MA plots", variable = self.plots_var)
        plots_cb.pack(anchor = 'w', pady = 2)
        create_tooltip(plots_cb,
                       "Volcano: effect size vs significance, with SNFG symbols on the labelled hits. MA: effect size vs mean abundance.")
        # Output frame
        output_frame = ttk.LabelFrame(master, text="Output", padding=10)
        output_frame.pack(fill=tk.X, padx=10, pady=5)
        self.output_folder_var = self.add_folder_input(output_frame, 0, "Save Results To:")
        return None

    def validate(self):
        for value, msg in ((self.csv_var.get(), "Please select an input file"),
                           (self.selected(self.treatment_box), "Please select the treatment samples"),
                           (self.selected(self.control_box), "Please select the control samples"),
                           (self.output_folder_var.get(), "Please select an output folder")):
            if not value:
                messagebox.showerror("Error", msg, parent = self)
                return 0
        return 1

    def apply(self):
        self.result = (self.csv_var.get(), self.selected(self.treatment_box), self.selected(self.control_box),
                       self.motifs_var.get(), self.plots_var.get(), self.output_folder_var.get())


class DataOverviewDialog(BaseDialog):
    def body(self, master):
        self.title("Data Overview")
        input_frame = ttk.LabelFrame(master, text = "Input Data", padding = 10)
        input_frame.pack(fill = tk.X, padx = 10, pady = 5)
        help_text = ("Glycans in the first column, one sample per remaining column.\n"
                     "Group selection is optional and only colours the PCA.")
        self.file_var = self.add_file_input(input_frame, 0, "CSV/Excel File:", help_text)
        groups_frame = ttk.LabelFrame(master, text = "Sample Groups (optional)", padding = 10)
        groups_frame.pack(fill = tk.X, padx = 10, pady = 5)
        self.group_a = self.add_group_selector(groups_frame, 0, "Group A:")
        self.group_b = self.add_group_selector(groups_frame, 1, "Group B:")
        self.file_var.trace_add('write', lambda *a: self.populate_groups(self.file_var.get(), False, self.group_a, self.group_b))
        output_frame = ttk.LabelFrame(master, text = "Output", padding = 10)
        output_frame.pack(fill = tk.X, padx = 10, pady = 5)
        self.output_dir_var = self.add_folder_input(output_frame, 0, "Save Plots To:")
        return None

    def validate(self):
        for value, msg in ((self.file_var.get(), "Please select an input file"),
                           (self.output_dir_var.get(), "Please select an output folder")):
            if not value:
                messagebox.showerror("Error", msg, parent = self)
                return 0
        return 1

    def apply(self):
        a, b = self.selected(self.group_a), self.selected(self.group_b)
        groups = [1 if i in a else 2 for i in range(1, self.group_a.size() + 1)] if a and b else None
        self.result = (self.file_var.get(), groups, self.output_dir_var.get())


class GetHeatmapDialog(BaseDialog):
    def body(self, master):
        self.title("Generate Heatmap")
        input_frame = ttk.LabelFrame(master, text="Input Data", padding=10)
        input_frame.pack(fill=tk.X, padx=10, pady=5)
        help_text = ("CSV Format Help:\n\n"
                    "Ideally, rows are samples and columns are glycans (but the function can deal with the opposite)\n"
                    "Glycans should be ideally in IUPAC-condensed\n"
                    "If you do NOT analyze motifs, the glycan format does not matter at all")
        self.input_file_var = self.add_file_input(input_frame, 0, "Select Input CSV or Excel File:", help_text)
        # Analysis options frame
        options_frame = ttk.LabelFrame(master, text="Analysis Options", padding=10)
        options_frame.pack(fill=tk.X, padx=10, pady=5)
        self.motif_analysis_var, self.clr_transform_var, self.show_all_var = tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar()
        for text, var, tip in (("Motif Analysis", self.motif_analysis_var,
                                "Cluster on known and exhaustive substructures instead of whole glycans. Needs IUPAC-condensed sequences."),
                               ("CLR Transform", self.clr_transform_var,
                                "Centered log-ratio. Use this when your values are relative abundances that sum to a constant, which is the norm for MS glycomics."),
                               ("Show All Features", self.show_all_var,
                                "Print every row and column label. Readable up to roughly 50 features, unreadable beyond that.")):
            cb = ttk.Checkbutton(options_frame, text = text, variable = var)
            cb.pack(anchor = 'w', padx = 5, pady = 2)
            create_tooltip(cb, tip)
        # Output frame
        output_frame = ttk.LabelFrame(master, text="Output", padding=10)
        output_frame.pack(fill=tk.X, padx=10, pady=5)
        self.output_file_var = self.add_folder_input(output_frame, 0, "Save Heatmap To:")
        return None

    def validate(self):
        for value, msg in ((self.input_file_var.get(), "Please select an input file"),
                           (self.output_file_var.get(), "Please select an output location")):
            if not value:
                messagebox.showerror("Error", msg, parent = self)
                return 0
        return 1

    def apply(self):
        self.result = (self.input_file_var.get(), self.motif_analysis_var.get(),
                       self.clr_transform_var.get(), self.show_all_var.get(),
                       os.path.join(self.output_file_var.get(), f"heatmap_{time.strftime('%Y%m%d_%H%M%S')}.png"))


class LectinArrayAnalysisDialog(BaseDialog):
    def body(self, master):
        self.title("Lectin Array Analysis")
        # Input frame
        input_frame = ttk.LabelFrame(master, text="Input Data", padding=10)
        input_frame.pack(fill=tk.X, padx=10, pady=5)
        help_text = ("CSV Format Help:\n\n"
                    "Format data as samples as rows and lectins as columns\n"
                    "First column should contain sample names\n"
                    "Column headers should contain lectin names")
        self.file_var = self.add_file_input(input_frame, 0, "Select CSV/Excel:", help_text)
        # Groups frame
        groups_frame = ttk.LabelFrame(master, text="Sample Groups", padding=10)
        groups_frame.pack(fill=tk.X, padx=10, pady=5)
        groups_help = "Sample names are listed once you pick a file. Ctrl-click or Shift-click to select several."
        self.treatment_box = self.add_group_selector(groups_frame, 0, "Treatment Samples:", groups_help)
        self.control_box = self.add_group_selector(groups_frame, 1, "Control Samples:", groups_help)
        self.file_var.trace_add('write', lambda *a: self.populate_groups(self.file_var.get(), True, self.treatment_box,
                                                                         self.control_box))
        # Analysis options
        options_frame = ttk.LabelFrame(master, text="Analysis Options", padding=10)
        options_frame.pack(fill=tk.X, padx=10, pady=5)
        self.paired_var = tk.BooleanVar()
        paired_cb = ttk.Checkbutton(options_frame, text = "Paired Analysis", variable = self.paired_var)
        paired_cb.pack(anchor = 'w', pady = 5)
        create_tooltip(paired_cb,
                       "Tick only if each treatment sample has a matching control from the same subject, in the same selection order.")
        # Output frame
        output_frame = ttk.LabelFrame(master, text="Output", padding=10)
        output_frame.pack(fill=tk.X, padx=10, pady=5)
        self.output_dir_var = self.add_folder_input(output_frame, 0, "Save Results To:")
        return None

    def validate(self):
        for value, msg in ((self.file_var.get(), "Please select an input file"),
                           (self.selected(self.treatment_box), "Please select the treatment samples"),
                           (self.selected(self.control_box), "Please select the control samples"),
                           (self.output_dir_var.get(), "Please select an output directory")):
            if not value:
                messagebox.showerror("Error", msg, parent = self)
                return 0
        return 1

    def apply(self):
        self.result = (self.file_var.get(), self.selected(self.treatment_box), self.selected(self.control_box),
                       self.paired_var.get(), self.output_dir_var.get())


class CanonicalizeIUPACDialog(BaseDialog):
  def body(self, master):
    self.title("Canonicalize IUPAC Sequences")
    # Input frame
    input_frame = ttk.LabelFrame(master, text="Input Sequences", padding=10)
    input_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    #help_text = "Enter one or more glycan sequences, one per line"
    # Text area for input
    self.input_text = ScrolledText(input_frame, height=10, width=50, font=('Courier', 10))
    self.input_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    # Output frame
    output_frame = ttk.LabelFrame(master, text="Canonicalized Sequences", padding=10)
    output_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    # Text area for output with readonly state
    self.output_text = ScrolledText(output_frame, height = 10, width = 50, font = ('Courier', 10), state = 'disabled')
    self.output_text.tag_configure('ERROR', foreground = 'red')
    self.output_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    # Buttons frame
    button_frame = ttk.Frame(master)
    button_frame.pack(fill=tk.X, padx=10, pady=5)
    # Canonicalize button
    self.canonicalize_btn = ttk.Button(button_frame, text="Canonicalize",
                                      command=self.canonicalize_sequences,
                                      style='Modern.TButton')
    self.canonicalize_btn.pack(side=tk.LEFT, padx=5)
    # Copy button
    self.copy_btn = ttk.Button(button_frame, text="Copy Results",
                              command=self.copy_to_clipboard,
                              style='Modern.TButton')
    self.copy_btn.pack(side=tk.LEFT, padx=5)
    # Clear button
    self.clear_btn = ttk.Button(button_frame, text="Clear All",
                               command=self.clear_all,
                               style='Modern.TButton')
    self.clear_btn.pack(side=tk.LEFT, padx=5)
    return self.input_text

  def canonicalize_sequences(self):
    # Get input text
    input_sequences = self.input_text.get(1.0, tk.END).strip().split('\n')
    results = []
    errors = []
    # Process each sequence
    for i, seq in enumerate(input_sequences):
      seq = seq.strip()
      if not seq:
        continue
      try:
        canonical = canonicalize_iupac(seq)
        results.append(f"{canonical}")
      except Exception as e:
        errors.append(f"Error in sequence {i+1} ({seq}): {str(e)}")
    # Update output text
    self.output_text.config(state='normal')
    self.output_text.delete(1.0, tk.END)
    if results:
      self.output_text.insert(tk.END, '\n'.join(results))
    if errors:
      self.output_text.insert(tk.END, '\n\n' + '\n'.join(errors), 'ERROR')
    self.output_text.config(state='disabled')

  def copy_to_clipboard(self):
    output_text = self.output_text.get(1.0, tk.END).strip()
    if output_text:
      self.clipboard_clear()
      self.clipboard_append(output_text)
      messagebox.showinfo("Success", "Results copied to clipboard")

  def clear_all(self):
    self.input_text.delete(1.0, tk.END)
    self.output_text.config(state='normal')
    self.output_text.delete(1.0, tk.END)
    self.output_text.config(state='disabled')

  def apply(self):
    # Just close the dialog when OK is pressed
    pass


class GlycoworkGUI:
    def __init__(self):
        self.app = tk.Tk()
        self.app.title("glycowork Analysis Suite")
        self.app.geometry("800x600")
        # Configure style
        self.setup_styles()
        # Create main container
        self.main_container = ttk.Frame(self.app, padding="10")
        self.main_container.pack(fill=tk.BOTH, expand=True)
        # Setup UI components
        self.setup_ui()
        self.setup_menu()
        self.setup_icon()
        # Initialize logging
        self.setup_logging()
        self.draw_folder, self.last_output = '', ''

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        BG, SURFACE, ACCENT, ACCENT_DARK = '#EEF2F7', '#FFFFFF', '#2B5EA7', '#1A3F7A'
        TEXT, MUTED, BORDER = '#1A202C', '#5A6882', '#C5CDD8'
        style.configure('.', background = BG, foreground = TEXT, font = ('Helvetica', 10))
        style.configure('TFrame', background = BG)
        style.configure('TLabel', background = BG, foreground = TEXT)
        style.configure('TCheckbutton', background = BG)
        style.configure('TLabelframe', background = BG, bordercolor = BORDER, relief = 'groove')
        style.configure('TLabelframe.Label', background = BG, foreground = ACCENT, font = ('Helvetica', 10, 'bold'))
        style.configure('TEntry', fieldbackground = SURFACE, bordercolor = BORDER, padding = 4)
        style.configure('TCombobox', fieldbackground = SURFACE, bordercolor = BORDER)
        style.configure('TButton', background = ACCENT, foreground = SURFACE, padding = (10, 5),
                        relief = 'flat', borderwidth = 0, font = ('Helvetica', 10))
        style.map('TButton', background = [('active', ACCENT_DARK), ('pressed', '#0F2A5A')],
                  foreground = [('active', SURFACE)])
        style.configure('Tool.TButton', padding = (10, 9), font = ('Helvetica', 10, 'bold'))
        style.configure('Modern.TButton', padding = (8, 4))
        style.configure('Header.TLabel', font = ('Helvetica', 17, 'bold'), foreground = ACCENT, background = BG)
        style.configure('Sub.TLabel', font = ('Helvetica', 10), foreground = MUTED, background = BG)
        style.configure('Sidebar.TFrame', background = '#DDE4EE')
        style.configure('Modern.Horizontal.TProgressbar', thickness = 18, troughcolor = '#D0D8E4',
                        background = '#38A169')
        style.configure('TPanedwindow', background = BORDER)
        self.app.configure(bg = BG)

    def setup_ui(self):
        # Header
        ttk.Label(self.main_container, text = "glycowork Analysis Suite", style = 'Header.TLabel').pack(pady = (0, 2))
        ttk.Label(self.main_container, text = "Glycoinformatics Toolkit", style = 'Sub.TLabel').pack(pady = (0, 14))
        self.app.minsize(720, 500)
        # Create main content area with sidebar and work area
        self.content = ttk.PanedWindow(self.main_container, orient=tk.HORIZONTAL)
        self.content.pack(fill=tk.BOTH, expand=True)
        # Sidebar with tools
        sidebar = ttk.Frame(self.content, style='Sidebar.TFrame')
        self.content.add(sidebar, weight=1)
        # Tool buttons
        sections = [("Draw", [("GlycoDraw", "Draw one glycan, with live preview", self.open_glyco_draw),
                              ("Batch Draw", "Draw every glycan in a spreadsheet", self.open_glyco_draw_excel)]),
                    ("Convert",
                     [("Canonicalize IUPAC", "Translate WURCS, GlycoCT, Oxford, GLYCAM, and more into IUPAC-condensed",
                       self.open_canonicalize_iupac)]),
                    ("Analyze",
                     [("Data Overview", "Coverage and PCA, to sanity-check a dataset first", self.open_data_overview),
                      ("Differential Expression", "Two-group test with volcano and MA plots",
                       self.open_differential_expression),
                      ("Heatmap", "Hierarchically clustered abundance heatmap", self.open_get_heatmap),
                      ("Lectin Array", "Map lectin binding onto glycan motifs", self.open_lectin_array)])]
        for section, items in sections:
            ttk.Label(sidebar, text = section.upper(), style = 'Sub.TLabel',
                      background = '#DDE4EE').pack(anchor = 'w', padx = 12, pady = (10, 2))
            for text, tooltip, command in items:
                btn = ttk.Button(sidebar, text = text, command = command, style = 'Tool.TButton', width = 22)
                btn.pack(pady = 3, padx = 10)
                create_tooltip(btn, tooltip)
        # Work area with log
        self.work_area = ttk.Frame(self.content)
        self.content.add(self.work_area, weight=3)

    def setup_logging(self):
        # Log view frame
        log_frame = ttk.LabelFrame(self.work_area, text="Operation Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        # Add log view
        self.log_view = ScrolledText(log_frame, height = 10, width = 50, font = ('Courier', 10),
                                     bg = '#FFFFFF', fg = '#1A202C', relief = 'flat', borderwidth = 1)
        self.log_view.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        # Configure tags for different message types
        self.log_view.tag_configure('INFO', foreground='black')
        self.log_view.tag_configure('ERROR', foreground='red')
        self.log_view.tag_configure('SUCCESS', foreground='green')
        # Add clear button below log
        btn_row = ttk.Frame(log_frame)
        btn_row.pack(pady = 5)
        ttk.Button(btn_row, text = "Open Output Folder", command = self.open_output, style = 'Tool.TButton').pack(
            side = tk.LEFT, padx = 4)
        ttk.Button(btn_row, text = "Clear Log", command = self.clear_log, style = 'Tool.TButton').pack(side = tk.LEFT,
                                                                                                       padx = 4)

    def open_output(self):
        if not self.last_output or not os.path.isdir(self.last_output):
            messagebox.showinfo("No output yet", "Run an analysis first.")
            return
        if sys.platform == 'win32':
            os.startfile(self.last_output)
        else:
            subprocess.run(['open' if sys.platform == 'darwin' else 'xdg-open', self.last_output])

    def clear_log(self):
        self.log_view.delete(1.0, tk.END)

    def log(self, message, level='INFO'):
        timestamp = time.strftime('%H:%M:%S')
        self.log_view.insert(tk.END, f"[{timestamp}] {message}\n", level)
        self.log_view.see(tk.END)

    def setup_menu(self):
        menu_bar = tk.Menu(self.app)
        self.app.config(menu=menu_bar)
        help_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about_info)

    def setup_icon(self):
        try:
            icon_path = self.resource_path("glycowork.ico")
            self.app.iconbitmap(icon_path)
        except Exception:
            pass

    @staticmethod
    def resource_path(relative_path):
        return os.path.join(getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__))), relative_path)

    def run(self):
        self.app.mainloop()

    def show_about_info(self):
        from glycowork import __version__
        about_message = f"""glycowork v{__version__}

For more information and citation, please refer to:
Thomès, L., et al. (2021). Glycowork: A Python package for glycan data science
and machine learning. Glycobiology, 31(10), 1240-1244.
DOI: 10.1093/glycob/cwab067
Or our documentation at:
https://bojarlab.github.io/glycowork/"""
        messagebox.showinfo("About glycowork", about_message)

    def open_glyco_draw(self):
        while True:
            res = GlycoDrawDialog(self.app).result
            if not res:
                break
            sequence, compact, vertical, linkage, highlight, fmt = res
            if not self.draw_folder:
                self.draw_folder = filedialog.askdirectory(title = "Select Folder to Save Glycans")
                if not self.draw_folder:
                    continue
            safe = re.sub(r'[<>:"/\\|?*]', '_', sequence)[:120]
            file_path = os.path.join(self.draw_folder, f"{safe}.{fmt}")
            try:
                GlycoDraw(sequence, filepath = file_path, compact = compact, vertical = vertical,
                          show_linkage = linkage, highlight_motif = highlight)
                self.last_output = self.draw_folder
                self.log(f"Drew {sequence} to {file_path}", 'SUCCESS')
            except Exception as e:
                self.log(f"Failed to draw {sequence}: {e}", 'ERROR')
                messagebox.showerror("Error", f"An error occurred: {e}")
            if not messagebox.askyesno("Continue", "Draw another glycan?"):
                break

    def run_task(self, title, status, fn, success_msg, out_dir = ''):
        self.last_output = out_dir
        progress = ProgressDialog(self.app, title)
        progress.update_status(status)
        outcome = queue.Queue()
        def worker():
            try:
                fn()
                outcome.put((success_msg, 'SUCCESS'))
            except Exception as e:
                outcome.put((f"{title} failed: {e}", 'ERROR'))
            finally:
                plt.close('all')
        def poll():
            try:
                msg, level = outcome.get_nowait()
            except queue.Empty:
                self.app.after(100, poll)
                return
            self.log(msg, level)
            if level == 'ERROR':
                progress.end()
                messagebox.showerror("Error", msg)
            else:
                progress.finish()
        threading.Thread(target = worker, daemon = True).start()
        self.app.after(100, poll)

    def open_glyco_draw_excel(self):
        if not (res := GlycoDrawExcelDialog(self.app).result):
            return
        csv_path, out_folder, compact = res
        self.run_task("Batch Drawing Glycans", "Processing glycans from file...",
                      lambda: plot_glycans_excel(csv_path, out_folder, compact = compact),
                      f"Drew glycans from {csv_path} to {out_folder}", out_folder)

    def open_differential_expression(self):
        if not (res := DifferentialExpressionDialog(self.app).result):
            return
        csv_path, treatment, control, motifs, plots, out_folder = res
        def analyze():
            df_out = get_differential_expression(df = csv_path, group1 = control, group2 = treatment, motifs = motifs)
            plot_glycans_excel(df_out, out_folder)
            if plots:
                get_volcano(df_out, annotate_volcano = not motifs, filepath = os.path.join(out_folder, "volcano.png"))
                get_ma(df_out, filepath = os.path.join(out_folder, "ma_plot.png"))
        self.run_task("Differential Expression Analysis", "Analyzing data...", analyze,
                      f"Analysis complete. Results saved to {out_folder}", out_folder)

    def open_data_overview(self):
        if not (res := DataOverviewDialog(self.app).result):
            return
        in_path, groups, out_dir = res
        def explore():
            get_coverage(in_path, filepath = os.path.join(out_dir, "coverage.png"))
            get_pca(in_path, groups = groups, filepath = os.path.join(out_dir, "pca.png"))
        self.run_task("Data Overview", "Profiling dataset...", explore,
                      f"Coverage and PCA plots saved to {out_dir}", out_dir)

    def open_get_heatmap(self):
        if not (res := GetHeatmapDialog(self.app).result):
            return
        in_path, motifs, clr, show_all, out_path = res
        self.run_task("Generating Heatmap", "Analyzing data...",
                      lambda: get_heatmap(df = in_path, motifs = motifs, feature_set = ["known", "exhaustive"],
                                          transform = "CLR" if clr else '', show_all = show_all, filepath = out_path),
                      f"Heatmap saved to {out_path}", os.path.dirname(out_path))

    def open_lectin_array(self):
        if not (res := LectinArrayAnalysisDialog(self.app).result):
            return
        file_path, treatment, control, paired, out_dir = res
        self.run_task("Lectin Array Analysis", "Analyzing data...",
                      lambda: plot_glycans_excel(get_lectin_array(df = file_path, group1 = control,
                                                                  group2 = treatment, paired = paired), out_dir),
                      f"Analysis complete. Results saved to {out_dir}", out_dir)

    def open_canonicalize_iupac(self):
        CanonicalizeIUPACDialog(self.app)


if __name__ == "__main__":
    gui = GlycoworkGUI()
    gui.run()
