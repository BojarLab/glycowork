import os
import sys
import time
import threading
import tkinter as tk
from tkinter import simpledialog, filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
import matplotlib
matplotlib.use('TkAgg')
from glycowork.motif.draw import GlycoDraw, plot_glycans_excel
from glycowork.motif.analysis import get_differential_expression, get_heatmap, get_lectin_array
from glycowork.motif.processing import canonicalize_iupac

class BaseDialog(simpledialog.Dialog):
    def __init__(self, parent, title=None):
        self.style = ttk.Style()
        self.style.configure('Modern.TButton', padding=5)
        self.style.configure('Modern.TEntry', padding=3)
        self.style.configure('Modern.TCheckbutton', padding=3)
        super().__init__(parent, title)

    def create_tooltip(self, widget, text):
        tooltip = ttk.Label(widget.winfo_toplevel(), text=text, background="#FFFFEA", relief='solid', borderwidth=1, wraplength=200)

        def show_tooltip(event):
            tooltip.lift()
            x = widget.winfo_rootx() + widget.winfo_width() + 5
            y = widget.winfo_rooty()
            tooltip.place(x=x, y=y)

        def hide_tooltip(event):
            tooltip.place_forget()

        if not hasattr(widget, '_tooltip'):
            widget._tooltip = tooltip
        widget.bind('<Enter>', show_tooltip, add='+')
        widget.bind('<Leave>', hide_tooltip, add='+')
        tooltip.bind('<Enter>', hide_tooltip, add='+')

    def add_file_input(self, master, row, label_text, help_text=None, filetypes=None):
        if filetypes is None:
            filetypes = [("CSV Files", "*.csv"), ("Excel Files", "*.xlsx")]
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
            self.create_tooltip(help_btn, help_text)
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

    def add_group_indices_input(self, master, row, label_text, help_text=None):
        frame = ttk.Frame(master)
        frame.grid(row=row, column=0, columnspan=3, sticky='ew', pady=5)
        frame.grid_columnconfigure(1, weight=1)
        ttk.Label(frame, text=label_text).grid(row=0, column=0, sticky='w', padx=(0,5))
        entry = ttk.Entry(frame)
        entry.grid(row=0, column=1, sticky='ew', padx=5)
        # Add validation
        vcmd = (frame.register(self.validate_indices), '%P')
        entry.configure(validate='key', validatecommand=vcmd)
        if help_text:  # Add help button if help text provided
            help_btn = ttk.Label(frame, text="?", cursor="question_arrow")
            help_btn.grid(row=0, column=2, padx=(5,0))
            self.create_tooltip(help_btn, help_text)
        return entry

    def validate_indices(self, value):
        if value == "": return True
        if not all(c in "0123456789, " for c in value): return False
        try:
            # Check if we can parse the indices
            self.parse_indices(value)
            return True
        except ValueError:
            return False

    def parse_indices(self, indices_str):
        try:
            return [int(index.strip()) for index in indices_str.split(',') if index.strip()]
        except ValueError:
            messagebox.showerror("Error", "Please enter valid comma-separated numerical indices.")
            return []


class GlycoDrawDialog(BaseDialog):
    def body(self, master):
        self.title("Draw Glycan")
        # Frame for sequence input
        seq_frame = ttk.LabelFrame(master, text="Glycan Sequence", padding=10)
        seq_frame.pack(fill=tk.X, padx=10, pady=5)
        self.sequence_entry = ttk.Entry(seq_frame, width=40)
        self.sequence_entry.pack(fill=tk.X, padx=5, pady=5)
        # Frame for options
        options_frame = ttk.LabelFrame(master, text="Options", padding=10)
        options_frame.pack(fill=tk.X, padx=10, pady=5)
        self.compact_var = tk.BooleanVar()
        compact_check = ttk.Checkbutton(options_frame, text="Compact Display", variable=self.compact_var)
        compact_check.pack(pady=5)
        # Recent sequences dropdown
        self.recent_sequences = self.load_recent_sequences()
        if self.recent_sequences:
            history_frame = ttk.LabelFrame(master, text="Recent Sequences", padding=10)
            history_frame.pack(fill=tk.X, padx=10, pady=5)
            history_dropdown = ttk.Combobox(history_frame, values=self.recent_sequences)
            history_dropdown.pack(fill=tk.X, pady=5)
            history_dropdown.bind('<<ComboboxSelected>>', lambda e: (self.sequence_entry.delete(0, tk.END),
                                                                     self.sequence_entry.insert(0, history_dropdown.get())))
        return self.sequence_entry

    def load_recent_sequences(self):
        if not hasattr(GlycoDrawDialog, '_recent_sequences'):
            GlycoDrawDialog._recent_sequences = []
        return GlycoDrawDialog._recent_sequences[-10:]

    def save_sequence(self, sequence):
        if not hasattr(GlycoDrawDialog, '_recent_sequences'):
            GlycoDrawDialog._recent_sequences = []
        if sequence and sequence not in GlycoDrawDialog._recent_sequences:
            GlycoDrawDialog._recent_sequences.append(sequence)
            # Keep only last 10 sequences
            GlycoDrawDialog._recent_sequences = GlycoDrawDialog._recent_sequences[-10:]

    def apply(self):
        sequence = self.sequence_entry.get()
        self.save_sequence(sequence)
        self.result = sequence, self.compact_var.get()


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
        ttk.Checkbutton(output_frame, text="Compact Display", variable=self.compact_var).pack(pady=5)
        return None

    def apply(self):
        csv_path = self.csv_var.get()
        folder_path = self.folder_var.get()
        if not csv_path:
            messagebox.showerror("Error", "Please select an input file")
            return
        if not folder_path:
            messagebox.showerror("Error", "Please select an output folder")
            return
        self.result = (csv_path, folder_path, self.compact_var.get())


class ProgressDialog(tk.Toplevel):
    def __init__(self, parent, title="Processing", determinate=False):
        super().__init__(parent)
        self.title(title)
        self.geometry("400x150")
        self.transient(parent)
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
        if not self.cancelled:
            elapsed = int(time.time() - self.start_time)
            minutes = elapsed // 60
            seconds = elapsed % 60
            self.time_var.set(f"Time elapsed: {minutes}:{seconds:02d}")
            self.after(1000, self.update_time)

    def request_cancel(self):
        if messagebox.askyesno("Cancel Operation", "Are you sure you want to cancel the operation?"):
            self.cancelled = True
            self.status_var.set("Cancelling...")
            self.cancel_btn.configure(state='disabled')
            self.end()

    def update_status(self, message, progress_value=None):
        self.status_var.set(message)
        if progress_value is not None and self.progress['mode'] == 'determinate':
            self.progress['value'] = progress_value
        self.update()

    def end(self):
        self.progress.stop()
        self.destroy()

    def finish(self, message="Operation completed successfully"):
        self.progress.stop()
        self.status_var.set(message)
        self.cancel_btn.configure(text="Close", command=self.destroy)
        self.update()
        self.after(2000, self.destroy)


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
        groups_help = ("Specify column indices (1-based) for your groups.\n" "Example: 1,2,3 for first three columns")
        self.treatment_entry = self.add_group_indices_input(groups_frame, 0, "Treatment Group Columns:", groups_help)
        self.control_entry = self.add_group_indices_input(groups_frame, 1, "Control Group Columns:", groups_help)
        # Analysis options
        options_frame = ttk.LabelFrame(master, text="Analysis Options", padding=10)
        options_frame.pack(fill=tk.X, padx=10, pady=5)
        self.motifs_var = tk.BooleanVar()
        ttk.Checkbutton(options_frame, text="Perform Motif-based Analysis", variable=self.motifs_var).pack(pady=5)
        # Output frame
        output_frame = ttk.LabelFrame(master, text="Output", padding=10)
        output_frame.pack(fill=tk.X, padx=10, pady=5)
        self.output_folder_var = self.add_folder_input(output_frame, 0, "Save Results To:")
        return None

    def apply(self):
        csv_path = self.csv_var.get()
        treatment = self.parse_indices(self.treatment_entry.get())
        control = self.parse_indices(self.control_entry.get())
        output_path = self.output_folder_var.get()
        if not csv_path:
            messagebox.showerror("Error", "Please select an input file")
            return
        if not treatment:
            messagebox.showerror("Error", "Please specify treatment group columns")
            return
        if not control:
            messagebox.showerror("Error", "Please specify control group columns")
            return
        if not output_path:
            messagebox.showerror("Error", "Please select an output folder")
            return
        self.result = (csv_path, treatment, control, self.motifs_var.get(), output_path)


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
        self.motif_analysis_var = tk.BooleanVar()
        ttk.Checkbutton(options_frame, text="Motif Analysis", variable=self.motif_analysis_var).pack(padx=5, pady=2)
        self.clr_transform_var = tk.BooleanVar()
        ttk.Checkbutton(options_frame, text="CLR Transform", variable=self.clr_transform_var).pack(padx=5, pady=2)
        self.show_all_var = tk.BooleanVar()
        ttk.Checkbutton(options_frame, text="Show All Features", variable=self.show_all_var).pack(padx=5, pady=2)
        # Output frame
        output_frame = ttk.LabelFrame(master, text="Output", padding=10)
        output_frame.pack(fill=tk.X, padx=10, pady=5)
        self.output_file_var = self.add_folder_input(output_frame, 0, "Save Heatmap To:")
        return None

    def apply(self):
        input_path = self.input_file_var.get()
        output_path = self.output_file_var.get()
        if not input_path:
            messagebox.showerror("Error", "Please select an input file")
            return
        if not output_path:
            messagebox.showerror("Error", "Please select an output location")
            return
        self.result = (input_path,
                      self.motif_analysis_var.get(),
                      self.clr_transform_var.get(),
                      self.show_all_var.get(),
                      os.path.join(output_path, "heatmap.png"))


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
        groups_help = ("Specify row indices (1-based) for your groups.\n" "Example: 1,2,3 for first three rows")
        self.treatment_entry = self.add_group_indices_input(groups_frame, 0, "Treatment Group Rows:", groups_help)
        self.control_entry = self.add_group_indices_input(groups_frame, 1, "Control Group Rows:", groups_help)
        # Analysis options
        options_frame = ttk.LabelFrame(master, text="Analysis Options", padding=10)
        options_frame.pack(fill=tk.X, padx=10, pady=5)
        self.paired_var = tk.BooleanVar()
        ttk.Checkbutton(options_frame, text="Paired Analysis", variable=self.paired_var).pack(pady=5)
        # Output frame
        output_frame = ttk.LabelFrame(master, text="Output", padding=10)
        output_frame.pack(fill=tk.X, padx=10, pady=5)
        self.output_dir_var = self.add_folder_input(output_frame, 0, "Save Results To:")
        return None

    def apply(self):
        file_path = self.file_var.get()
        treatment = self.parse_indices(self.treatment_entry.get())
        control = self.parse_indices(self.control_entry.get())
        output_path = self.output_dir_var.get()
        if not file_path:
            messagebox.showerror("Error", "Please select an input file")
            return
        if not treatment:
            messagebox.showerror("Error", "Please specify treatment group rows")
            return
        if not control:
            messagebox.showerror("Error", "Please specify control group rows")
            return
        if not output_path:
            messagebox.showerror("Error", "Please select an output directory")
            return
        self.result = (file_path, treatment, control, self.paired_var.get(), output_path)


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
    self.output_text = ScrolledText(output_frame, height=10, width=50, font=('Courier', 10), state='disabled')
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
      self.app.clipboard_clear()
      self.app.clipboard_append(output_text)
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

    def setup_styles(self):
        style = ttk.Style()
        style.configure('Header.TLabel', font=('Helvetica', 16, 'bold'))
        style.configure('Tool.TButton', padding=10)
        style.configure('Sidebar.TFrame', background='#f0f0f0')

    def setup_ui(self):
        # Header
        header = ttk.Label(self.main_container, text="glycowork Analysis Suite", style='Header.TLabel')
        header.pack(pady=(0, 20))
        # Create main content area with sidebar and work area
        self.content = ttk.PanedWindow(self.main_container, orient=tk.HORIZONTAL)
        self.content.pack(fill=tk.BOTH, expand=True)
        # Sidebar with tools
        sidebar = ttk.Frame(self.content, style='Sidebar.TFrame')
        self.content.add(sidebar, weight=1)
        # Tool buttons
        tools = [
            ("GlycoDraw", "Draw individual glycans", self.open_glyco_draw),
            ("Batch Draw", "Draw glycans from Excel/CSV", self.open_glyco_draw_excel),
            ("Canonicalize IUPAC", "Standardize glycan sequences", self.open_canonicalize_iupac),
            ("Differential Expression", "Analyze differential expression", self.open_differential_expression),
            ("Heatmap", "Generate heatmap visualization", self.open_get_heatmap),
            ("Lectin Array", "Perform lectin array analysis", self.open_lectin_array)
        ]
        for text, tooltip, command in tools:
            btn = ttk.Button(sidebar, text=text, command=command, style='Tool.TButton', width=20)
            btn.pack(pady=5, padx=10)
            self.create_tooltip(btn, tooltip)
        # Work area with log
        self.work_area = ttk.Frame(self.content)
        self.content.add(self.work_area, weight=3)

    def create_tooltip(self, widget, text):
        tooltip = ttk.Label(self.app, text=text, background="#FFFFEA", relief='solid',borderwidth=1, wraplength=200)

        def enter(event):
            x = widget.winfo_rootx() + widget.winfo_width() + 5
            y = widget.winfo_rooty()
            tooltip.place(x=x, y=y)
            tooltip.lift()

        def leave(event):
            tooltip.place_forget()

        widget.bind('<Enter>', enter)
        widget.bind('<Leave>', leave)
        tooltip.bind('<Enter>', leave)

    def setup_logging(self):
        # Log view frame
        log_frame = ttk.LabelFrame(self.work_area, text="Operation Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        # Add log view
        self.log_view = ScrolledText(log_frame, height=10, width=50, font=('Courier', 10))
        self.log_view.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        # Configure tags for different message types
        self.log_view.tag_configure('INFO', foreground='black')
        self.log_view.tag_configure('ERROR', foreground='red')
        self.log_view.tag_configure('SUCCESS', foreground='green')
        # Add clear button below log
        clear_btn = ttk.Button(log_frame, text="Clear Log", command=self.clear_log, style='Tool.TButton')
        clear_btn.pack(pady=5)

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
        try:
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")
        return os.path.join(base_path, relative_path)

    def run(self):
        self.app.mainloop()

    def show_about_info(self):
        about_message = """glycowork v1.6

For more information and citation, please refer to:
Thomès, L., et al. (2021). Glycowork: A Python package for glycan data science
and machine learning. Glycobiology, 31(10), 1240-1244.
DOI: 10.1093/glycob/cwab067
Or our documentation at:
https://bojarlab.github.io/glycowork/"""
        messagebox.showinfo("About glycowork", about_message)

    def open_glyco_draw(self):
        folder_path = filedialog.askdirectory(title="Select Folder to Save Glycans")
        if not folder_path:
            return

        while True:
            dialog_result = GlycoDrawDialog(self.app)
            if dialog_result.result:
                glycan_sequence, compact = dialog_result.result
                file_path = os.path.join(folder_path, f"{glycan_sequence}.pdf")
                try:
                    GlycoDraw(glycan_sequence, filepath=file_path, compact=compact)
                except Exception as e:
                    messagebox.showerror("Error", f"An error occurred: {str(e)}")
                if not messagebox.askyesno("Continue", "Do you want to draw another glycan?"):
                    break
            else:
                break

    def open_glyco_draw_excel(self):
        dialog_result = GlycoDrawExcelDialog(self.app)
        if dialog_result.result:
            csv_file_path, output_folder, compact = dialog_result.result
            progress = ProgressDialog(self.app, "Batch Drawing Glycans")
            try:
                progress.update_status("Processing glycans from file...")
                plot_glycans_excel(csv_file_path, output_folder, compact=compact)
                self.log(f"Successfully drew glycans from {csv_file_path} to {output_folder}", 'SUCCESS')
                progress.finish("Batch drawing completed successfully")
            except Exception as e:
                error_msg = f"Error during batch drawing: {str(e)}"
                self.log(error_msg, 'ERROR')
                messagebox.showerror("Error", error_msg)
            finally:
                if not progress.winfo_exists():
                    progress.destroy()

    def open_differential_expression(self):
        dialog_result = DifferentialExpressionDialog(self.app)
        if dialog_result.result:
            csv_file_path, treatment_indices, control_indices, motifs, output_folder = dialog_result.result
            progress_dialog = ProgressDialog(self.app)
            threading.Thread(target=self.run_differential_expression,
                           args=(csv_file_path, treatment_indices, control_indices,
                                 motifs, output_folder, progress_dialog),
                           daemon=True).start()

    def open_get_heatmap(self):
        dialog_result = GetHeatmapDialog(self.app)
        if dialog_result.result:
            input_file_path, motif_analysis, clr_transform, show_all, output_file_path = dialog_result.result
            progress = ProgressDialog(self.app, "Generating Heatmap")
            try:
                progress.update_status("Analyzing data...")
                transform = "CLR" if clr_transform else ''
                g = get_heatmap(df=input_file_path, motifs=motif_analysis,
                             feature_set=["known", "exhaustive"], transform=transform,
                             show_all=show_all, return_plot=True)
                if progress.cancelled:
                    return
                progress.update_status("Saving heatmap...")
                fig = g.fig
                fig.savefig(output_file_path, format="png", dpi=300, bbox_inches='tight')
                self.log(f"Heatmap saved to {output_file_path}", 'SUCCESS')
            except Exception as e:
                error_msg = f"Error generating heatmap: {str(e)}"
                self.log(error_msg, 'ERROR')
                messagebox.showerror("Error", error_msg)
            finally:
                progress.destroy()

    def open_lectin_array(self):
        dialog_result = LectinArrayAnalysisDialog(self.app)
        if dialog_result.result:
            file_path, treatment_indices, control_indices, paired, output_directory = dialog_result.result
            progress = ProgressDialog(self.app, "Lectin Array Analysis")
            try:
                progress.update_status("Analyzing data...")
                df_out = get_lectin_array(df=file_path, group1=control_indices, group2=treatment_indices, paired=paired)
                if progress.cancelled:
                    return
                progress.update_status("Saving results...")
                plot_glycans_excel(df_out, output_directory)
                self.log(f"Analysis complete. Results saved to {output_directory}", 'SUCCESS')
            except Exception as e:
                error_msg = f"Error during analysis: {str(e)}"
                self.log(error_msg, 'ERROR')
                messagebox.showerror("Error", error_msg)
            finally:
                progress.destroy()

    def open_canonicalize_iupac(self):
        _ = CanonicalizeIUPACDialog(self.app)

    def run_differential_expression(self, csv_file_path, treatment_indices, control_indices, motifs, output_folder, progress_dialog):
        try:
            progress_dialog.update_status("Reading input file...")
            df_out = get_differential_expression(df=csv_file_path,
                                   group1=control_indices,
                                   group2=treatment_indices,
                                   motifs=motifs)
            progress_dialog.update_status("Saving results...")
            plot_glycans_excel(df_out, output_folder)
            self.log(f"Analysis complete. Results saved to {output_folder}", 'SUCCESS')
            progress_dialog.finish("Analysis completed successfully")
        except Exception as e:
            error_msg = f"Error during analysis: {str(e)}"
            self.log(error_msg, 'ERROR')
            messagebox.showerror("Error", error_msg)
        finally:
            if not progress_dialog.winfo_exists():
                progress_dialog.destroy()


if __name__ == "__main__":
    gui = GlycoworkGUI()
    gui.run()
