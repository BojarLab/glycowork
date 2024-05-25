import os
import sys
import threading
import tkinter as tk
from tkinter import simpledialog, filedialog, messagebox, ttk
from glycowork.motif.draw import GlycoDraw, plot_glycans_excel
from glycowork.motif.analysis import get_differential_expression, get_heatmap, get_lectin_array


# Function to get the resource path within the executable environment
def resource_path(relative_path):
    """ Get the absolute path to the resource, works for dev and PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


class ProgressDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Processing")
        self.progress = ttk.Progressbar(self, orient = "horizontal", length = 300, mode = 'indeterminate')
        self.progress.pack(pady = 20)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.begin()

    def begin(self):
        self.progress.start(10)

    def end(self):
        self.progress.stop()
        self.destroy()

    def on_close(self):
        messagebox.showwarning("Warning", "Process is running. Please wait...")


class GlycoDrawDialog(simpledialog.Dialog):
    def body(self, master):
        self.title("GlycoDraw Input")
        tk.Label(master, text = "Glycan Sequence:").grid(row = 0)
        self.sequence_entry = tk.Entry(master)
        self.sequence_entry.grid(row = 0, column = 1)
        self.compact_var = tk.BooleanVar()
        self.compact_check = tk.Checkbutton(master, text = "Compact", variable = self.compact_var)
        self.compact_check.grid(row = 1, columnspan = 2)
        return self.sequence_entry  # to put focus on the glycan sequence entry widget

    def apply(self):
        glycan_sequence = self.sequence_entry.get()
        compact = self.compact_var.get()
        self.result = glycan_sequence, compact


def openGlycoDrawDialog():
    # Ask for directory only once
    folder_path = filedialog.askdirectory(title = "Select Folder to Save Glycans")
    if not folder_path:
        return  # User cancelled the folder selection

    # Continuously allow user to enter sequences and save them
    while True:
        dialog_result = GlycoDrawDialog(app)
        if dialog_result.result:
            glycan_sequence, compact = dialog_result.result
            file_path = os.path.join(folder_path, f"{glycan_sequence}.pdf")
            GlycoDraw(glycan_sequence, filepath = file_path, compact = compact)
            # Optionally, ask if the user wants to continue or not
            if not messagebox.askyesno("Continue", "Do you want to draw another glycan?"):
                break
        else:
            break  # Exit if user cancels the glycan entry dialog


class GlycoDrawExcelDialog(simpledialog.Dialog):
    def body(self, master):
        self.title("GlycoDrawExcel Input")
        tk.Label(master, text = "Select CSV or Excel File:").grid(row = 0)
        
        self.csv_entry = tk.Entry(master)
        self.csv_entry.grid(row = 0, column = 1)
        self.csv_button = tk.Button(master, text = "Browse...", command = self.browse_csv)
        self.csv_button.grid(row = 0, column = 2)
        
        tk.Label(master, text = "Output Folder:").grid(row = 1)
        self.folder_entry = tk.Entry(master)
        self.folder_entry.grid(row = 1, column = 1)
        self.folder_button = tk.Button(master, text = "Browse...", command = self.browse_folder)
        self.folder_button.grid(row = 1, column = 2)

        self.compact_var = tk.BooleanVar()
        self.compact_check = tk.Checkbutton(master, text = "Compact", variable = self.compact_var)
        self.compact_check.grid(row = 2, columnspan = 3)
        
        return self.csv_entry  # to put focus on the csv file entry widget
    
    def browse_csv(self):
        file_path = filedialog.askopenfilename(filetypes = [("CSV Files", "*.csv"), ("Excel Files", "*.xlsx")])
        if file_path:
            self.csv_entry.delete(0, tk.END)
            self.csv_entry.insert(0, file_path)
    
    def browse_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.folder_entry.delete(0, tk.END)
            self.folder_entry.insert(0, folder_path)

    def apply(self):
        csv_file_path = self.csv_entry.get()
        output_folder = self.folder_entry.get()
        compact = self.compact_var.get()
        self.result = csv_file_path, output_folder, compact


def openGlycoDrawExcelDialog():
    dialog_result = GlycoDrawExcelDialog(app)
    if dialog_result.result:
        csv_file_path, output_folder, compact = dialog_result.result
        plot_glycans_excel(csv_file_path, output_folder, compact = compact)



class DifferentialExpressionDialog(simpledialog.Dialog):
    def body(self, master):
        self.title("Differential Expression Input")
        
        # CSV file selection
        tk.Label(master, text="CSV or Excel File:").grid(row = 0, sticky = tk.W)
        self.csv_file_var = tk.StringVar(master)
        self.csv_entry = tk.Entry(master, textvariable = self.csv_file_var, state = 'readonly')
        self.csv_entry.grid(row = 0, column = 1)
        self.csv_browse = tk.Button(master, text = "Browse...", command = self.browse_csv)
        self.csv_browse.grid(row = 0, column = 2)

        # Output folder selection
        tk.Label(master, text = "Output Folder:").grid(row = 4, sticky = tk.W)
        self.output_folder_var = tk.StringVar(master)
        self.output_folder_entry = tk.Entry(master, textvariable = self.output_folder_var, state = 'readonly')
        self.output_folder_entry.grid(row = 4, column = 1)
        self.output_folder_browse = tk.Button(master, text = "Browse...", command = self.browse_output_folder)
        self.output_folder_browse.grid(row = 4, column = 2)
        
        # Treatment group indices
        tk.Label(master, text = "Treatment Group Columns:").grid(row = 1, sticky = tk.W)
        self.treatment_entry = tk.Entry(master)
        self.treatment_entry.grid(row = 1, column = 1, columnspan = 2, sticky = tk.W+tk.E)
        
        # Control group indices
        tk.Label(master, text = "Control Group Columns:").grid(row = 2, sticky = tk.W)
        self.control_entry = tk.Entry(master)
        self.control_entry.grid(row = 2, column = 1, columnspan = 2, sticky = tk.W+tk.E)
        
        # Motifs option
        tk.Label(master, text="Motif-based analysis:").grid(row = 3, sticky = tk.W)
        self.motifs_var = tk.BooleanVar(master)
        self.motifs_check = tk.Checkbutton(master, variable = self.motifs_var)
        self.motifs_check.grid(row = 3, column = 1, sticky = tk.W)
        
        return self.csv_entry  # to put focus on the csv file entry widget

    def apply(self):
        csv_file_path = self.csv_file_var.get()
        treatment_indices = self.parse_indices(self.treatment_entry.get())
        control_indices = self.parse_indices(self.control_entry.get())
        motifs = self.motifs_var.get()
        output_folder = self.output_folder_var.get()
        self.result = csv_file_path, treatment_indices, control_indices, motifs, output_folder

    def parse_indices(self, indices_str):
        try:
            # Convert comma-separated string to a list of integers
            return [int(index.strip()) for index in indices_str.split(',')]
        except ValueError:
            # Handle the case where conversion fails
            messagebox.showerror("Error", "Please enter valid, comma-separated numerical indices.")
            return []

    def browse_csv(self):
        file_path = filedialog.askopenfilename(filetypes = [("CSV Files", "*.csv"), ("Excel Files", "*.xlsx")])
        if file_path:
            self.csv_file_var.set(file_path)

    def browse_output_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.output_folder_var.set(folder_path)


def run_differential_expression(csv_file_path, treatment_indices, control_indices, motifs, output_folder, progress_dialog):
    try:
        df_out = get_differential_expression(df = csv_file_path,
                               group1 = control_indices,
                               group2 = treatment_indices,
                               motifs = motifs)
        plot_glycans_excel(df_out, output_folder)
    finally:
        progress_dialog.end()


def openDifferentialExpressionDialog():
    dialog_result = DifferentialExpressionDialog(app)
    if dialog_result.result:
        csv_file_path, treatment_indices, control_indices, motifs, output_folder = dialog_result.result
        progress_dialog = ProgressDialog(app)
        threading.Thread(target = run_differential_expression, args = (csv_file_path, treatment_indices, control_indices, motifs, output_folder, progress_dialog), daemon = True).start()


class GetHeatmapDialog(simpledialog.Dialog):
    def body(self, master):
        self.title("Get Heatmap Input")
        
        # Input file selection
        tk.Label(master, text = "Select Input CSV or Excel File:").grid(row = 0, sticky = tk.W)
        self.input_file_entry = tk.Entry(master)
        self.input_file_entry.grid(row = 0, column = 1)
        self.input_file_browse = tk.Button(master, text = "Browse...", command = self.browse_input_file)
        self.input_file_browse.grid(row = 0, column = 2)
        
        # Motif analysis option
        self.motif_analysis_var = tk.BooleanVar()
        self.motif_analysis_check = tk.Checkbutton(master, text = "Motif Analysis", variable = self.motif_analysis_var)
        self.motif_analysis_check.grid(row = 1, columnspan = 3, sticky = tk.W)
        
        # Output PDF file selection
        tk.Label(master, text = "Select Output for PDF File:").grid(row = 2, sticky = tk.W)
        self.output_file_entry = tk.Entry(master)
        self.output_file_entry.grid(row = 2, column = 1)
        self.output_file_browse = tk.Button(master, text = "Browse...", command = self.browse_output_file)
        self.output_file_browse.grid(row = 2, column = 2)

        return self.input_file_entry  # to put focus on the input file entry widget

    def browse_input_file(self):
        file_path = filedialog.askopenfilename(filetypes = [("CSV Files", "*.csv"), ("Excel Files", "*.xlsx")])
        if file_path:
            self.input_file_entry.delete(0, tk.END)
            self.input_file_entry.insert(0, file_path)

    def browse_output_file(self):
        file_path = filedialog.asksaveasfilename(filetypes = [("PDF Files", "*.pdf")], defaultextension = ".pdf")
        if file_path:
            self.output_file_entry.delete(0, tk.END)
            self.output_file_entry.insert(0, file_path)

    def apply(self):
        input_file_path = self.input_file_entry.get()
        motif_analysis = self.motif_analysis_var.get()
        output_file_path = self.output_file_entry.get()
        self.result = input_file_path, motif_analysis, output_file_path


def openGetHeatmapDialog():
    dialog_result = GetHeatmapDialog(app)
    if dialog_result.result:
        input_file_path, motif_analysis, output_file_path = dialog_result.result
        get_heatmap(input_file_path, motifs = motif_analysis, feature_set = ["known", "exhaustive"], filepath = output_file_path)


class LectinArrayAnalysisDialog(simpledialog.Dialog):
    def body(self, master):
        self.title("Lectin Array Analysis Input")
        
        # CSV or Excel file selection
        tk.Label(master, text="Select CSV or Excel File:").grid(row = 0, sticky = tk.W)
        self.file_entry = tk.Entry(master)
        self.file_entry.grid(row = 0, column = 1)
        self.file_browse = tk.Button(master, text = "Browse...", command = self.browse_file)
        self.file_browse.grid(row = 0, column = 2)
        
        # Treatment group indices
        tk.Label(master, text = "Treatment Group Columns (comma-separated):").grid(row = 1, sticky = tk.W)
        self.treatment_entry = tk.Entry(master)
        self.treatment_entry.grid(row = 1, column = 1, columnspan = 2, sticky = tk.W+tk.E)
        
        # Control group indices
        tk.Label(master, text = "Control Group Columns (comma-separated):").grid(row = 2, sticky = tk.W)
        self.control_entry = tk.Entry(master)
        self.control_entry.grid(row = 2, column = 1, columnspan = 2, sticky = tk.W+tk.E)
        
        # Paired analysis option
        tk.Label(master, text = "Paired Analysis:").grid(row = 3, sticky = tk.W)
        self.paired_var = tk.BooleanVar()
        self.paired_check = tk.Checkbutton(master, variable = self.paired_var)
        self.paired_check.grid(row = 3, column = 1, sticky = tk.W)

        # Output directory selection
        tk.Label(master, text = "Output Directory:").grid(row = 4, sticky = tk.W)
        self.output_dir_entry = tk.Entry(master)
        self.output_dir_entry.grid(row = 4, column = 1)
        self.output_dir_browse = tk.Button(master, text = "Browse...", command = self.browse_output_directory)
        self.output_dir_browse.grid(row = 4, column = 2)

        return self.file_entry  # Set focus to the file entry

    def apply(self):
        # This method processes the input when the user presses OK
        file_path = self.file_entry.get()
        treatment_indices = self.parse_indices(self.treatment_entry.get())
        control_indices = self.parse_indices(self.control_entry.get())
        paired = self.paired_var.get()
        output_directory = self.output_dir_entry.get()
        if file_path and treatment_indices and control_indices and output_directory:
            self.result = file_path, treatment_indices, control_indices, paired, output_directory
        else:
            messagebox.showerror("Error", "Please complete all fields correctly.")
            self.result = None  # Prevent dialog from closing

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes = [("CSV Files", "*.csv"), ("Excel Files", "*.xlsx")])
        if file_path:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, file_path)

    def browse_output_directory(self):
        directory = filedialog.askdirectory()
        if directory:
            self.output_dir_entry.delete(0, tk.END)
            self.output_dir_entry.insert(0, directory)

    def parse_indices(self, indices_str):
        try:
            return [int(index.strip()) for index in indices_str.split(',')]
        except ValueError:
            messagebox.showerror("Error", "Invalid indices. Please enter valid, comma-separated numerical indices.")
            return None


def openLectinArrayAnalysisDialog():
    dialog_result = LectinArrayAnalysisDialog(app)
    if dialog_result.result:
        file_path, treatment_indices, control_indices, paired, output_directory = dialog_result.result
        df_out = get_lectin_array(df = file_path, group1 = control_indices, group2 = treatment_indices, paired = paired)
        plot_glycans_excel(df_out, output_directory)


def show_about_info():
    about_message = "glycowork v1.3\n\n" \
                    "For more information and citation, please refer to:\n" \
                    "Thomès, L., et al. (2021). Glycowork: A Python package for glycan data science and machine learning. Glycobiology, 31(10), 1240-1244.\n" \
                    "DOI: 10.1093/glycob/cwab067\n" \
                    "Or our documentation at:\n" \
                    "https://bojarlab.github.io/glycowork/"
    messagebox.showinfo("About glycowork", about_message)


app = tk.Tk()
app.title("glycowork GUI")
app.geometry("300x150")

btn_function1 = tk.Button(app, text = "Run GlycoDraw", command = openGlycoDrawDialog)
btn_function1.pack(pady = 5)
btn_function2 = tk.Button(app, text = "Run GlycoDrawExcel", command = openGlycoDrawExcelDialog)
btn_function2.pack(pady = 5)
btn_function3 = tk.Button(app, text = "Run DifferentialExpression", command = openDifferentialExpressionDialog)
btn_function3.pack(pady = 5)
btn_function4 = tk.Button(app, text = "Run Get Heatmap", command = openGetHeatmapDialog)
btn_function4.pack(pady = 5)
btn_function5 = tk.Button(app, text = "Run Lectin Array Analysis", command = openLectinArrayAnalysisDialog)
btn_function5.pack(pady = 5)

menu_bar = tk.Menu(app)
app.config(menu = menu_bar)
help_menu = tk.Menu(menu_bar, tearoff = 0)
menu_bar.add_cascade(label = "Help", menu = help_menu)
help_menu.add_command(label = "About", command = show_about_info)
icon_path = resource_path("glycowork.ico")
app.iconbitmap(icon_path)

app.mainloop()
