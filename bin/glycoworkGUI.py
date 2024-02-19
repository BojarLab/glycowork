import os
import sys
import tkinter as tk
from tkinter import simpledialog, filedialog, messagebox
from glycowork.motif.draw import GlycoDraw, plot_glycans_excel
from glycowork.motif.analysis import get_differential_expression


# Function to get the resource path within the executable environment
def resource_path(relative_path):
    """ Get the absolute path to the resource, works for dev and PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, relative_path)


class GlycoDrawDialog(simpledialog.Dialog):
    def body(self, master):
        self.title("GlycoDraw Input")
        tk.Label(master, text = "Glycan Sequence:").grid(row = 0)
        self.sequence_entry = tk.Entry(master)
        self.sequence_entry.grid(row = 0, column = 1)
        return self.sequence_entry  # to put focus on the glycan sequence entry widget

    def apply(self):
        glycan_sequence = self.sequence_entry.get()
        file_path = filedialog.asksaveasfilename(filetypes = [("PDF Image", "*.pdf")], defaultextension = ".pdf")
        if file_path:  # ensuring the user didn't cancel the save dialog
            self.result = glycan_sequence, file_path


def openGlycoDrawDialog():
    dialog_result = GlycoDrawDialog(app)
    if dialog_result.result:
        glycan_sequence, file_path = dialog_result.result
        GlycoDraw(glycan_sequence, filepath = file_path)


class GlycoDrawExcelDialog(simpledialog.Dialog):
    def body(self, master):
        self.title("GlycoDrawExcel Input")
        tk.Label(master, text="Select CSV File:").grid(row = 0)
        
        self.csv_entry = tk.Entry(master)
        self.csv_entry.grid(row = 0, column = 1)
        self.csv_button = tk.Button(master, text = "Browse...", command = self.browse_csv)
        self.csv_button.grid(row = 0, column = 2)
        
        tk.Label(master, text = "Output Folder:").grid(row = 1)
        self.folder_entry = tk.Entry(master)
        self.folder_entry.grid(row = 1, column = 1)
        self.folder_button = tk.Button(master, text = "Browse...", command = self.browse_folder)
        self.folder_button.grid(row = 1, column = 2)
        
        return self.csv_entry  # to put focus on the csv file entry widget
    
    def browse_csv(self):
        file_path = filedialog.askopenfilename(filetypes = [("CSV Files", "*.csv")])
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
        self.result = csv_file_path, output_folder


def openGlycoDrawExcelDialog():
    dialog_result = GlycoDrawExcelDialog(app)
    if dialog_result.result:
        csv_file_path, output_folder = dialog_result.result
        plot_glycans_excel(csv_file_path, output_folder)



class DifferentialExpressionDialog(simpledialog.Dialog):
    def body(self, master):
        self.title("Differential Expression Input")
        
        # CSV file selection
        tk.Label(master, text="CSV File:").grid(row = 0, sticky = tk.W)
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
        file_path = filedialog.askopenfilename(filetypes = [("CSV Files", "*.csv")])
        if file_path:
            self.csv_file_var.set(file_path)

    def browse_output_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.output_folder_var.set(folder_path)


def openDifferentialExpressionDialog():
    dialog_result = DifferentialExpressionDialog(app)
    if dialog_result.result:
        csv_file_path, treatment_indices, control_indices, motifs, output_folder = dialog_result.result
        df_out = get_differential_expression(df = csv_file_path,
                               group1 = treatment_indices,
                               group2 = control_indices,
                               motifs = motifs)
        plot_glycans_excel(df_out, output_folder)


app = tk.Tk()
app.title("glycowork GUI")
app.geometry("300x150")

btn_function1 = tk.Button(app, text = "Run GlycoDraw", command = openGlycoDrawDialog)
btn_function1.pack(pady = 5)
btn_function2 = tk.Button(app, text = "Run GlycoDrawExcel", command = openGlycoDrawExcelDialog)
btn_function2.pack(pady = 5)
btn_function3 = tk.Button(app, text="Run DifferentialExpression", command= openDifferentialExpressionDialog)
btn_function3.pack(pady = 5)

icon_path = resource_path("glycowork.ico")
app.iconbitmap(icon_path)
app.mainloop()
