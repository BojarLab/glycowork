import os
import sys
import threading
import tkinter as tk
from tkinter import simpledialog, filedialog, messagebox, ttk
import matplotlib
matplotlib.use('TkAgg')
from glycowork.motif.draw import GlycoDraw, plot_glycans_excel
from glycowork.motif.analysis import get_differential_expression, get_heatmap, get_lectin_array


class BaseDialog(simpledialog.Dialog):
    def create_tooltip(self, widget, text):
        def enter(event):
            self.tooltip = tk.Toplevel(widget)
            self.tooltip.wm_overrideredirect(True)
            self.tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            label = tk.Label(self.tooltip, text=text, justify='left', background="#ffffff", relief='solid', borderwidth=1,
                           font=("Arial", "8", "normal"))
            label.pack(ipadx=1)
            
        def leave(event):
            if hasattr(self, 'tooltip'):
                self.tooltip.destroy()
                
        widget.bind('<Enter>', enter)
        widget.bind('<Leave>', leave)

    def add_file_input(self, master, row, label_text, help_text=None):
        tk.Label(master, text=label_text).grid(row=row, sticky=tk.W)
        entry_var = tk.StringVar(master)
        entry = tk.Entry(master, textvariable=entry_var, state='readonly')
        entry.grid(row=row, column=1)
        browse_btn = tk.Button(master, text="Browse...", command=lambda: self.browse_file(entry_var))
        browse_btn.grid(row=row, column=2)
        
        if help_text:
            help_icon = tk.Label(master, text="?", font=("Arial", 10, "bold"), fg="blue", cursor="question_arrow")
            help_icon.grid(row=row, column=3, padx=(5, 0))
            self.create_tooltip(help_icon, help_text)
            
        return entry_var

    def add_folder_input(self, master, row, label_text):
        tk.Label(master, text=label_text).grid(row=row, sticky=tk.W)
        entry_var = tk.StringVar(master)
        entry = tk.Entry(master, textvariable=entry_var, state='readonly')
        entry.grid(row=row, column=1)
        browse_btn = tk.Button(master, text="Browse...", command=lambda: self.browse_folder(entry_var))
        browse_btn.grid(row=row, column=2)
        return entry_var

    def add_group_indices_input(self, master, row, label_text):
        tk.Label(master, text=label_text).grid(row=row, sticky=tk.W)
        entry = tk.Entry(master)
        entry.grid(row=row, column=1, columnspan=2, sticky=tk.W+tk.E)
        return entry

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

    def parse_indices(self, indices_str):
        try:
            return [int(index.strip()) for index in indices_str.split(',')]
        except ValueError:
            messagebox.showerror("Error", "Please enter valid comma-separated numerical indices.")
            return []


class GlycoDrawDialog(BaseDialog):
    def body(self, master):
        self.title("GlycoDraw Input")
        tk.Label(master, text="Glycan Sequence:").grid(row=0)
        self.sequence_entry = tk.Entry(master)
        self.sequence_entry.grid(row=0, column=1)
        self.compact_var = tk.BooleanVar()
        self.compact_check = tk.Checkbutton(master, text="Compact", variable=self.compact_var)
        self.compact_check.grid(row=1, columnspan=2)
        return self.sequence_entry

    def apply(self):
        self.result = self.sequence_entry.get(), self.compact_var.get()


class GlycoDrawExcelDialog(BaseDialog):
    def body(self, master):
        self.title("GlycoDrawExcel Input")
        help_text = ("CSV Format Help:\n\n"
                    "Glycans should be in first column (ideally in IUPAC-condensed)\n"
                    "Other columns (e.g., abundances or intensities) are permitted")
        self.csv_var = self.add_file_input(master, 0, "Select CSV or Excel File:", help_text)
        self.folder_var = self.add_folder_input(master, 1, "Output Folder:")
        self.compact_var = tk.BooleanVar()
        self.compact_check = tk.Checkbutton(master, text="Compact",  variable=self.compact_var)
        self.compact_check.grid(row=2, columnspan=3)
        return None

    def apply(self):
        self.result = (self.csv_var.get(), self.folder_var.get(), 
                      self.compact_var.get())


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


class DifferentialExpressionDialog(BaseDialog):
    def body(self, master):
        self.title("Differential Expression Input")
        help_text = ("CSV Format Help:\n\n"
                    "Glycans should be in first column (ideally in IUPAC-condensed)\n"
                    "If you do NOT analyze motifs, the glycan format does not matter at all\n"
                    "Other columns should be the abundances (each sample one column)")
        self.csv_var = self.add_file_input(master, 0, "CSV or Excel File:", help_text)
        self.output_folder_var = self.add_folder_input(master, 4, "Output Folder:")
        self.treatment_entry = self.add_group_indices_input(master, 1, "Treatment Group Columns:")
        self.control_entry = self.add_group_indices_input(master, 2, "Control Group Columns:")
        # Motifs option
        tk.Label(master, text="Motif-based analysis:").grid(row=3, sticky=tk.W)
        self.motifs_var = tk.BooleanVar(master)
        self.motifs_check = tk.Checkbutton(master, variable=self.motifs_var)
        self.motifs_check.grid(row=3, column=1, sticky=tk.W)
        return None

    def apply(self):
        self.result = (self.csv_var.get(), 
                      self.parse_indices(self.treatment_entry.get()),
                      self.parse_indices(self.control_entry.get()),
                      self.motifs_var.get(),
                      self.output_folder_var.get())


class GetHeatmapDialog(BaseDialog):
    def body(self, master):
        self.title("Get Heatmap Input")
        help_text = ("CSV Format Help:\n\n"
                    "Ideally, rows are samples and columns are glycans (but the function can deal with the opposite)\n"
                    "Glycans should be ideally in IUPAC-condensed\n"
                    "If you do NOT analyze motifs, the glycan format does not matter at all")
        self.input_file_var = self.add_file_input(master, 0, "Select Input CSV or Excel File:", help_text)
        # Checkboxes row
        self.motif_analysis_var = tk.BooleanVar()
        self.motif_analysis_check = tk.Checkbutton(master, text="Motif Analysis", variable=self.motif_analysis_var)
        self.motif_analysis_check.grid(row=1, columnspan=3, sticky=tk.W)
        self.clr_transform_var = tk.BooleanVar()
        self.clr_transform_check = tk.Checkbutton(master, text="CLR?", variable=self.clr_transform_var)
        self.clr_transform_check.grid(row=1, column=1, sticky=tk.W)
        self.show_all_var = tk.BooleanVar()
        self.show_all_check = tk.Checkbutton(master, text="Show all?", variable=self.show_all_var)
        self.show_all_check.grid(row=1, column=2, sticky=tk.W)
        self.output_file_var = self.add_folder_input(master, 2, "Select Output for Heatmap File:")
        return None

    def apply(self):
        input_file_path = self.input_file_var.get()
        output_path = self.output_file_var.get()
        if input_file_path and output_path:
            self.result = (input_file_path, 
                         self.motif_analysis_var.get(),
                         self.clr_transform_var.get(),
                         self.show_all_var.get(),
                         output_path + "/output.png")
        else:
            messagebox.showerror("Error", "Please complete all fields correctly.")
            self.result = None


class LectinArrayAnalysisDialog(BaseDialog):
    def body(self, master):
        self.title("Lectin Array Analysis Input")
        help_text = ("CSV Format Help:\n\n"
                    "Format data as samples as rows and lectins as columns (first column = sample names)\n"
                    "Have lectin names in the column names")
        self.file_var = self.add_file_input(master, 0, "Select CSV or Excel File:", help_text)
        self.treatment_entry = self.add_group_indices_input(master, 1, "Treatment Group Rows (comma-separated):")
        self.control_entry = self.add_group_indices_input(master, 2, "Control Group Rows (comma-separated):")
        # Paired analysis option
        tk.Label(master, text="Paired Analysis:").grid(row=3, sticky=tk.W)
        self.paired_var = tk.BooleanVar()
        self.paired_check = tk.Checkbutton(master, variable=self.paired_var)
        self.paired_check.grid(row=3, column=1, sticky=tk.W)
        self.output_dir_var = self.add_folder_input(master, 4, "Output Directory:")
        return None

    def apply(self):
        file_path = self.file_var.get()
        treatment_indices = self.parse_indices(self.treatment_entry.get())
        control_indices = self.parse_indices(self.control_entry.get())
        output_directory = self.output_dir_var.get()
        if file_path and treatment_indices and control_indices and output_directory:
            self.result = (file_path, treatment_indices, control_indices, self.paired_var.get(), output_directory)
        else:
            messagebox.showerror("Error", "Please complete all fields correctly.")
            self.result = None


class GlycoworkGUI:
    def __init__(self):
        self.app = tk.Tk()
        self.app.title("glycowork GUI")
        self.app.geometry("300x225")
        self.setup_ui()

    def setup_ui(self):
        buttons = [
            ("Run GlycoDraw", self.open_glyco_draw),
            ("Run GlycoDrawExcel", self.open_glyco_draw_excel),
            ("Run DifferentialExpression", self.open_differential_expression),
            ("Run Get Heatmap", self.open_get_heatmap),
            ("Run Lectin Array Analysis", self.open_lectin_array)
        ]
        
        for text, command in buttons:
            btn = tk.Button(self.app, text=text, command=command)
            btn.pack(pady=5)

        self.setup_menu()
        self.setup_icon()

    def setup_menu(self):
        menu_bar = tk.Menu(self.app)
        self.app.config(menu=menu_bar)
        help_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about_info)

    def setup_icon(self):
        icon_path = self.resource_path("glycowork.ico")
        self.app.iconbitmap(icon_path)

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
        about_message = """glycowork v1.5

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
            try:
                plot_glycans_excel(csv_file_path, output_folder, compact=compact)
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {str(e)}")

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
            try:
                transform = "CLR" if clr_transform else ''
                g = get_heatmap(df=input_file_path, motifs=motif_analysis,
                             feature_set=["known", "exhaustive"], transform=transform,
                             show_all=show_all, return_plot=True)
                fig = g.fig
                fig.savefig(output_file_path, format="png", dpi=300, bbox_inches='tight')
            except Exception as e:
                messagebox.showerror("Error Saving File", f"An error occurred: {str(e)}")

    def open_lectin_array(self):
        dialog_result = LectinArrayAnalysisDialog(self.app)
        if dialog_result.result:
            file_path, treatment_indices, control_indices, paired, output_directory = dialog_result.result
            df_out = get_lectin_array(df=file_path, group1=control_indices, group2=treatment_indices, paired=paired)
            plot_glycans_excel(df_out, output_directory)

    def run_differential_expression(self, csv_file_path, treatment_indices, control_indices, motifs, output_folder, progress_dialog):
        try:
            df_out = get_differential_expression(df=csv_file_path,
                                   group1=control_indices,
                                   group2=treatment_indices,
                                   motifs=motifs)
            plot_glycans_excel(df_out, output_folder)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
        finally:
            progress_dialog.end()


if __name__ == "__main__":
    gui = GlycoworkGUI()
    gui.run()
