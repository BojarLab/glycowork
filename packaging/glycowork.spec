# PyInstaller recipe for the glycowork desktop app (packaging/glycowork_app.py, which is not part of the glycowork package). From the repository root,
# with glycowork, PySide6, and pyinstaller installed:
#     pyinstaller packaging/glycowork.spec
# gives dist/glycowork (Windows, Linux) or dist/glycowork.app (macOS); packaging/glycowork.iss turns the Windows folder into an installer
import os
import re
import sys
from importlib.metadata import version
from PyInstaller.utils.hooks import collect_data_files, copy_metadata
# Every shipped data file by pattern, so renaming or compressing one (v12_sugarbase.json.xz, datasets as .csv.xz) needs no change here
datas = collect_data_files('glycowork') + collect_data_files('glycorender') + [(os.path.join(SPECPATH, 'glycowork.ico'), '.')]
# These read their own version from package metadata at import
for package in ('glycowork', 'glycorender'):
    datas += copy_metadata(package)
icon = os.path.join(SPECPATH, 'glycowork.icns' if sys.platform == 'darwin' else 'glycowork.ico')
a = Analysis([os.path.join(SPECPATH, 'glycowork_app.py')], datas = datas, hiddenimports = ['openpyxl.cell._writer', 'scipy._cyutility'],
             # The optional ml and chem dependencies, which the app never imports, the other Qt bindings, and test and notebook tooling
             excludes = ['torch', 'torch_geometric', 'xgboost', 'huggingface_hub', 'rdkit', 'py3Dmol', 'glycontact', 'glyles', 'pubchempy', 'tkinter', 'PyQt5',
                         'PyQt6', 'PySide2', 'IPython', 'notebook', 'jupyter_client', 'nbdev', 'pytest', 'pandas.tests', 'numpy.tests', 'scipy.io.tests'])
pyz = PYZ(a.pure)
# Windows version resource: antivirus heuristics count an executable without publisher, product and version as suspicious
version_resource = None
if sys.platform == 'win32':
    from PyInstaller.utils.win32.versioninfo import VSVersionInfo, FixedFileInfo, StringFileInfo, StringTable, StringStruct, VarFileInfo, VarStruct
    numbers = (tuple(map(int, re.findall(r'\d+', version('glycowork'))[:4])) + (0, 0, 0, 0))[:4]
    version_resource = VSVersionInfo(ffi = FixedFileInfo(filevers = numbers, prodvers = numbers),
                                     kids = [StringFileInfo([StringTable('040904B0', [StringStruct(k, v) for k, v in (
                                         ('CompanyName', 'Bojar Lab, University of Gothenburg'), ('FileDescription', 'glycowork: glycan data science'),
                                         ('FileVersion', version('glycowork')), ('InternalName', 'glycowork'), ('LegalCopyright', 'MIT License'),
                                         ('OriginalFilename', 'glycowork.exe'), ('ProductName', 'glycowork'), ('ProductVersion', version('glycowork')))])]),
                                             VarFileInfo([VarStruct('Translation', [1033, 1200])])])
exe = EXE(pyz, a.scripts, [], exclude_binaries = True, name = 'glycowork', console = False, icon = icon, version = version_resource)
coll = COLLECT(exe, a.binaries, a.datas, name = 'glycowork')
if sys.platform == 'darwin':
    app = BUNDLE(coll, name = 'glycowork.app', icon = icon, bundle_identifier = 'org.bojarlab.glycowork',
                 info_plist = {'CFBundleShortVersionString': version('glycowork'), 'NSHighResolutionCapable': True})
