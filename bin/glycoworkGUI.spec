# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
from PyInstaller.utils.hooks import collect_all

import glycowork
PKG = Path(glycowork.__file__).parent
ICON = Path(SPECPATH) / 'glycowork.ico'

data_files = [
    (str(PKG / 'glycan_data' / 'glycan_motifs.csv'), 'glycowork/glycan_data'),
    (str(PKG / 'glycan_data' / 'v12_lib.pkl'), 'glycowork/glycan_data'),
    (str(PKG / 'glycan_data' / 'v12_df_species.csv'), 'glycowork/glycan_data'),
    (str(PKG / 'glycan_data' / 'v12_glycan_binding.csv'), 'glycowork/glycan_data'),
    (str(PKG / 'glycan_data' / 'v12_sugarbase.json'), 'glycowork/glycan_data'),
    (str(PKG / 'glycan_data' / 'lectin_specificity.json'), 'glycowork/glycan_data'),
    (str(PKG / 'glycan_data' / 'datasets'), 'glycowork/glycan_data/datasets'),
    (str(PKG / 'motif' / 'mz_to_composition.csv'), 'glycowork/motif'),
    (str(PKG / 'motif' / 'common_names.json'), 'glycowork/motif'),
    (str(PKG / 'motif' / 'wurcs_tokens.json'), 'glycowork/motif'),
    (str(PKG / 'motif' / 'backup_gids.json'), 'glycowork/motif'),
    (str(PKG / 'motif' / 'glyconnect_to_glytoucan.json'), 'glycowork/motif'),
    (str(ICON), '.')
]

# Optional dependencies and anything the GUI provably never imports
excludes = [
    'torch',
    'torch_geometric',
    'xgboost',
    'huggingface_hub',
    'glyles',
    'pubchempy',
    'py3Dmol',
    'glycontact',
    'pytest',
    'notebook',
    'jupyter',
    'nbdev',
    'IPython',
    'bokeh',
    'PyQt5',
    'PyQt6',
    'PySide2',
    'PySide6',
    'wx',
    'pandas.tests',
    'numpy.tests',
    'scipy.io.tests'
]

gr_datas, gr_binaries, gr_hidden = collect_all('glycorender')

a = Analysis(
    ['glycoworkGUI.py'],
    pathex=[],
    binaries=gr_binaries,
    datas=data_files + gr_datas,
    hiddenimports=['matplotlib.backends.backend_agg', 'openpyxl.cell._writer', 'scipy._cyutility'] + gr_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='glycoworkGUI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ICON),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name='glycoworkGUI',
)