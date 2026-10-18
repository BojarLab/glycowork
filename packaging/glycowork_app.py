"""glycowork desktop app: draw glycans from any notation, convert and describe many sequences at once, and analyze glycomics, glycoproteomics, and lectin array data.
Not part of the glycowork package: run it with python packaging/glycowork_app.py (needs PySide6), or build the installable app with packaging/glycowork.spec"""
import contextlib
import io
import itertools
import json
import multiprocessing as mp
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.request
import warnings
import pandas as pd
try:
    from PySide6.QtCore import Qt, QObject, Signal, QTimer, QSettings, QAbstractTableModel, QModelIndex, QSortFilterProxyModel, QRectF, QSize, QUrl
    from PySide6.QtGui import QColor, QDesktopServices, QFont, QIcon, QImage, QKeySequence, QPainter, QPalette, QPixmap, QFontDatabase, QShortcut
    from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout, QLabel, QPushButton, QToolButton,
                                   QComboBox, QCheckBox, QLineEdit, QPlainTextEdit, QTextBrowser, QListWidget, QListWidgetItem, QTreeWidget, QTreeWidgetItem,
                                   QTableView, QAbstractItemView, QSplitter, QStackedWidget, QTabWidget, QScrollArea, QFrame, QFileDialog, QMessageBox,
                                   QInputDialog, QDockWidget, QProgressBar, QStyledItemDelegate, QStyleOptionViewItem, QStyle, QMenu, QSizePolicy)
except ImportError as error:
    raise ImportError('The glycowork app needs PySide6: pip install PySide6') from error

INK, MUTED, ACCENT, ACCENT_DARK, ACCENT_LIGHT, WASH, UP, DOWN = '#1d2733', '#7a8696', '#2b5ea7', '#1f4a87', '#dce7f6', '#f3f6fa', '#c84b55', '#2d6a9f'
GROUP_COLORS = ['#2d6a9f', '#c84b55', '#3a8d5c', '#d08a2e', '#7b5aa6', '#4a9fb5', '#a35d7a', '#7f8c3a']
EXAMPLE_GLYCAN = 'Neu5Ac(a2-3)Gal(b1-4)[Fuc(a1-3)]GlcNAc(b1-2)Man(a1-3)[Neu5Gc(a2-6)Gal(b1-4)GlcNAc(b1-2)Man(a1-6)]Man(b1-4)GlcNAc(b1-4)[Fuc(a1-6)]GlcNAc'
EXAMPLE_DATASET = 'glycomics_human_colorectal_N_PMID26085185'
EMBLEM = 'Neu5Ac(a2-3)Gal(b1-4)[Fuc(a1-3)]GlcNAc'
# Feature sets for motif analyses, as quantify_motifs names them; known and exhaustive are glycowork's default pair
FEATURE_SETS = [('known', 'Known motifs'), ('exhaustive', 'Mono- and disaccharides'), ('terminal1', 'Terminal monosaccharides'), ('terminal2', 'Terminal disaccharides'),
                ('terminal3', 'Terminal trisaccharides'), ('size_branch', 'Size and branching')]
# key, label, description, groups needed (0: optional, 2: exactly two, 3: three or more), transforms (default first), data kinds it applies to
ANALYSES = [('overview', 'Overview: coverage and PCA', 'How often every glycan is detected across samples (get_coverage), and a PCA of the samples (get_pca), colored by group if '
             'groups are set. A good first look at any dataset.', 0, [('None', None), ('CLR', 'CLR'), ('ALR', 'ALR')], ('glycomics', 'glycoproteomics')),
            ('diff', 'Differential expression', "Compares two groups per glycan or motif (get_differential_expression): Welch's t-test, or a paired test, on log-ratio "
             'transformed abundances with variance moderation and Benjamini-Hochberg correction; with volcano and MA plots. Log2FC is the second group relative to '
             'the first.', 2, [('Automatic', None), ('CLR', 'CLR'), ('ALR', 'ALR')], ('glycomics', 'glycoproteomics')),
            ('anova', 'ANOVA', "Compares three or more groups per glycan or motif (get_glycanova): one-way ANOVA with omega-squared effect sizes and Tukey's post-hoc "
             'test.', 3, [('Automatic', None), ('CLR', 'CLR'), ('ALR', 'ALR')], ('glycomics', 'glycoproteomics')),
            ('heatmap', 'Clustered heatmap', 'Hierarchically clustered heatmap of glycan or motif abundances across samples (get_heatmap).', 0,
             [('CLR', 'CLR'), ('ALR', 'ALR'), ('None', '')], ('glycomics', 'glycoproteomics')),
            ('pcoa', 'PCoA with PERMANOVA', 'Principal coordinates of the sample distances (get_pcoa), and with groups a PERMANOVA test of whether they differ. '
             'Euclidean distance on CLR data is the Aitchison distance.', 0, [('CLR', 'CLR'), ('ALR', 'ALR'), ('None', '')], ('glycomics', 'glycoproteomics')),
            ('lectin', 'Lectin array', 'Maps differential lectin binding between two groups of samples onto the glycan motifs these lectins recognize '
             '(get_lectin_array).', 2, [('None', ''), ('log2', 'log2')], ('lectin',))]
CLASSES = {'O': 'O-glycan', 'N': 'N-glycan', 'free': 'Free oligosaccharide', 'lipid': 'Glycolipid', 'lipid/free': 'Glycolipid or free oligosaccharide', 'repeat': 'Repeat unit',
           'composition': 'Composition'}
DATA_KINDS = [('Glycomics: glycans × samples', 'glycomics'), ('Glycoproteomics: glycoforms × samples', 'glycoproteomics'), ('Lectin array: samples × lectins', 'lectin')]
# Columns whose sign says which way a feature changed, colored for significant rows
SIGNED = ('Log2FC', 'Effect size', 'Residual effect size', 'meandiff')
DATA_FILTER = 'Data files (*.csv *.tsv *.txt *.xlsx *.xls *.gwp *.gwa);;All files (*)'
# glycowork's modules import each other in a cycle, so two threads importing them at once (drawing, sequence facts, dataset list) can deadlock in the import system
IMPORT_LOCK = threading.Lock()
STYLE = f"""
QWidget#panel {{ background: {WASH}; }}
QLabel[role="heading"] {{ font-weight: 600; color: {INK}; padding-top: 8px; }}
QLabel[role="hint"] {{ color: {MUTED}; }}
QLabel[role="error"] {{ color: {UP}; }}
QLabel#title {{ font-size: 28pt; font-weight: 300; color: {ACCENT}; }}
QLabel#picture {{ background: white; border: 1px solid #dfe5ee; border-radius: 4px; color: {MUTED}; }}
QPushButton#run {{ background: {ACCENT}; color: white; border: none; border-radius: 4px; padding: 8px 18px; font-weight: 600; }}
QPushButton#run:hover {{ background: {ACCENT_DARK}; }}
QPushButton#run:disabled {{ background: #a9bede; }}
QHeaderView::section {{ background: {WASH}; border: none; border-bottom: 1px solid #d5deea; border-right: 1px solid #e6ebf2; padding: 4px 6px; font-weight: 600; }}
QTableView {{ gridline-color: #edf1f6; selection-background-color: {ACCENT_LIGHT}; selection-color: {INK}; }}
QListWidget, QTreeWidget {{ selection-background-color: {ACCENT_LIGHT}; selection-color: {INK}; }}
QSplitter::handle {{ background: #e3e9f1; }}
QTabWidget::pane {{ border-top: 1px solid #d5deea; }}
"""


def _worker_main(jobs, messages):
    """Reads data and runs glycowork's analyses in their own process: the window stays responsive, and Cancel ends the process instead of waiting for it"""
    class Pipe(io.TextIOBase):
        def write(self, text):
            messages.put(('log', text))
            return len(text)

    sys.stdout = sys.stderr = Pipe()
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from glycowork.glycan_data.data_entry import read_abundances
        from glycowork.motif import analysis
        from glycowork.motif.processing import GLYCOFORM_ID, is_composition
    except Exception:
        messages.put(('fatal', traceback.format_exc()))
        return
    # The plotting functions end with plt.show(), which only warns under Agg
    warnings.filterwarnings('ignore', message = '.*non-interactive.*')
    warnings.filterwarnings('ignore', message = '.*FigureCanvasAgg.*')
    glycanish = lambda x: isinstance(x, str) and ('(' in x or is_composition(x))
    messages.put(('ready', None))
    parent = mp.parent_process()
    while True:
        try:
            job = jobs.get(timeout = 1)
        except queue.Empty:
            if parent is not None and not parent.is_alive():
                return
            continue
        if job is None:
            return
        kind, payload = job
        try:
            if kind == 'load':
                raw = read_abundances(payload)
                contrasts, paired = dict(getattr(raw, '_contrasts', None) or {}), bool(getattr(raw, '_paired', False))
                provenance = {k: v for k, v in (getattr(raw, '_provenance', None) or {}).items() if isinstance(v, str) and v}
                df = pd.DataFrame(analysis._glycans_to_column(raw))
                labels = df.iloc[:, 0]
                if len(df) and all(isinstance(x, str) and GLYCOFORM_ID.fullmatch(x) for x in labels):
                    data_kind = 'glycoproteomics'
                elif sum(map(glycanish, labels)) >= len(labels) / 2:
                    data_kind = 'glycomics'
                elif df.shape[1] > 1 and sum(map(glycanish, df.columns[1:])) >= (df.shape[1] - 1) / 2:
                    # Glycans as columns and samples as rows are turned into the glycans x samples layout every analysis reads
                    df = df.set_index(df.columns[0]).T.rename_axis('glycan').reset_index()
                    data_kind = 'glycomics'
                else:
                    data_kind = 'lectin'
                name = os.path.splitext(os.path.basename(str(payload)))[0]
                messages.put(('loaded', {'name': name if os.path.exists(str(payload)) else re.sub(r'^(glycomics|glycoproteomics|lectin_array)_', '', name), 'path': payload, 'df': df, 'contrasts': contrasts,
                                         'paired': paired, 'provenance': provenance, 'kind': data_kind}))
            elif kind == 'run':
                p = payload
                df, groups, a = p['df'], p['groups'], p['analysis']
                samples, labels = [s for _, members in groups for s in members], [name for name, members in groups for _ in members]
                motif_kw = {'motifs': p['motifs'], 'feature_set': p['feature_set']} if p['motifs'] else {}
                tables, figures, out = {}, {}, {}
                if a == 'lectin':
                    # Samples are the rows of a lectin array, so the excluded ones are dropped as rows
                    rows = df[df.iloc[:, 0].isin(samples)].reset_index(drop = True)
                    tables['Lectin array'] = analysis.get_lectin_array(rows, group1 = groups[0][1], group2 = groups[1][1], paired = p['paired'], transform = p['transform'])
                else:
                    sub = df[[df.columns[0]] + samples] if samples else df
                    if a == 'overview':
                        figures['Coverage'] = ('get_coverage', {'df': sub})
                        figures['PCA'] = ('get_pca', {'df': sub, 'groups': labels or None, 'transform': p['transform'], **motif_kw})
                    elif a == 'diff':
                        res = analysis.get_differential_expression(sub, group1 = groups[0][1], group2 = groups[1][1], paired = p['paired'], transform = p['transform'],
                                                                   **motif_kw)
                        tables['Differential expression'] = res
                        # Drawings and labels of more hits than this pile up into an unreadable plot
                        hits = int(res['significant'].sum()) if 'significant' in res.columns else 0
                        figures['Volcano'] = ('get_volcano', {'df_res': res, 'annotate_volcano': p['annotate'] and not p['motifs'] and p['kind'] == 'glycomics' and hits <= 15,
                                                              'label_changed': hits <= 30})
                        figures['MA plot'] = ('get_ma', {'df_res': res})
                    elif a == 'anova':
                        res, posthoc = analysis.get_glycanova(sub, groups = labels, transform = p['transform'], **motif_kw)
                        tables['ANOVA'] = res
                        if posthoc:
                            tables['Post-hoc tests'] = pd.concat(posthoc, names = ['Glycan', None]).reset_index(level = 0).reset_index(drop = True)
                    elif a == 'heatmap':
                        figures['Heatmap'] = ('get_heatmap', {'df': sub, 'transform': p['transform'], 'show_all': p['show_all'], **motif_kw})
                    elif a == 'pcoa':
                        figures['PCoA'] = ('get_pcoa', {'df': sub, 'groups': labels or None, 'transform': p['transform'], **motif_kw})
                for n, (name, (func, kwargs)) in enumerate(figures.items()):
                    path = os.path.join(p['folder'], f'{p["stamp"]}_{n}.png')
                    result = getattr(analysis, func)(**kwargs, filepath = path)
                    plt.close('all')
                    out[name] = (path, (func, kwargs))
                    if func == 'get_pcoa':
                        # The coordinates come with the figure, and PERMANOVA and the explained variance in their attrs
                        tables['PCoA coordinates'] = result.reset_index(names = 'Sample')
                        tables['PCoA coordinates'].attrs = result.attrs
                messages.put(('result', {'tables': tables, 'figures': out}))
            elif kind == 'figure':
                (func, kwargs), path = payload
                getattr(analysis, func)(**kwargs, filepath = path)
                messages.put(('saved', path))
        except Exception:
            messages.put(('error', traceback.format_exc()))
        finally:
            plt.close('all')


class Runner(QObject):
    """Owns the worker process and relays its messages as signals"""
    ready, log, done, failed = Signal(), Signal(str), Signal(str, object), Signal(str)

    def __init__(self):
        super().__init__()
        self.context, self.open_jobs, self.dead = mp.get_context('spawn'), 0, False
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.poll)
        self.start()

    def start(self):
        self.jobs, self.messages = self.context.Queue(), self.context.Queue()
        self.process = self.context.Process(target = _worker_main, args = (self.jobs, self.messages), name = 'glycowork worker')
        self.process.start()
        self.timer.start()

    def submit(self, kind, payload):
        self.open_jobs += 1
        self.jobs.put((kind, payload))

    def cancel(self):
        self.process.terminate()
        self.process.join(5)
        self.open_jobs = 0
        self.start()

    def shutdown(self):
        self.timer.stop()
        if self.process.is_alive():
            self.jobs.put(None)
            self.process.join(1.5)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(1)

    def poll(self):
        while True:
            try:
                kind, content = self.messages.get_nowait()
            except (queue.Empty, EOFError, OSError):
                break
            if kind == 'ready':
                self.ready.emit()
            elif kind == 'log':
                self.log.emit(content)
            elif kind == 'fatal':
                self.dead = True
                self.failed.emit(content)
            else:
                self.open_jobs = max(self.open_jobs - 1, 0)
                if kind == 'error':
                    self.failed.emit(content)
                else:
                    self.done.emit(kind, content)
        if not self.process.is_alive() and not self.dead:
            self.timer.stop()
            if self.open_jobs:
                self.open_jobs = 0
                self.failed.emit('The glycowork worker stopped unexpectedly, most likely because it ran out of memory.')
            self.start()


class GlycanImages(QObject):
    """SNFG drawings from GlycoDraw, rendered in a background thread and cached by glycan and drawing options: compact ones for tables, full ones for previews"""
    ready = Signal(object)

    def __init__(self):
        super().__init__()
        self.images, self.pending, self.requests, self.order, self.large = {}, set(), queue.PriorityQueue(), itertools.count(), []
        threading.Thread(target = self.render_loop, daemon = True).start()

    def get(self, glycan, compact = True, urgent = False, **options):
        """The QImage (null if GlycoDraw cannot draw it), or None while it is still being rendered"""
        if not isinstance(glycan, str) or not glycan:
            return QImage()
        key = (glycan, compact, tuple(sorted(options.items())))
        if key in self.images:
            return self.images[key]
        if key not in self.pending or urgent:
            self.pending.add(key)
            self.requests.put((0 if urgent else 1, next(self.order), key))
        return None

    def render_loop(self):
        try:
            with IMPORT_LOCK:
                from glycowork.motif.draw import GlycoDraw
                from glycorender.render import convert_svg_to_png
        except ImportError:
            # Without the drawing stack every structure is shown as text
            GlycoDraw = convert_svg_to_png = None
        while True:
            key = self.requests.get()[2]
            if key in self.images:
                continue
            try:
                # Compact drawings are shown at up to 0.4 px per SVG unit in tables, so 0.8 stays sharp on 2x screens
                image = QImage.fromData(convert_svg_to_png(GlycoDraw(key[0], compact = key[1], suppress = True, **dict(key[2])).as_svg(), None, return_bytes = True,
                                                           scale = 0.8 if key[1] else 2))
            except Exception:
                image = QImage()
            self.images[key] = image
            if not key[1]:
                # A full N-glycan drawing takes ~2.4 MB and typing in the Draw tab makes one per pause, so only the newest 60 are kept
                self.large.append(key)
                if len(self.large) > 60:
                    old = self.large.pop(0)
                    self.images.pop(old, None)
                    self.pending.discard(old)
            self.ready.emit(key)


def _describe(text, motifs = False):
    """Facts about one glycan in any notation, as the Draw tab and the Sequences tab show them; 'error' says why nothing could be read"""
    with IMPORT_LOCK:
        from glycowork.glycan_data.loader import stringify_dict
        from glycowork.motif.processing import canonicalize_iupac, canonicalize_composition, get_class, is_composition, glytoucan_to_glycan
        from glycowork.motif.tokenization import glycan_to_composition, glycan_to_mass
        from glycowork.motif.annotate import annotate_glycan
    info = {'input': text}
    try:
        try:
            glycan = canonicalize_iupac(text)
        except ValueError:
            # canonicalize_iupac refuses Byonic-style compositions (HexNAc(4)Hex(5)) and points to canonicalize_composition
            if not is_composition(text):
                raise
            glycan = canonicalize_composition(text, as_string = True)
        info['glycan'] = glycan
        if is_composition(glycan):
            info.update(kind = 'composition', composition = canonicalize_composition(glycan, as_string = True), mass = glycan_to_mass(glycan),
                        permethylated = glycan_to_mass(glycan, sample_prep = 'permethylated'))
            return info
        # glycan_to_composition raises on what canonicalize_iupac could not read and hands back unchanged; {} is a real glycan whose residues have no composition key
        # (Kdo, heptoses, QuiNAc), which glycan_to_mass weighs atom by atom
        composition = glycan_to_composition(glycan)
        info.update(kind = get_class(glycan) or 'unclassified', composition = f'{stringify_dict(composition)} ({canonicalize_composition(composition, as_string = True)})' if composition else '',
        found = glytoucan_to_glycan(glycan, revert = True, verbose = False)
        info['glytoucan'] = found if found != glycan else ''
        if motifs:
            info['motifs'] = list(annotate_glycan(glycan, condense = True).columns)
    except Exception as error:
        info['error'] = f'{type(error).__name__}: {error}'
    return info


class SequenceInfo(QObject):
    """_describe in a background thread, only ever for the newest request, so typing never queues up stale work"""
    done = Signal(int, object)

    def __init__(self):
        super().__init__()
        self.request, self.wake, self.counter = None, threading.Event(), 0
        threading.Thread(target = self.loop, daemon = True).start()

    def submit(self, text):
        self.counter += 1
        self.request = (self.counter, text)
        self.wake.set()
        return self.counter

    def loop(self):
        while True:
            self.wake.wait()
            self.wake.clear()
            request_id, text = self.request
            info = _describe(text, motifs = True)
            with contextlib.suppress(RuntimeError):
                self.done.emit(request_id, info)


class LogStream(QObject):
    """sys.stdout and sys.stderr of the window's process (glycowork's notes and warnings), into the log from any thread"""
    text = Signal(str)

    def write(self, text):
        self.text.emit(text)
        return len(text)

    def flush(self):
        pass


def _combo(items):
    box = QComboBox()
    for label, value in items:
        box.addItem(label, value)
    return box


def _label(text, role = 'heading'):
    label = QLabel(text)
    label.setProperty('role', role)
    label.setWordWrap(role != 'heading')
    return label


def _panel():
    """A settings column: a scroll area with the panel background, and the layout to fill"""
    scroll, body = QScrollArea(), QWidget()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    scroll.setMinimumWidth(320)
    body.setObjectName('panel')
    layout = QVBoxLayout(body)
    layout.setContentsMargins(14, 8, 14, 14)
    scroll.setWidget(body)
    return scroll, layout


def _dot(color):
    pixmap = QPixmap(12, 12)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor(color))
    painter.drawEllipse(1, 1, 10, 10)
    painter.end()
    return QIcon(pixmap)


class Picture(QLabel):
    """An image scaled down to fit (and up to max_scale logical pixels per image pixel), sharp on high-DPI screens; text while there is none"""

    def __init__(self, max_scale = 0.5):
        super().__init__()
        self.image, self.max_scale = QImage(), max_scale
        self.setObjectName('picture')
        self.setAlignment(Qt.AlignCenter)
        self.setWordWrap(True)
        self.setMinimumSize(120, 80)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

    def set_image(self, image, text = ''):
        self.image = image if image is not None else QImage()
        if self.image.isNull():
            self.setPixmap(QPixmap())
            self.setText(text)
        else:
            self.setText('')
            self.rescale()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self.image.isNull():
            self.rescale()

    def rescale(self):
        ratio, width, height = self.devicePixelRatioF(), self.width() - 24, self.height() - 24
        if width < 10 or height < 10:
            return
        scale = min(width / self.image.width(), height / self.image.height(), self.max_scale)
        pixmap = QPixmap.fromImage(self.image.scaled(max(1, int(self.image.width() * scale * ratio)), max(1, int(self.image.height() * scale * ratio)),
                                                     Qt.KeepAspectRatio, Qt.SmoothTransformation))
        pixmap.setDevicePixelRatio(ratio)
        self.setPixmap(pixmap)


class FrameModel(QAbstractTableModel):
    """A DataFrame as a read-only table that sorts on raw values; a glycan column gets its SNFG drawing in an extra first column"""

    def __init__(self, images):
        super().__init__()
        self.columns, self.search, self.rows, self.strong = [], [], 0, []
        images.ready.connect(self.image_ready)

    def set_frame(self, df, glycan_column = None):
        self.beginResetModel()
        self.columns = [{'title': 'Structure', 'kind': 'structure', 'raw': [str(g) for g in df[glycan_column]], 'text': [str(g) for g in df[glycan_column]]}] if glycan_column is not None else []
        for name in df.columns:
            values = df[name]
            if pd.api.types.is_bool_dtype(values):
                self.columns.append({'title': str(name), 'kind': 'text', 'raw': [bool(v) for v in values], 'text': ['yes' if v else '' for v in values]})
            elif pd.api.types.is_numeric_dtype(values):
                raw = [None if pd.isna(v) else float(v) for v in values]
                self.columns.append({'title': str(name), 'kind': 'number', 'raw': raw, 'text': ['' if v is None else str(int(v)) if v.is_integer() and abs(v) < 1e15 else f'{v:.3g}' for v in raw]})
            else:
                text = [', '.join(map(str, v)) if isinstance(v, (list, tuple, set)) else '' if pd.isna(v) else str(v) for v in values]
                self.columns.append({'title': str(name), 'kind': 'text', 'raw': text, 'text': text})
        self.rows = len(df)
        self.search = [' '.join(c['text'][r] for c in self.columns).lower() for r in range(self.rows)]
        # Significant rows of a statistics table are set in bold, with their effects colored by direction
        self.strong = [bool(v) for v in df['significant'].tolist()] if 'significant' in df.columns and pd.api.types.is_bool_dtype(df['significant']) else [False] * self.rows
        self.endResetModel()

    def rowCount(self, parent = QModelIndex()):
        return 0 if parent.isValid() else self.rows

    def columnCount(self, parent = QModelIndex()):
        return 0 if parent.isValid() else len(self.columns)

    def headerData(self, section, orientation, role = Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self.columns[section]['title']
        return None

    def data(self, index, role = Qt.DisplayRole):
        column, row = self.columns[index.column()], index.row()
        if role == Qt.DisplayRole:
            return column['text'][row]
        if role == Qt.UserRole:
            return column['raw'][row]
        if role == Qt.ToolTipRole:
            return column['text'][row] or None
        if role == Qt.TextAlignmentRole:
            return int((Qt.AlignRight if column['kind'] == 'number' else Qt.AlignLeft) | Qt.AlignVCenter)
        if role == Qt.FontRole and self.strong[row]:
            font = QFont()
            font.setBold(True)
            return font
        if role == Qt.ForegroundRole and self.strong[row] and column['title'] in SIGNED and column['raw'][row]:
            return QColor(UP if column['raw'][row] > 0 else DOWN)
        return None

    def image_ready(self, key):
        if self.rows and self.columns and self.columns[0]['kind'] == 'structure' and key[1]:
            self.dataChanged.emit(self.index(0, 0), self.index(self.rows - 1, 0), [Qt.DisplayRole])


class TableProxy(QSortFilterProxyModel):
    """Sorts on raw values (empty cells last in ascending order) and filters on the text of every column"""

    def __init__(self):
        super().__init__()
        self.needle = ''
        self.setSortRole(Qt.UserRole)

    def set_needle(self, text):
        # Qt 6.9 replaced invalidateFilter with beginFilterChange/endFilterChange
        modern = hasattr(self, 'beginFilterChange')
        if modern:
            self.beginFilterChange()
        self.needle = text.strip().lower()
        if modern:
            self.endFilterChange()
        else:
            self.invalidateFilter()

    def filterAcceptsRow(self, row, parent):
        return not self.needle or all(word in self.sourceModel().search[row] for word in self.needle.split())

    def lessThan(self, left, right):
        a, b = left.data(Qt.UserRole), right.data(Qt.UserRole)
        if a is None or b is None:
            return a is None and b is not None
        try:
            return a < b
        except TypeError:
            return str(a) < str(b)


class StructureDelegate(QStyledItemDelegate):
    """Paints the SNFG drawing of a cell's glycan, or its text until the drawing is ready (or if GlycoDraw cannot draw it)"""

    def __init__(self, images, parent):
        super().__init__(parent)
        self.images = images

    def paint(self, painter, option, index):
        image = self.images.get(index.data(Qt.UserRole))
        if image is None or image.isNull():
            return super().paint(painter, option, index)
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        opt.text = ''
        opt.widget.style().drawControl(QStyle.CE_ItemViewItem, opt, painter, opt.widget)
        rect = opt.rect.adjusted(6, 4, -6, -4)
        scale = min(rect.width() / image.width(), rect.height() / image.height(), 0.5)
        painter.save()
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.drawImage(QRectF(rect.x(), rect.y() + (rect.height() - image.height() * scale) / 2, image.width() * scale, image.height() * scale), image)
        painter.restore()


class FrameView(QWidget):
    """A filter box and a sortable table of a DataFrame with SNFG drawings; callers add their buttons to .bar"""
    open_glycan = Signal(str)

    def __init__(self, images):
        super().__init__()
        self.df, self.glycan_column = pd.DataFrame(), None
        self.model, self.proxy = FrameModel(images), TableProxy()
        self.proxy.setSourceModel(self.model)
        self.table = QTableView()
        self.table.setModel(self.proxy)
        self.table.setSortingEnabled(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.setWordWrap(False)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setHighlightSections(False)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.delegate, self.plain = StructureDelegate(images, self.table), QStyledItemDelegate(self.table)
        self.filter = QLineEdit()
        self.filter.setPlaceholderText('Filter rows (all words must match)')
        self.filter.setClearButtonEnabled(True)
        self.count = _label('', role = 'hint')
        self.count.setWordWrap(False)
        self.bar = QHBoxLayout()
        self.bar.addWidget(self.filter, 1)
        self.bar.addWidget(self.count)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(self.bar)
        layout.addWidget(self.table, 1)
        self.filter.textChanged.connect(lambda text: (self.proxy.set_needle(text), self.update_count()))
        self.table.doubleClicked.connect(lambda index: self.open_row(self.proxy.mapToSource(index).row()))
        self.table.customContextMenuRequested.connect(self.context_menu)
        QShortcut(QKeySequence.Copy, self.table, self.copy_rows).setContext(Qt.WidgetShortcut)

    def set_frame(self, df, glycan_column = None):
        self.df, self.glycan_column = df.reset_index(drop = True), glycan_column
        self.model.set_frame(self.df, glycan_column)
        self.table.setItemDelegateForColumn(0, self.delegate if glycan_column is not None else self.plain)
        self.table.verticalHeader().setDefaultSectionSize(54 if glycan_column is not None else 24)
        self.table.sortByColumn(-1, Qt.AscendingOrder)
        self.table.resizeColumnsToContents()
        for c, column in enumerate(self.model.columns):
            self.table.setColumnWidth(c, 220 if column['kind'] == 'structure' else min(max(self.table.columnWidth(c), 60), 340))
        self.update_count()

    def update_count(self):
        self.count.setText(f'{self.proxy.rowCount()} of {self.model.rows} rows' if self.proxy.needle else f'{self.model.rows} rows')

    def glycan(self, row):
        value = self.df[self.glycan_column].iat[row] if self.glycan_column is not None and 0 <= row < len(self.df) else None
        return value if isinstance(value, str) and value else None

    def open_row(self, row):
        if self.glycan(row):
            self.open_glycan.emit(self.glycan(row))

    def selected_rows(self):
        return sorted({self.proxy.mapToSource(index).row() for index in self.table.selectionModel().selectedRows()})

    def copy_rows(self):
        rows = self.selected_rows() or list(range(len(self.df)))
        QApplication.clipboard().setText(self.df.iloc[rows].to_csv(sep = '\t', index = False))

    def context_menu(self, position):
        index = self.table.indexAt(position)
        menu = QMenu(self)
        menu.addAction('Copy selected rows', self.copy_rows)
        if index.isValid() and self.glycan(self.proxy.mapToSource(index).row()):
            glycan = self.glycan(self.proxy.mapToSource(index).row())
            menu.addAction('Copy glycan', lambda: QApplication.clipboard().setText(glycan))
            menu.addAction('Open in Draw', lambda: self.open_glycan.emit(glycan))
        menu.exec(self.table.viewport().mapToGlobal(position))


class ResultsPanel(QWidget):
    """One result table, a drawing and every value of the selected row, and the table's export"""
    open_glycan = Signal(str)

    def __init__(self, images, df, glycan_column, note, run_task, store):
        super().__init__()
        self.images, self.df, self.glycan_column, self.run_task, self.store, self.shown = images, df, glycan_column, run_task, store, None
        self.view = FrameView(images)
        excel, csv = QPushButton('Excel…'), QPushButton('CSV…')
        excel.setToolTip('Excel workbook' + (', with an SNFG drawing next to every glycan' if glycan_column else ''))
        self.view.bar.addWidget(excel)
        self.view.bar.addWidget(csv)
        self.picture = Picture(0.4)
        self.picture.setMinimumHeight(160)
        self.values = QTextBrowser()
        self.values.setOpenLinks(False)
        self.open_button = QPushButton('Open in Draw')
        self.open_button.setEnabled(False)
        detail = QWidget()
        detail_layout = QVBoxLayout(detail)
        detail_layout.setContentsMargins(6, 0, 0, 0)
        detail_layout.addWidget(self.picture, 2)
        detail_layout.addWidget(self.values, 3)
        detail_layout.addWidget(self.open_button)
        self.picture.setVisible(glycan_column is not None)
        self.open_button.setVisible(glycan_column is not None)
        split = QSplitter(Qt.Horizontal)
        split.addWidget(self.view)
        split.addWidget(detail)
        split.setStretchFactor(0, 1)
        split.setSizes([900, 340])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        if note:
            layout.addWidget(_label(note, role = 'hint'))
        layout.addWidget(split, 1)
        self.view.set_frame(df, glycan_column)
        self.view.open_glycan.connect(self.open_glycan)
        self.view.table.selectionModel().currentRowChanged.connect(lambda current: self.show_row(self.view.proxy.mapToSource(current).row()))
        self.open_button.clicked.connect(lambda: self.shown and self.open_glycan.emit(self.shown))
        excel.clicked.connect(self.export_excel)
        csv.clicked.connect(self.export_csv)
        images.ready.connect(self.image_ready)
        self.values.setHtml(f'<p style="color:{MUTED}">Select a row to see all its values here.' + (' Double-click a glycan to open it in the Draw tab.' if glycan_column else '') + '</p>')

    def show_row(self, row):
        if row < 0:
            return
        self.shown = self.view.glycan(row)
        self.open_button.setEnabled(self.shown is not None)
        if self.glycan_column is not None:
            self.picture.set_image(self.images.get(self.shown, compact = False, urgent = True), 'Drawing…' if self.shown else '')
        cells = ''.join(f'<tr><td style="color:{MUTED}; padding-right:10px">{column["title"]}</td><td>{column["text"][row]}</td></tr>'
                        for column in self.view.model.columns if column['kind'] != 'structure')
        self.values.setHtml(f'<table>{cells}</table>')

    def image_ready(self, key):
        if self.shown and key == (self.shown, False, ()):
            self.picture.set_image(self.images.images[key], f'GlycoDraw cannot draw {self.shown}')

    def export_excel(self):
        path, _ = QFileDialog.getSaveFileName(self, 'Export table', os.path.join(self.store.value('folder', os.path.expanduser('~')), 'results.xlsx'), 'Excel (*.xlsx)')
        if not path:
            return
        self.store.setValue('folder', os.path.dirname(path))
        df, column = self.df, self.glycan_column

        def work():
            if column is None:
                df.to_excel(path, index = False)
                return
            with IMPORT_LOCK:
                from glycowork.motif.draw import plot_glycans_excel
            plot_glycans_excel(df.reset_index(drop = True), path, glycan_col_num = column, compact = True)

        self.run_task(work, f'Writing {os.path.basename(path)}')

    def export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, 'Export table', os.path.join(self.store.value('folder', os.path.expanduser('~')), 'results.csv'), 'CSV (*.csv)')
        if path:
            self.store.setValue('folder', os.path.dirname(path))
            self.df.to_csv(path, index = False)


class FigureView(QWidget):
    """A figure glycowork saved, scaled to fit; saving it again as SVG or PDF draws it anew in that format"""

    def __init__(self, path, spec, save):
        super().__init__()
        self.picture = Picture(0.5)
        self.picture.set_image(QImage(path), 'The figure could not be shown')
        save_button, copy_button = QPushButton('Save figure…'), QPushButton('Copy')
        save_button.setToolTip('PNG at 300 dpi, or as vector graphics: SVG or PDF')
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(copy_button)
        row.addWidget(save_button)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self.picture, 1)
        layout.addLayout(row)
        save_button.clicked.connect(lambda: save(spec, path))
        copy_button.clicked.connect(lambda: QApplication.clipboard().setImage(self.picture.image))


class DrawTab(QWidget):
    """One glycan in any notation: a live SNFG drawing with style options, its sequence facts and known motifs, and saving or copying the drawing"""
    status = Signal(str)
    motifs_ready = Signal(object)

    def __init__(self, images, store):
        super().__init__()
        self.images, self.store, self.info, self.key = images, store, {}, None
        self.facts = SequenceInfo()
        panel, layout = _panel()
        layout.addWidget(_label('Glycan'))
        self.sequence = QPlainTextEdit()
        self.sequence.setPlaceholderText('Type or paste a glycan')
        self.sequence.setFixedHeight(100)
        self.sequence.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        layout.addWidget(self.sequence)
        hint = _label('IUPAC-condensed or -extended, WURCS, GlycoCT, Oxford, GLYCAM, LinearCode, SMILES, GlyTouCan or GlyConnect IDs, common names (LNFP I), '
                      'or compositions (H5N4F1). <a href="example">Show an example</a>', role = 'hint')
        layout.addWidget(hint)
        layout.addWidget(_label('Style'))
        self.linkages, self.compact, self.vertical, self.shadow = QCheckBox('Show linkages'), QCheckBox('Compact'), QCheckBox('Vertical'), QCheckBox('Drop shadow')
        self.linkages.setChecked(True)
        grid = QGridLayout()
        for n, box in enumerate((self.linkages, self.compact, self.vertical, self.shadow)):
            grid.addWidget(box, n // 2, n % 2)
        layout.addLayout(grid)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.highlight = QComboBox()
        self.highlight.setEditable(True)
        self.highlight.setInsertPolicy(QComboBox.NoInsert)
        self.highlight.lineEdit().setPlaceholderText('Motif name or sequence')
        self.highlight.lineEdit().setClearButtonEnabled(True)
        self.highlight.setToolTip('Draws the rest of the glycan faded: a motif of glycowork\'s motif list, or any substructure in IUPAC-condensed')
        self.end_label = QLineEdit()
        self.end_label.setPlaceholderText('e.g., Ser, 2-AB, Cer')
        self.end_label.setToolTip('Text drawn attached to the reducing end')
        form.addRow('Highlight', self.highlight)
        form.addRow('Reducing end', self.end_label)
        layout.addLayout(form)
        layout.addWidget(_label('Recent'))
        self.recent = QListWidget()
        self.recent.setIconSize(QSize(120, 40))
        self.recent.setTextElideMode(Qt.ElideRight)
        self.recent.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.recent.setMinimumHeight(180)
        self.recent.setToolTip('Every glycan you stayed on for a moment; click one to draw it again')
        layout.addWidget(self.recent, 1)
        self.picture = Picture(0.75)
        self.picture.setText('The drawing appears here as you type')
        self.message = _label('', role = 'error')
        self.message.hide()
        facts = QFormLayout()
        facts.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.fields = {}
        for key, title in (('glycan', 'IUPAC-condensed'), ('kind', 'Class'), ('composition', 'Composition'), ('mass', 'Monoisotopic mass'),
                           ('permethylated', 'Permethylated'), ('glytoucan', 'GlyTouCan ID'), ('motifs', 'Known motifs')):
            self.fields[key] = QLabel()
            self.fields[key].setWordWrap(True)
            self.fields[key].setTextInteractionFlags(Qt.TextBrowserInteraction)
            facts.addRow(_label(title, role = 'hint'), self.fields[key])
        self.fields['glycan'].setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        self.fields['glytoucan'].setOpenExternalLinks(True)
        self.fields['motifs'].setToolTip('Click a motif to highlight it')
        save, copy_image, copy_sequence = QPushButton('Save image…'), QPushButton('Copy image'), QPushButton('Copy IUPAC')
        save.setToolTip('SVG or PDF (vector graphics), or PNG')
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        for button in (copy_sequence, copy_image, save):
            buttons.addWidget(button)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(14, 12, 14, 12)
        right_layout.addWidget(self.picture, 1)
        right_layout.addWidget(self.message)
        right_layout.addLayout(facts)
        right_layout.addLayout(buttons)
        split = QSplitter(Qt.Horizontal)
        split.addWidget(panel)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        split.setSizes([360, 1000])
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(split)
        # Sequence facts once typing pauses; the glycan joins the recent list once it stays a while
        self.typing, self.settle, self.restyle = QTimer(self), QTimer(self), QTimer(self)
        for timer, ms in ((self.typing, 350), (self.settle, 2500), (self.restyle, 300)):
            timer.setSingleShot(True)
            timer.setInterval(ms)
        self.typing.timeout.connect(self.refresh)
        self.settle.timeout.connect(self.add_recent)
        self.restyle.timeout.connect(self.redraw)
        self.sequence.textChanged.connect(lambda: (self.typing.start(), self.settle.start()))
        hint.linkActivated.connect(lambda _: self.show_glycan(EXAMPLE_GLYCAN))
        for box in (self.linkages, self.compact, self.vertical, self.shadow):
            box.toggled.connect(self.redraw)
        self.highlight.currentTextChanged.connect(lambda _: self.restyle.start())
        self.end_label.textChanged.connect(lambda _: self.restyle.start())
        self.fields['motifs'].linkActivated.connect(self.highlight.setEditText)
        self.recent.itemClicked.connect(lambda item: self.show_glycan(item.data(Qt.UserRole)))
        self.facts.done.connect(self.facts_ready)
        self.images.ready.connect(self.image_ready)
        self.motifs_ready.connect(lambda names: (self.highlight.addItems([''] + names), self.highlight.setEditText('')))
        save.clicked.connect(self.save_image)
        copy_image.clicked.connect(lambda: self.picture.image.isNull() or QApplication.clipboard().setImage(self.picture.image))
        copy_sequence.clicked.connect(lambda: self.info.get('glycan') and QApplication.clipboard().setText(self.info['glycan']))
        for glycan in json.loads(self.store.value('recent', '[]')):
            self.add_recent_item(glycan)

        def motifs():
            with IMPORT_LOCK:
                from glycowork.glycan_data.loader import motif_list
            self.motifs_ready.emit(sorted(motif_list.motif_name, key = str.lower))

        threading.Thread(target = motifs, daemon = True).start()

    def show_glycan(self, glycan):
        self.sequence.setPlainText(glycan)
        self.typing.stop()
        self.refresh()

    def refresh(self):
        text = self.sequence.toPlainText().strip()
        if text:
            self.facts.submit(text)
        else:
            self.info, self.key = {}, None
            self.facts_ready(self.facts.counter, {})

    def facts_ready(self, request_id, info):
        if request_id != self.facts.counter:
            return
        self.info = info
        error = info.get('error', '')
        self.message.setText(error)
        self.message.setVisible(bool(error))
        ok = bool(info) and not error
        self.fields['glycan'].setText(info.get('glycan', '') if ok else '')
        self.fields['kind'].setText(CLASSES.get(info.get('kind'), info.get('kind', '')) if ok else '')
        self.fields['composition'].setText(info.get('composition', '') if ok else '')
        self.fields['mass'].setText(f'{info["mass"]:.4f} Da' if ok else '')
        self.fields['permethylated'].setText(f'{info["permethylated"]:.4f} Da' if ok else '')
        gid = info.get('glytoucan', '') if ok else ''
        self.fields['glytoucan'].setText(f'<a href="https://glytoucan.org/Structures/Glycans/{gid}">{gid}</a>' if gid else f'<span style="color:{MUTED}">none known</span>' if ok and info.get('kind') != 'composition' else '')
        self.fields['motifs'].setText(', '.join(f'<a href="{m}">{m}</a>' for m in info.get('motifs', [])) if ok else '')
        if ok:
            self.redraw()
        else:
            self.key = None
            self.picture.set_image(None, '' if info else 'The drawing appears here as you type')

    def options(self):
        options = {'show_linkage': self.linkages.isChecked(), 'vertical': self.vertical.isChecked(), 'shadow': self.shadow.isChecked()}
        if self.highlight.currentText().strip():
            options['highlight_motif'] = self.highlight.currentText().strip()
        if self.end_label.text().strip():
            options['reducing_end_label'] = self.end_label.text().strip()
        return options

    def redraw(self):
        if not self.info.get('glycan') or self.info.get('error'):
            return
        options = self.options()
        self.key = (self.info['glycan'], self.compact.isChecked(), tuple(sorted(options.items())))
        image = self.images.get(self.info['glycan'], self.compact.isChecked(), urgent = True, **options)
        # While a new drawing renders, the previous one stays, so typing does not flicker
        if image is not None:
            self.picture.set_image(image, 'GlycoDraw cannot draw this glycan with these settings; details are in the log (View > Log)')

    def image_ready(self, key):
        if key == self.key:
            self.picture.set_image(self.images.images[key], 'GlycoDraw cannot draw this glycan with these settings; details are in the log (View > Log)')
        if key[1] and not key[2]:
            for i in range(self.recent.count()):
                item = self.recent.item(i)
                if item.data(Qt.UserRole) == key[0] and not self.images.images[key].isNull():
                    item.setIcon(QIcon(QPixmap.fromImage(self.images.images[key])))

    def add_recent_item(self, glycan, first = False):
        item = QListWidgetItem(glycan if len(glycan) < 60 else glycan[:57] + '…')
        item.setData(Qt.UserRole, glycan)
        item.setToolTip(glycan)
        image = self.images.get(glycan)
        if image is not None and not image.isNull():
            item.setIcon(QIcon(QPixmap.fromImage(image)))
        if first:
            self.recent.insertItem(0, item)
        else:
            self.recent.addItem(item)

    def add_recent(self):
        glycan = self.info.get('glycan')
        if not glycan or self.info.get('error') or (self.recent.count() and self.recent.item(0).data(Qt.UserRole) == glycan):
            return
        for i in reversed(range(self.recent.count())):
            if self.recent.item(i).data(Qt.UserRole) == glycan:
                self.recent.takeItem(i)
        self.add_recent_item(glycan, first = True)
        while self.recent.count() > 30:
            self.recent.takeItem(self.recent.count() - 1)
        self.store.setValue('recent', json.dumps([self.recent.item(i).data(Qt.UserRole) for i in range(self.recent.count())]))

    def save_image(self):
        glycan = self.info.get('glycan')
        if not glycan or self.info.get('error'):
            self.status.emit('Type a glycan first')
            return
        name = re.sub(r'[<>:"/\\|?*\s]', '_', glycan)[:80]
        path, chosen = QFileDialog.getSaveFileName(self, 'Save drawing', os.path.join(self.store.value('folder', os.path.expanduser('~')), name),
                                                   'SVG (*.svg);;PDF (*.pdf);;PNG (*.png)')
        if not path:
            return
        if os.path.splitext(path)[1].lower() not in ('.svg', '.pdf', '.png'):
            path += '.' + chosen[:3].lower()
        self.store.setValue('folder', os.path.dirname(path))
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            with IMPORT_LOCK:
                from glycowork.motif.draw import GlycoDraw
            GlycoDraw(glycan, compact = self.compact.isChecked(), suppress = True, filepath = path, **self.options())
            self.status.emit(f'Saved {os.path.basename(path)}')
        except Exception as error:
            QMessageBox.warning(self, 'Save drawing', f'The drawing could not be saved:\n\n{error}')
        finally:
            QApplication.restoreOverrideCursor()


class SequencesTab(QWidget):
    """Many glycans in any notation at once: the canonical IUPAC-condensed sequence, class, composition, masses, GlyTouCan ID, and drawing of each"""
    open_glycan = Signal(str)
    converted = Signal(object)
    status = Signal(str)

    def __init__(self, images, run_task, store):
        super().__init__()
        self.run_task, self.store = run_task, store
        panel, layout = _panel()
        layout.addWidget(_label('Glycans'))
        self.text = QPlainTextEdit()
        self.text.setPlaceholderText('One glycan per line, in any notation the Draw tab reads')
        self.text.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        self.text.setLineWrapMode(QPlainTextEdit.NoWrap)
        layout.addWidget(self.text, 1)
        load, clear = QPushButton('Load from file…'), QPushButton('Clear')
        load.setToolTip('A column of a CSV, TSV, or Excel table, or a text file with one glycan per line')
        row = QHBoxLayout()
        row.addWidget(load)
        row.addWidget(clear)
        layout.addLayout(row)
        layout.addWidget(_label('Every sequence is converted to IUPAC-condensed with canonicalize_iupac and described as in the Draw tab. Double-click a row to '
                                'open its glycan in the Draw tab; Excel… writes the table with a drawing next to every glycan.', role = 'hint'))
        self.convert_button = QPushButton('Convert')
        self.convert_button.setObjectName('run')
        bottom = QWidget()
        bottom.setObjectName('panel')
        bottom_layout = QHBoxLayout(bottom)
        bottom_layout.setContentsMargins(14, 8, 14, 12)
        bottom_layout.addWidget(self.convert_button, 1)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)
        left_layout.addWidget(panel, 1)
        left_layout.addWidget(bottom)
        self.view = FrameView(images)
        excel, csv, copy = QPushButton('Excel…'), QPushButton('CSV…'), QPushButton('Copy IUPAC')
        for button in (copy, excel, csv):
            self.view.bar.addWidget(button)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.addWidget(self.view)
        split = QSplitter(Qt.Horizontal)
        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        split.setSizes([380, 1000])
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(split)
        load.clicked.connect(self.load_file)
        clear.clicked.connect(self.text.clear)
        self.convert_button.clicked.connect(self.convert)
        self.converted.connect(self.show_table)
        self.view.open_glycan.connect(self.open_glycan)
        excel.clicked.connect(self.export_excel)
        csv.clicked.connect(self.export_csv)
        copy.clicked.connect(lambda: QApplication.clipboard().setText('\n'.join(self.view.df.get('IUPAC-condensed', pd.Series(dtype = str)).astype(str))))
        QShortcut(QKeySequence('Ctrl+Return'), self.text, self.convert).setContext(Qt.WidgetShortcut)

    def load_file(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Load glycans', self.store.value('folder', os.path.expanduser('~')), DATA_FILTER)
        if path:
            self.add_file(path)

    def add_file(self, path):
        self.store.setValue('folder', os.path.dirname(path))
        try:
            if os.path.splitext(path)[1].lower() in ('.xlsx', '.xls', '.csv', '.tsv'):
                # glycowork's own table reader: pandas' separator sniffing split a one-column CSV at the 'y' of its glycan header, and with it every Xyl
                with IMPORT_LOCK:
                    from glycowork.glycan_data.data_entry import _read_table
                df = _read_table(path, dtype = None)[0].rename(columns = str)
            else:
                df = pd.DataFrame({'glycan': [line.strip() for line in open(path, encoding = 'utf-8-sig', errors = 'replace') if line.strip()]})
        except Exception as error:
            QMessageBox.warning(self, 'Load glycans', f'Could not read {os.path.basename(path)}:\n\n{error}')
            return
        # Text is object dtype in pandas 2 but its own str dtype in pandas 3
        columns = [c for c in df.columns if df[c].dtype == object or pd.api.types.is_string_dtype(df[c])]
        if not columns:
            QMessageBox.warning(self, 'Load glycans', f'{os.path.basename(path)} has no column of text')
            return
        column = columns[0]
        if len(columns) > 1:
            column, ok = QInputDialog.getItem(self, 'Load glycans', 'Column with the glycans:', columns, 0, False)
            if not ok:
                return
        values = [str(v).strip() for v in df[column].dropna() if str(v).strip()]
        current = self.text.toPlainText().strip()
        self.text.setPlainText('\n'.join(([current] if current else []) + values))
        self.status.emit(f'Loaded {len(values)} glycans from {os.path.basename(path)}')

    def convert(self):
        lines = [line.strip() for line in self.text.toPlainText().splitlines() if line.strip()]
        if not lines:
            self.status.emit('Enter or load glycans first')
            return
        self.convert_button.setEnabled(False)

        def work():
            rows = []
            try:
                for line in lines:
                    info = _describe(line)
                    ok = 'error' not in info
                    rows.append({'Input': line, 'IUPAC-condensed': info.get('glycan', '') if ok else '', 'Class': CLASSES.get(info.get('kind'), info.get('kind', '')) if ok else '',
                                 'Composition': info.get('composition', ''), 'Mass': info.get('mass'), 'Permethylated mass': info.get('permethylated'),
                                 'GlyTouCan ID': info.get('glytoucan', ''), 'Note': info.get('error', '')})
            finally:
                self.converted.emit(pd.DataFrame(rows, columns = ['Input', 'IUPAC-condensed', 'Class', 'Composition', 'Mass', 'Permethylated mass', 'GlyTouCan ID', 'Note']))

        self.run_task(work, f'Converting {len(lines)} glycan{"s" if len(lines) > 1 else ""}')

    def show_table(self, df):
        self.convert_button.setEnabled(True)
        self.view.set_frame(df, 'IUPAC-condensed')
        failed = int((df['Note'] != '').sum())
        if failed:
            self.status.emit(f'{failed} of {len(df)} sequences could not be read; the Note column says why')

    def export_excel(self):
        if self.view.df.empty:
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Export glycans', os.path.join(self.store.value('folder', os.path.expanduser('~')), 'glycans.xlsx'), 'Excel (*.xlsx)')
        if not path:
            return
        self.store.setValue('folder', os.path.dirname(path))
        # plot_glycans_excel puts its drawings next to the first column, so the canonical sequences go first
        df = self.view.df[['IUPAC-condensed'] + [c for c in self.view.df.columns if c != 'IUPAC-condensed']]

        def work():
            with IMPORT_LOCK:
                from glycowork.motif.draw import plot_glycans_excel
            plot_glycans_excel(df, path, compact = True)

        self.run_task(work, f'Writing {os.path.basename(path)}')

    def export_csv(self):
        if self.view.df.empty:
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Export glycans', os.path.join(self.store.value('folder', os.path.expanduser('~')), 'glycans.csv'), 'CSV (*.csv)')
        if path:
            self.store.setValue('folder', os.path.dirname(path))
            self.view.df.to_csv(path, index = False)


class AnalyzeTab(QWidget):
    """Abundance data from a file or a shipped dataset, sample groups, glycowork's analyses run in the worker process, and their tables and figures"""
    open_glycan = Signal(str)
    busy = Signal(str)
    status = Signal(str)
    datasets_ready = Signal(object)

    def __init__(self, images, runner, store, run_task):
        super().__init__()
        self.images, self.runner, self.store, self.run_task = images, runner, store, run_task
        self.data, self.names, self.assignment, self.order, self.used_groups, self.shown_analysis = None, [], {}, [], [], None
        self.folder, self.stamp = tempfile.mkdtemp(prefix = 'glycowork-app-'), itertools.count()
        panel, layout = _panel()
        layout.addWidget(_label('Data'))
        self.open_button, self.datasets = QPushButton('Open file…'), QComboBox()
        self.datasets.addItem('Shipped datasets…')
        self.datasets.setMaxVisibleItems(30)
        self.datasets.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.datasets.setMinimumContentsLength(12)
        self.datasets.setToolTip('The glycomics, glycoproteomics, and lectin array datasets that come with glycowork, with their sample groups')
        row = QHBoxLayout()
        row.addWidget(self.open_button)
        row.addWidget(self.datasets, 1)
        layout.addLayout(row)
        self.data_info = _label('Glycans in the first column, one sample per column, as CSV, TSV, or Excel; or the export of Skyline, GlycoWorkbench, LaCyTools, '
                                'pGlyco, MSFragger-Glyco, Byonic, and other tools glycowork reads. Drop files anywhere in this window.', role = 'hint')
        layout.addWidget(self.data_info)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.kind = _combo(DATA_KINDS)
        self.kind.setToolTip('Recognized from the data; change it if the guess is wrong')
        form.addRow('Data type', self.kind)
        layout.addLayout(form)
        layout.addWidget(_label('Samples and groups'))
        self.samples = QTreeWidget()
        self.samples.setHeaderLabels(['Sample', 'Group'])
        self.samples.setRootIsDecorated(False)
        self.samples.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.samples.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.samples.setContextMenuPolicy(Qt.CustomContextMenu)
        self.samples.setMinimumHeight(220)
        self.samples.setToolTip('Select samples and assign them to a group, or double-click a Group cell and type a name. Samples without a group are left out '
                                'of analyses that compare groups.')
        layout.addWidget(self.samples, 1)
        self.assign, self.guess, self.clear_groups = QToolButton(), QPushButton('Guess from names'), QPushButton('Clear')
        self.assign.setText('Assign to group')
        self.assign.setPopupMode(QToolButton.InstantPopup)
        self.assign.setMenu(QMenu(self.assign))
        self.guess.setToolTip('Groups samples whose names differ only in a trailing replicate number (healthy_1, healthy_2, ...)')
        row = QHBoxLayout()
        row.addWidget(self.assign)
        row.addWidget(self.guess)
        row.addWidget(self.clear_groups)
        layout.addLayout(row)
        self.summary, self.reverse = _label('', role = 'hint'), QToolButton()
        self.reverse.setText('⇅')
        self.reverse.setToolTip('Reverse the order of the groups; the first group is the reference of a differential expression')
        self.reverse.setAutoRaise(True)
        row = QHBoxLayout()
        row.addWidget(self.summary, 1)
        row.addWidget(self.reverse, 0, Qt.AlignTop)
        layout.addLayout(row)
        self.paired = QCheckBox('Paired samples, in the same order in each group')
        layout.addWidget(self.paired)
        layout.addWidget(_label('Analysis'))
        self.analysis = QComboBox()
        layout.addWidget(self.analysis)
        self.analysis_hint = _label('', role = 'hint')
        layout.addWidget(self.analysis_hint)
        self.motifs = QCheckBox('Analyze motifs instead of whole glycans')
        self.motifs.setToolTip('Quantifies the chosen substructures in every sample and analyzes those instead; needs IUPAC-condensed glycans')
        layout.addWidget(self.motifs)
        self.features = QWidget()
        grid = QGridLayout(self.features)
        grid.setContentsMargins(18, 0, 0, 0)
        self.feature_boxes = {}
        for n, (key, label) in enumerate(FEATURE_SETS):
            self.feature_boxes[key] = QCheckBox(label)
            self.feature_boxes[key].setChecked(key in ('known', 'exhaustive'))
            grid.addWidget(self.feature_boxes[key], n, 0)
        self.features.setVisible(False)
        layout.addWidget(self.features)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.transform = QComboBox()
        self.transform.setToolTip('Log-ratio transformation for compositional data such as relative abundances; Automatic lets glycowork choose between CLR and ALR')
        form.addRow('Transform', self.transform)
        layout.addLayout(form)
        self.annotate, self.show_all = QCheckBox('SNFG drawings in the volcano plot'), QCheckBox('Label every row and column')
        self.annotate.setChecked(True)
        self.show_all.setToolTip('Readable up to about 60 glycans or samples')
        layout.addWidget(self.annotate)
        layout.addWidget(self.show_all)
        layout.addStretch(1)
        self.run_button, self.cancel_button = QPushButton('Run analysis'), QPushButton('Cancel')
        self.run_button.setObjectName('run')
        self.cancel_button.setEnabled(False)
        bottom = QWidget()
        bottom.setObjectName('panel')
        bottom_layout = QHBoxLayout(bottom)
        bottom_layout.setContentsMargins(14, 8, 14, 12)
        bottom_layout.addWidget(self.run_button, 1)
        bottom_layout.addWidget(self.cancel_button)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)
        left_layout.addWidget(panel, 1)
        left_layout.addWidget(bottom)
        welcome = QWidget()
        welcome_layout = QVBoxLayout(welcome)
        welcome_layout.addStretch(2)
        self.emblem = QLabel()
        self.emblem.setAlignment(Qt.AlignCenter)
        title = QLabel('glycowork')
        title.setObjectName('title')
        title.setAlignment(Qt.AlignCenter)
        self.welcome_text = _label('Analyze glycomics, glycoproteomics, and lectin array data: open a file on the left or drop it anywhere here, set the sample '
                                   'groups, pick an analysis, and press Run analysis. No data at hand? <a href="example">Open an example dataset</a>.', role = 'hint')
        self.welcome_text.setAlignment(Qt.AlignCenter)
        self.welcome_text.setFixedWidth(480)
        self.welcome_text.setMinimumHeight(self.welcome_text.heightForWidth(480))
        for widget in (self.emblem, title, self.welcome_text):
            welcome_layout.addWidget(widget, 0, Qt.AlignHCenter)
        welcome_layout.addStretch(3)
        self.data_view = FrameView(images)
        data_page = QWidget()
        data_layout = QVBoxLayout(data_page)
        data_layout.setContentsMargins(8, 8, 8, 8)
        data_layout.addWidget(self.data_view)
        self.views = QTabWidget()
        self.views.setDocumentMode(True)
        self.views.addTab(data_page, 'Data')
        self.pages = QStackedWidget()
        self.pages.addWidget(welcome)
        self.pages.addWidget(self.views)
        split = QSplitter(Qt.Horizontal)
        split.addWidget(left)
        split.addWidget(self.pages)
        split.setStretchFactor(1, 1)
        split.setSizes([380, 1000])
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(split)
        self.open_button.clicked.connect(self.choose_file)
        self.datasets.activated.connect(lambda index: index > 0 and self.load(self.datasets.itemData(index)))
        self.datasets_ready.connect(self.show_datasets)
        self.welcome_text.linkActivated.connect(lambda _: self.load(EXAMPLE_DATASET))
        self.kind.currentIndexChanged.connect(lambda _: self.show_data())
        self.samples.itemDoubleClicked.connect(lambda item, column: self.samples.editItem(item, 1))
        self.samples.itemChanged.connect(lambda item, column: column == 1 and self.set_group([item.text(0)], item.text(1).strip()))
        self.samples.customContextMenuRequested.connect(lambda position: self.fill_menu(QMenu(self)).exec(self.samples.viewport().mapToGlobal(position)))
        self.assign.menu().aboutToShow.connect(lambda: self.fill_menu(self.assign.menu()))
        self.guess.clicked.connect(self.guess_groups)
        self.clear_groups.clicked.connect(lambda: self.set_groups({}, []))
        self.reverse.clicked.connect(lambda: self.set_groups(self.assignment, self.order[::-1]))
        self.analysis.currentIndexChanged.connect(lambda _: self.show_options())
        self.motifs.toggled.connect(lambda _: self.show_options())
        self.run_button.clicked.connect(self.run)
        self.cancel_button.clicked.connect(self.cancel)
        self.data_view.open_glycan.connect(self.open_glycan)
        self.runner.done.connect(self.done)
        self.runner.failed.connect(self.failed)
        self.images.ready.connect(self.image_ready)
        self.images.get(EMBLEM, compact = False, urgent = True)
        self.set_groups({}, [])
        self.show_data()

        def datasets():
            with IMPORT_LOCK:
                from glycowork.glycan_data.loader import glycomics_data_loader, glycoproteomics_data_loader, lectin_array_data_loader
            items = []
            for loader, label in ((glycomics_data_loader, 'Glycomics'), (glycoproteomics_data_loader, 'Glycoproteomics'), (lectin_array_data_loader, 'Lectin array')):
                provenance = {}
                with contextlib.suppress(Exception):
                    loader._load_contrasts()
                    provenance = loader._provenance_map
                for name in sorted(dir(loader), key = str.lower):
                    meta = provenance.get(name) or provenance.get(loader.prefix + name) or {}
                    items.append((f'{label}: {name}', loader.prefix + name, ' · '.join(str(meta[k]) for k in ('species', 'tissue', 'source_type', 'tumor_entity',
                                                                                                             'glycan_class', 'publication') if isinstance(meta.get(k), str))))
            self.datasets_ready.emit(items)

        threading.Thread(target = datasets, daemon = True).start()

    def show_datasets(self, items):
        for text, name, tip in items:
            self.datasets.addItem(text, name)
            self.datasets.setItemData(self.datasets.count() - 1, tip or name, Qt.ToolTipRole)

    def image_ready(self, key):
        if key == (EMBLEM, False, ()) and not self.images.images[key].isNull():
            pixmap = QPixmap.fromImage(self.images.images[key])
            # Run as a script, the app has no bundled .ico, so the emblem becomes its icon
            if QApplication.windowIcon().isNull():
                QApplication.setWindowIcon(QIcon(pixmap))
            pixmap.setDevicePixelRatio(2.4)
            self.emblem.setPixmap(pixmap)

    def choose_file(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Open data', self.store.value('folder', os.path.expanduser('~')), DATA_FILTER)
        if path:
            self.store.setValue('folder', os.path.dirname(path))
            self.load(path)

    def load(self, path):
        self.datasets.setCurrentIndex(0)
        self.runner.submit('load', path)
        self.set_busy(f'Reading {os.path.basename(str(path)).removeprefix("glycomics_")}')

    def set_busy(self, text):
        self.run_button.setEnabled(not text)
        self.cancel_button.setEnabled(bool(text))
        self.busy.emit(text)

    def show_data(self):
        """Lists the samples of the data as the chosen data type has them (columns, or the rows of a lectin array), with the analyses for that type"""
        kind = self.kind.currentData()
        df = self.data['df'] if self.data else None
        names = ([str(x) for x in df.iloc[:, 0]] if kind == 'lectin' else [str(c) for c in df.columns[1:]]) if df is not None else []
        if names != self.names:
            self.names = names
            contrasts = self.data['contrasts'] if self.data else {}
            self.samples.blockSignals(True)
            self.samples.clear()
            for name in names:
                item = QTreeWidgetItem([name, ''])
                item.setFlags(item.flags() | Qt.ItemIsEditable)
                self.samples.addTopLevelItem(item)
            self.samples.blockSignals(False)
            self.paired.setChecked(bool(self.data and self.data['paired']))
            self.set_groups({s: g for s, g in contrasts.items() if s in names}, list(dict.fromkeys(g for s, g in contrasts.items() if s in names)))
        current = self.analysis.currentData()
        self.analysis.blockSignals(True)
        self.analysis.clear()
        for key, label, _, _, _, kinds in ANALYSES:
            if kind in kinds:
                self.analysis.addItem(label, key)
        self.analysis.setCurrentIndex(max(self.analysis.findData(current), 0))
        self.analysis.blockSignals(False)
        self.show_options()
        if df is not None:
            self.data_view.set_frame(df, df.columns[0] if kind == 'glycomics' else None)

    def show_options(self):
        key = self.analysis.currentData()
        spec = next((a for a in ANALYSES if a[0] == key), None)
        if spec is None:
            return
        self.analysis_hint.setText(spec[2])
        # Every analysis starts out with its own default transform
        if key != self.shown_analysis:
            self.shown_analysis = key
            self.transform.clear()
            for label, value in spec[4]:
                self.transform.addItem(label, value)
        self.motifs.setVisible(key != 'lectin')
        self.features.setVisible(key != 'lectin' and self.motifs.isChecked())
        self.annotate.setVisible(key == 'diff' and self.kind.currentData() == 'glycomics')
        self.annotate.setEnabled(not self.motifs.isChecked())
        self.show_all.setVisible(key == 'heatmap')
        self.paired.setVisible(key in ('diff', 'lectin'))

    def set_groups(self, assignment, order):
        """Shows sample groups: assignment {sample: group}, order the groups' order, of which unused ones are dropped"""
        self.assignment = {s: g for s, g in assignment.items() if g}
        self.order = [g for g in dict.fromkeys(list(order) + list(self.assignment.values())) if g in self.assignment.values()]
        self.samples.blockSignals(True)
        for i in range(self.samples.topLevelItemCount()):
            item = self.samples.topLevelItem(i)
            group = self.assignment.get(item.text(0), '')
            item.setText(1, group)
            item.setIcon(1, _dot(GROUP_COLORS[self.order.index(group) % len(GROUP_COLORS)]) if group else QIcon())
        self.samples.blockSignals(False)
        self.samples.resizeColumnToContents(0)
        counts = {g: list(self.assignment.values()).count(g) for g in self.order}
        text = ' → '.join(f'{g} ({n})' for g, n in counts.items())
        if len(self.order) == 2:
            text += f'. {self.order[1]} is compared against {self.order[0]}, so a positive Log2FC means higher in {self.order[1]}.'
        left_out = len(self.names) - len(self.assignment)
        if self.order and left_out:
            text += f' {left_out} sample{"s" if left_out > 1 else ""} without a group {"are" if left_out > 1 else "is"} left out.'
        self.summary.setText(text if self.order else 'No groups yet: assign samples to groups to compare them.')
        self.reverse.setVisible(len(self.order) > 1)

    def set_group(self, samples, group):
        self.set_groups({**self.assignment, **{s: group for s in samples}}, self.order)

    def fill_menu(self, menu):
        menu.clear()
        selected = [item.text(0) for item in self.samples.selectedItems()]
        if not selected:
            menu.addAction('Select samples first').setEnabled(False)
            return menu
        for n, group in enumerate(self.order):
            menu.addAction(_dot(GROUP_COLORS[n % len(GROUP_COLORS)]), group, lambda g = group: self.set_group(selected, g))
        menu.addAction('New group…', lambda: self.new_group(selected))
        menu.addSeparator()
        menu.addAction('No group (leave out)', lambda: self.set_group(selected, ''))
        return menu

    def new_group(self, samples):
        name, ok = QInputDialog.getText(self, 'New group', f'Name of the group of {len(samples)} sample{"s" if len(samples) > 1 else ""}:',
                                        text = f'Group {len(self.order) + 1}')
        if ok and name.strip():
            self.set_group(samples, name.strip())

    def guess_groups(self):
        guess = {s: re.sub(r'[_\-. ]*\d+[A-Za-z]?$', '', s) or s for s in self.names}
        if len(set(guess.values())) in (1, len(self.names)):
            self.status.emit('The sample names do not tell groups apart by a trailing replicate number; assign the groups by hand')
            return
        self.set_groups(guess, [])

    def run(self):
        if self.data is None:
            self.status.emit('Open data first')
            return
        key = self.analysis.currentData()
        _, label, _, need, _, _ = next(a for a in ANALYSES if a[0] == key)
        groups = [(g, [s for s in self.names if self.assignment.get(s) == g]) for g in self.order]
        problem = ''
        if need == 2 and len(groups) != 2:
            problem = f'{label} compares exactly two groups, and {len(groups)} {"is" if len(groups) == 1 else "are"} set.'
        elif need == 3 and len(groups) < 3:
            problem = f'{label} compares three or more groups, and {len(groups)} {"is" if len(groups) == 1 else "are"} set; for two groups, use differential expression.'
        elif need and min(len(m) for _, m in groups) < 2:
            problem = 'Every group needs at least two samples to estimate its variance.'
        elif need == 2 and self.paired.isChecked() and len(groups[0][1]) != len(groups[1][1]):
            problem = 'Paired samples need groups of the same size, in matching order.'
        elif self.motifs.isChecked() and key != 'lectin' and not any(box.isChecked() for box in self.feature_boxes.values()):
            problem = 'Choose at least one kind of motif to analyze.'
        if problem:
            QMessageBox.information(self, label, problem + ('' if 'motif' in problem else '\n\nSelect samples and use Assign to group, or double-click a Group '
                                                                                           'cell and type a group name.'))
            return
        self.used_groups = [g for g, _ in groups]
        self.runner.submit('run', {'analysis': key, 'df': self.data['df'], 'kind': self.kind.currentData(), 'groups': groups, 'paired': self.paired.isChecked(),
                                   'motifs': self.motifs.isChecked() and key != 'lectin', 'feature_set': [k for k, box in self.feature_boxes.items() if box.isChecked()],
                                   'transform': self.transform.currentData(), 'annotate': self.annotate.isChecked(), 'show_all': self.show_all.isChecked(),
                                   'folder': self.folder, 'stamp': next(self.stamp)})
        self.running = label
        self.set_busy(f'Running {label.lower() if not label.startswith(("PCoA", "ANOVA")) else label}')

    def cancel(self):
        self.runner.cancel()
        self.set_busy('')
        self.status.emit('Canceled')

    def done(self, kind, content):
        if not self.runner.open_jobs:
            self.set_busy('')
        if kind == 'loaded':
            self.data = content
            self.names = []
            self.kind.blockSignals(True)
            self.kind.setCurrentIndex(max(self.kind.findData(content['kind']), 0))
            self.kind.blockSignals(False)
            self.show_data()
            df, prov = content['df'], content['provenance']
            what = {'glycomics': ('glycans', 'samples'), 'glycoproteomics': ('glycoforms', 'samples'), 'lectin': ('samples', 'lectins')}[content['kind']]
            details = ' · '.join(prov[k] for k in ('species', 'tissue', 'tumor_entity', 'publication') if k in prov)
            self.data_info.setText(f'<b>{content["name"]}</b>: {df.shape[0]} {what[0]} × {df.shape[1] - 1} {what[1]}' + (f'<br>{details}' if details else ''))
            while self.views.count() > 1:
                widget = self.views.widget(1)
                self.views.removeTab(1)
                widget.deleteLater()
            self.pages.setCurrentWidget(self.views)
            self.views.setCurrentIndex(0)
            self.status.emit(f'Read {content["name"]}' + (f' with {len(self.order)} sample groups' if self.order else ''))
        elif kind == 'result':
            self.show_results(content)
        elif kind == 'saved':
            self.status.emit(f'Saved {os.path.basename(content)}')

    def failed(self, text):
        if not self.runner.open_jobs:
            self.set_busy('')
        QMessageBox.warning(self, 'glycowork', f'{text.strip().splitlines()[-1]}\n\nThe full error is in the log (View > Log).')

    def show_results(self, content):
        while self.views.count() > 1:
            widget = self.views.widget(1)
            self.views.removeTab(1)
            widget.deleteLater()
        drawable = self.kind.currentData() != 'glycoproteomics'
        for name, df in content['tables'].items():
            attrs, notes = df.attrs, []
            if name == 'Differential expression' and len(self.used_groups) == 2:
                notes.append(f'Log2FC: {self.used_groups[1]} relative to {self.used_groups[0]}')
            if attrs.get('test'):
                notes.append(str(attrs['test']))
            if attrs.get('transform'):
                notes.append(f'{attrs["transform"]} transformed')
            if 'alpha' in attrs and 'significant' in df.columns:
                notes.append(f'significance threshold {float(attrs["alpha"]):.3g} for the corrected p-values (adjusted to the sample size)')
            if 'significant' in df.columns and pd.api.types.is_bool_dtype(df['significant']):
                notes.append(f'{int(df["significant"].sum())} of {len(df)} significant')
            if 'permanova' in attrs:
                notes.append(f'PERMANOVA pseudo-F {float(attrs["permanova"][0]):.3g}, p = {float(attrs["permanova"][1]):.3g}')
            if 'explained_variance' in attrs and len(attrs['explained_variance']) > 1:
                notes.append(f'PCo1 explains {attrs["explained_variance"][0]:.0%} and PCo2 {attrs["explained_variance"][1]:.0%} of the variance')
            column = next((c for c in ('Glycan', 'motif') if c in df.columns), None) if drawable else None
            panel = ResultsPanel(self.images, df, column, ' · '.join(notes), self.run_task, self.store)
            panel.open_glycan.connect(self.open_glycan)
            self.views.addTab(panel, name)
        for name, (path, spec) in content['figures'].items():
            self.views.addTab(FigureView(path, spec, self.save_figure), name)
        self.views.setCurrentIndex(1 if self.views.count() > 1 else 0)
        self.status.emit(f'{self.running} done')

    def save_figure(self, spec, png):
        path, chosen = QFileDialog.getSaveFileName(self, 'Save figure', os.path.join(self.store.value('folder', os.path.expanduser('~')), 'figure.svg'),
                                                   'SVG (*.svg);;PDF (*.pdf);;PNG, 300 dpi (*.png)')
        if not path:
            return
        if os.path.splitext(path)[1].lower() not in ('.svg', '.pdf', '.png'):
            path += '.' + chosen[:3].lower()
        self.store.setValue('folder', os.path.dirname(path))
        if path.lower().endswith('.png'):
            shutil.copyfile(png, path)
            self.status.emit(f'Saved {os.path.basename(path)}')
        else:
            # Vector formats are drawn anew from the same call, in the worker
            self.runner.submit('figure', (spec, path))
            self.set_busy(f'Saving {os.path.basename(path)}')


class MainWindow(QMainWindow):
    task_finished = Signal(object, str, str)
    update_found = Signal(str)
    update_progress = Signal(int)
    update_downloaded = Signal(bool)

    def __init__(self, paths):
        super().__init__()
        self.store = QSettings()
        self.setWindowTitle('glycowork')
        self.setAcceptDrops(True)
        self.images, self.runner, self.activities, self.since, self.tasks = GlycanImages(), Runner(), {}, None, itertools.count()
        self.draw = DrawTab(self.images, self.store)
        self.sequences = SequencesTab(self.images, self.run_task, self.store)
        self.analyze = AnalyzeTab(self.images, self.runner, self.store, self.run_task)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.addTab(self.draw, 'Draw')
        self.tabs.addTab(self.sequences, 'Sequences')
        self.tabs.addTab(self.analyze, 'Analyze')
        self.setCentralWidget(self.tabs)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        self.log.setMaximumBlockCount(20000)
        self.log_dock = QDockWidget('Log', self)
        self.log_dock.setObjectName('log')
        self.log_dock.setWidget(self.log)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.log_dock)
        self.log_dock.hide()
        self.status_text, self.elapsed, self.progress = QLabel('Starting glycowork…'), QLabel(), QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setMaximumWidth(200)
        self.progress.setMaximumHeight(14)
        self.progress.setTextVisible(False)
        self.progress.hide()
        self.statusBar().addWidget(self.status_text, 1)
        self.statusBar().addPermanentWidget(self.elapsed)
        self.statusBar().addPermanentWidget(self.progress)
        self.update_button, self.update_url, self.update_path = QPushButton(), None, None
        self.update_button.setFlat(True)
        self.update_button.setStyleSheet(f'color: {ACCENT}; font-weight: 600;')
        self.update_button.hide()
        self.statusBar().addPermanentWidget(self.update_button)
        self.clock = QTimer(self)
        self.clock.setInterval(1000)
        self.clock.timeout.connect(self.tick)
        file_menu = self.menuBar().addMenu('&File')
        file_menu.addAction('Open data…', lambda: (self.tabs.setCurrentWidget(self.analyze), self.analyze.choose_file()), QKeySequence.Open)
        file_menu.addAction('Open example dataset', lambda: (self.tabs.setCurrentWidget(self.analyze), self.analyze.load(EXAMPLE_DATASET)))
        file_menu.addAction('Load glycans into Sequences…', lambda: (self.tabs.setCurrentWidget(self.sequences), self.sequences.load_file()))
        file_menu.addSeparator()
        file_menu.addAction('Save drawing…', lambda: (self.tabs.setCurrentWidget(self.draw), self.draw.save_image()), QKeySequence.Save)
        file_menu.addSeparator()
        file_menu.addAction('Quit', self.close, QKeySequence.Quit)
        view_menu = self.menuBar().addMenu('&View')
        view_menu.addAction(self.log_dock.toggleViewAction())
        help_menu = self.menuBar().addMenu('&Help')
        help_menu.addAction('Documentation', lambda: QDesktopServices.openUrl(QUrl('https://bojarlab.github.io/glycowork/')))
        help_menu.addAction('glycowork on GitHub', lambda: QDesktopServices.openUrl(QUrl('https://github.com/BojarLab/glycowork')))
        help_menu.addAction('Report a problem…', lambda: QDesktopServices.openUrl(QUrl('https://github.com/BojarLab/glycowork/issues/new')))
        help_menu.addSeparator()
        help_menu.addAction('About glycowork', self.about)
        for tab in (self.draw, self.sequences, self.analyze):
            tab.status.connect(lambda text: self.statusBar().showMessage(text, 8000))
        for tab in (self.sequences, self.analyze):
            tab.open_glycan.connect(lambda glycan: (self.draw.show_glycan(glycan), self.tabs.setCurrentWidget(self.draw)))
        self.analyze.busy.connect(lambda text: self.set_activity('worker', text))
        self.runner.log.connect(self.append_log)
        self.runner.failed.connect(lambda text: self.append_log(text + '\n'))
        self.runner.ready.connect(lambda: self.activities or self.status_text.setText('Ready'))
        self.task_finished.connect(self.task_done)
        self.update_button.clicked.connect(self.install_update)
        self.update_found.connect(lambda tag: (self.update_button.setText(f'Update to glycowork {tag}'), self.update_button.show()))
        self.update_progress.connect(lambda percent: self.update_button.setText(f'Downloading update… {percent}%'))
        self.update_downloaded.connect(lambda ok: (self.update_button.setEnabled(True), self.update_button.setText('Install update and restart' if ok else 'Download failed, retry')))
        if self.store.value('geometry'):
            self.restoreGeometry(self.store.value('geometry'))
            self.restoreState(self.store.value('state'))
        else:
            self.resize(1440, 900)
        self.tabs.setCurrentIndex(int(self.store.value('tab', 0)))
        data = [p for p in paths if os.path.isfile(p)]
        if data:
            self.tabs.setCurrentWidget(self.analyze)
            self.analyze.load(data[0])
        if getattr(sys, 'frozen', False) and sys.platform == 'win32':
            # The installed app offers the newest GitHub release that carries a Windows installer; without network access it simply offers nothing
            def check():
                try:
                    from importlib.metadata import version
                    from packaging.version import Version
                    with urllib.request.urlopen('https://api.github.com/repos/BojarLab/glycowork/releases/latest', timeout = 10) as response:
                        release = json.load(response)
                    asset = next(a for a in release['assets'] if a['name'].endswith('-Windows-Setup.exe'))
                    if Version(release['tag_name'].lstrip('v')) > Version(version('glycowork')):
                        self.update_url = asset['browser_download_url']
                        self.update_found.emit(release['tag_name'].lstrip('v'))
                except Exception:
                    pass

            threading.Thread(target = check, daemon = True).start()

    def set_activity(self, key, text):
        """What the app is busy with, by source; the status bar shows the newest, with a busy bar and the time since the first began"""
        if text:
            self.activities[key] = text
        else:
            self.activities.pop(key, None)
        if self.activities and self.since is None:
            self.since = time.time()
            self.clock.start()
        elif not self.activities:
            self.since = None
            self.clock.stop()
            self.elapsed.setText('')
        self.progress.setVisible(bool(self.activities))
        self.status_text.setText(list(self.activities.values())[-1] + '…' if self.activities else 'Ready')
        self.tick()

    def tick(self):
        if self.since is not None:
            seconds = int(time.time() - self.since)
            self.elapsed.setText(f'{seconds // 60}:{seconds % 60:02d}')

    def run_task(self, work, label):
        """Runs work in a background thread, with label in the status bar while it runs"""
        key = next(self.tasks)
        self.set_activity(key, label)

        def run():
            error = ''
            try:
                work()
            except Exception as caught:
                error = f'{type(caught).__name__}: {caught}'
                traceback.print_exc()
            self.task_finished.emit(key, label, error)

        threading.Thread(target = run, daemon = True).start()

    def task_done(self, key, label, error):
        self.set_activity(key, '')
        if error:
            QMessageBox.warning(self, 'glycowork', f'{label} failed:\n\n{error}')
        else:
            self.statusBar().showMessage(f'{label}: done', 6000)

    def append_log(self, text):
        cursor = self.log.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text)
        self.log.setTextCursor(cursor)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile() and os.path.isfile(url.toLocalFile())]
        if not paths:
            return
        if self.tabs.currentWidget() is self.sequences:
            for path in paths:
                self.sequences.add_file(path)
        else:
            self.tabs.setCurrentWidget(self.analyze)
            self.analyze.load(paths[0])

    def install_update(self):
        if self.update_path:
            # Closing first releases the app's files (and asks if an analysis is still running); the installer updates the same folder in place and starts the new version
            if self.close():
                subprocess.Popen([self.update_path, '/SILENT', '/SUPPRESSMSGBOXES', '/NORESTART', '/RELAUNCH=1'])
            return
        self.update_button.setEnabled(False)
        self.update_button.setText('Downloading update…')

        def work():
            path = os.path.join(tempfile.gettempdir(), self.update_url.rsplit('/', 1)[1])
            try:
                with urllib.request.urlopen(self.update_url, timeout = 30) as response, open(path + '.part', 'wb') as file:
                    total, done = int(response.headers.get('Content-Length') or 0), 0
                    while chunk := response.read(1 << 20):
                        file.write(chunk)
                        done += len(chunk)
                        if total:
                            self.update_progress.emit(done * 100 // total)
                os.replace(path + '.part', path)
                self.update_path = path
            except Exception:
                pass
            self.update_downloaded.emit(self.update_path is not None)

        threading.Thread(target = work, daemon = True).start()

    def about(self):
        from importlib.metadata import version, PackageNotFoundError
        try:
            number = version('glycowork')
        except PackageNotFoundError:
            number = ''
        QMessageBox.about(self, 'About glycowork', f'<h3>glycowork {number}</h3><p>Glycan data science: drawing, sequence handling, and statistical analysis of glycomics, '
                          'glycoproteomics, and lectin array data.</p><p>Thomès L, Burkholz R, Bojar D. Glycowork: A Python package for glycan data science and machine '
                          'learning. <i>Glycobiology</i> 31(10), 1240-1244 (2021).</p><p><a href="https://bojarlab.github.io/glycowork/">bojarlab.github.io/glycowork</a></p>')

    def closeEvent(self, event):
        if self.runner.open_jobs and QMessageBox.question(self, 'glycowork', 'An analysis is still running. Quit anyway?') != QMessageBox.Yes:
            event.ignore()
            return
        self.store.setValue('geometry', self.saveGeometry())
        self.store.setValue('state', self.saveState())
        self.store.setValue('tab', self.tabs.currentIndex())
        self.runner.shutdown()
        shutil.rmtree(self.analyze.folder, ignore_errors = True)
        event.accept()


def main():
    if sys.platform == 'win32':
        import ctypes
        # Without its own app ID, Windows groups the window under python.exe and shows Python's taskbar icon
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('BojarLab.glycowork')
    app = QApplication(sys.argv)
    app.setOrganizationName('BojarLab')
    app.setApplicationName('glycowork')
    app.setStyle('Fusion')
    palette = QPalette()
    for role, color in ((QPalette.Window, WASH), (QPalette.Base, '#ffffff'), (QPalette.AlternateBase, '#f8fafc'), (QPalette.Text, INK), (QPalette.WindowText, INK),
                        (QPalette.ButtonText, INK), (QPalette.Button, '#e9eef5'), (QPalette.Highlight, ACCENT), (QPalette.HighlightedText, '#ffffff'),
                        (QPalette.ToolTipBase, '#ffffff'), (QPalette.ToolTipText, INK), (QPalette.Link, ACCENT), (QPalette.PlaceholderText, MUTED)):
        palette.setColor(role, QColor(color))
    app.setPalette(palette)
    app.setStyleSheet(STYLE)
    if os.path.exists(os.path.join(getattr(sys, '_MEIPASS', ''), 'glycowork.ico')):
        app.setWindowIcon(QIcon(os.path.join(sys._MEIPASS, 'glycowork.ico')))
    if sys.argv[1:2] == ['--self-test']:
        # For the app build: draws a glycan in this process, then reads a shipped dataset and runs a differential expression with its figures in the worker,
        # all without a window; exits with 0 only if every step worked
        images, runner, outcome, log = GlycanImages(), Runner(), {}, []
        runner.log.connect(log.append)
        runner.failed.connect(lambda text: (log.append(text), app.quit()))

        def step(kind, content):
            outcome[kind] = content
            if kind != 'loaded':
                app.quit()
                return
            contrasts = content['contrasts']
            runner.submit('run', {'analysis': 'diff', 'df': content['df'], 'kind': content['kind'], 'paired': content['paired'], 'motifs': False, 'feature_set': [],
                                  'groups': [(g, [s for s in contrasts if contrasts[s] == g]) for g in dict.fromkeys(contrasts.values())], 'transform': None,
                                  'annotate': True, 'show_all': False, 'folder': tempfile.mkdtemp(prefix = 'glycowork-self-test-'), 'stamp': 0})

        runner.done.connect(step)
        images.get(EMBLEM, compact = False)
        runner.submit('load', EXAMPLE_DATASET)
        QTimer.singleShot(600000, app.quit)
        app.exec()
        runner.shutdown()
        result, drawing = outcome.get('result', {}), images.images.get((EMBLEM, False, ()), QImage())
        checks = {'SNFG drawing in the window process': not drawing.isNull(), f'{EXAMPLE_DATASET} read': 'loaded' in outcome,
                  'differential expression': len(result.get('tables', {}).get('Differential expression', [])) > 0,
                  'volcano and MA plot saved': len(result.get('figures', {})) == 2 and all(os.path.getsize(path) > 0 for path, _ in result['figures'].values())}
        with open('glycowork_self_test.log', 'w', encoding = 'utf-8') as file:
            file.write(''.join(log) + '\n' + ''.join(f'{name}: {"ok" if ok else "FAILED"}\n' for name, ok in checks.items()))
        sys.exit(0 if all(checks.values()) else 1)
    # glycowork reports through print and warnings; the app has no console, so both go to the log
    stream = LogStream()
    sys.stdout = sys.stderr = stream
    window = MainWindow([os.path.abspath(p) for p in sys.argv[1:]])
    stream.text.connect(window.append_log)
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    # Turns a worker process start of the built app into the worker instead of a second window
    mp.freeze_support()
    main()
