from pathlib import Path
from glycowork.glycan_data.loader import unwrap, resolve_motif_name, lib
from glycowork.motif.regex import get_match
from glycowork.motif.graph import glycan_to_nxGraph, subgraph_isomorphism, compare_glycans, graph_to_string, resolve_anchor
from glycowork.motif.tokenization import get_core, get_modification
from glycowork.motif.processing import min_process_glycans, rescue_glycans, in_lib, expand_lib, get_matching_indices, parse_floating_bit, is_composition, canonicalize_composition, _COMP_ORDER, PDB_TO_IUPAC
import warnings
import html
import hashlib
from io import BytesIO
from typing import Any
import networkx as nx
import drawsvg as draw
import numpy as np
import pandas as pd
import struct
import re
from math import sin, cos, radians, sqrt, atan, degrees


# Adjusted SNFG color palette
col_dict_base = {
    'snfg_white': '#FFFFFF', 'snfg_alt_blue': '#0385AE', 'snfg_green': '#058F60', 'snfg_yellow': '#FCC326',
    'snfg_light_blue': '#91D3E3', 'snfg_pink': '#F39EA0', 'snfg_purple': '#A15989', 'snfg_brown': '#9F6D55',
    'snfg_orange': '#EF6130', 'snfg_red': '#C23537', 'black': '#000000', 'grey': '#7F7F7F'
}

col_dict_transparent = {
    'snfg_white': '#FFFFFF', 'snfg_alt_blue': '#CDE7EF', 'snfg_green': '#CDE9DF', 'snfg_yellow': '#FFF6DE',
    'snfg_light_blue': '#EEF8FB', 'snfg_pink': '#FDF0F1', 'snfg_purple': '#F1E6ED', 'snfg_brown': '#F1E9E5',
    'snfg_orange': '#FDE7E0', 'snfg_red': '#F7E0E0', 'black': '#D9D9D9', 'grey': '#ECECEC'
}

_SCALAR_COLS = {
    'snfg_white': '#FFFFFF', 'snfg_alt_blue': 'darkblue', 'snfg_green': 'green', 'snfg_yellow': 'darkgoldenrod',
    'snfg_light_blue': 'skyblue', 'snfg_pink': 'orchid', 'snfg_purple': 'purple', 'snfg_brown': 'saddlebrown',
    'snfg_orange': 'orangered', 'snfg_red': 'firebrick', 'black': '#000000', 'grey': '#7F7F7F'}

# Shape-color mapping
sugar_dict = {
    "Hex": ['Hex', 'snfg_white', False], "Glc": ['Hex', 'snfg_alt_blue', False],
    "Glcf": ['Hex', 'snfg_alt_blue', True], "Man": ['Hex', 'snfg_green', False],
    "Manf": ['Hex', 'snfg_green', True], "Gal": ['Hex', 'snfg_yellow', False],
    "Galf": ['Hex', 'snfg_yellow', True], "Gul": ['Hex', 'snfg_orange', False],
    "Alt": ['Hex', 'snfg_pink', False], "All": ['Hex', 'snfg_purple', False],
    "Tal": ['Hex', 'snfg_light_blue', False], "Ido": ['Hex', 'snfg_brown', False],
    "Hexf": ['Hex', 'snfg_white', True], "Altf": ['Hex', 'snfg_pink', True],

    "HexNAc": ['HexNAc', 'snfg_white', False], "GlcNAc": ['HexNAc', 'snfg_alt_blue', False],
    "GlcfNAc": ['HexNAc', 'snfg_alt_blue', True], "ManNAc": ['HexNAc', 'snfg_green', False],
    "ManfNAc": ['HexNAc', 'snfg_green', True], "GalNAc": ['HexNAc', 'snfg_yellow', False],
    "GalfNAc": ['HexNAc', 'snfg_yellow', True], "GulNAc": ['HexNAc', 'snfg_orange', False],
    "AltNAc": ['HexNAc', 'snfg_pink', False], "AllNAc": ['HexNAc', 'snfg_purple', False],
    "TalNAc": ['HexNAc', 'snfg_light_blue', False], "IdoNAc": ['HexNAc', 'snfg_brown', False],

    "HexN": ['HexN', 'snfg_white', False], "GlcN": ['HexN', 'snfg_alt_blue', False],
    "ManN": ['HexN', 'snfg_green', False], "GalN": ['HexN', 'snfg_yellow', False],
    "GulN": ['HexN', 'snfg_orange', False], "AltN": ['HexN', 'snfg_pink', False],
    "AllN": ['HexN', 'snfg_purple', False], "TalN": ['HexN', 'snfg_light_blue', False],
    "IdoN": ['HexN', 'snfg_brown', False],

    "HexA": ['HexA', 'snfg_white', False], "GlcA": ['HexA', 'snfg_alt_blue', False],
    "ManA": ['HexA', 'snfg_green', False], "GalA": ['HexA', 'snfg_yellow', False],
    "GulA": ['HexA', 'snfg_orange', False], "AltA": ['HexA_2', 'snfg_pink', False],
    "AllA": ['HexA', 'snfg_purple', False], "TalA": ['HexA', 'snfg_light_blue', False],
    "IdoA": ['HexA_2', 'snfg_brown', False],

    "dHex": ['dHex', 'snfg_white', False], "Qui": ['dHex', 'snfg_alt_blue', False],
    "Rha": ['dHex', 'snfg_green', False], "6dGul": ['dHex', 'snfg_orange', False], "Rhaf": ['dHex', 'snfg_green', True],
    "6dAlt": ['dHex', 'snfg_pink', False], "6dAltf": ['dHex', 'snfg_pink', True],
    "6dTal": ['dHex', 'snfg_light_blue', False], "Fuc": ['dHex', 'snfg_red', False],
    "Fucf": ['dHex', 'snfg_red', True], "6dTalf": ['dHex', 'snfg_light_blue', True],

    "dHexNAc": ['dHexNAc', 'snfg_white', False], "QuiNAc": ['dHexNAc', 'snfg_alt_blue', False],
    "RhaNAc": ['dHexNAc', 'snfg_green', False], "6dAltNAc": ['dHexNAc', 'snfg_pink', False],
    "6dTalNAc": ['dHexNAc', 'snfg_light_blue', False], "FucNAc": ['dHexNAc', 'snfg_red', False],
    "FucfNAc": ['dHexNAc', 'snfg_red', True],

    "ddHex": ['ddHex', 'snfg_white', False], "Oli": ['ddHex', 'snfg_alt_blue', False],
    "Tyv": ['ddHex', 'snfg_green', False], "Abe": ['ddHex', 'snfg_orange', False],
    "Par": ['ddHex', 'snfg_pink', False], "Parf": ['ddHex', 'snfg_pink', True], "Dig": ['ddHex', 'snfg_purple', False],
    "Col": ['ddHex', 'snfg_light_blue', False],

    "Pen": ['Pen', 'snfg_white', False], "Penf": ['Pen', 'snfg_white', True], "Ara": ['Pen', 'snfg_green', False],
    "Araf": ['Pen', 'snfg_green', True], "Lyx": ['Pen', 'snfg_yellow', False],
    "Lyxf": ['Pen', 'snfg_yellow', True], "Xyl": ['Pen', 'snfg_orange', False],
    "Xylf": ['Pen', 'snfg_orange', True], "Rib": ['Pen', 'snfg_pink', False],
    "Ribf": ['Pen', 'snfg_pink', True],

    "dNon": ['dNon', 'snfg_white', False], "Kdn": ['dNon', 'snfg_green', False],
    "Neu5Ac": ['dNon', 'snfg_purple', False], "Neu5Gc": ['dNon', 'snfg_light_blue', False],
    "Neu": ['dNon', 'snfg_brown', False], "Sia": ['dNon', 'snfg_red', False],

    "ddNon": ['ddNon', 'snfg_white', False], "Pse": ['ddNon', 'snfg_green', False],
    "Leg": ['ddNon', 'snfg_yellow', False], "Aci": ['ddNon', 'snfg_pink', False],
    "4eLeg": ['ddNon', 'snfg_light_blue', False],

    "Unknown": ['Unknown', 'snfg_white', False], "Bac": ['Unknown', 'snfg_alt_blue', False],
    "LDManHep": ['Unknown', 'snfg_green', False], "Kdo": ['Unknown', 'snfg_yellow', False],
    "Kdof": ['Unknown', 'snfg_yellow', True], "Dha": ['Unknown', 'snfg_orange', False],
    "DDManHep": ['Unknown', 'snfg_pink', False], "MurNAc": ['Unknown', 'snfg_purple', False],
    "MurNGc": ['Unknown', 'snfg_light_blue', False], "Mur": ['Unknown', 'snfg_brown', False],

    "Assigned": ['Assigned', 'snfg_white', False], "Api": ['Assigned', 'snfg_alt_blue', False],
    "Apif": ['Assigned', 'snfg_alt_blue', True], "Fru": ['Assigned', 'snfg_green', False],
    "Fruf": ['Assigned', 'snfg_green', True], "Tag": ['Assigned', 'snfg_yellow', False],
    "Tagf": ['Assigned', 'snfg_yellow', True], "Sor": ['Assigned', 'snfg_orange', False],
    "Sorf": ['Assigned', 'snfg_orange', True], "Psi": ['Assigned', 'snfg_pink', False],
    "Psif": ['Assigned', 'snfg_pink', True],
    "non_glycan": ['Assigned', 'black', False],

    "blank": ['empty', 'snfg_white', False], "text": ['text', None, None], "-": ['empty', None, None],
    "red_end": ['red_end', None, None], "free": ['free', None, None], "show": ['empty', None, None], "hide": ['empty', None, None],
    "04X": ['04X', None, None], "15A": ['15A', None, None], "02A": ['02A', None, None], "13X": ['13X', None, None],
    "24X": ['24X', None, None], "35X": ['35X', None, None], "04A": ['04A', None, None], "15X": ['15X', None, None],
    "02X": ['02X', None, None], "13A": ['13A', None, None], "24A": ['24A', None, None], "35A": ['35A', None, None],
    "25A": ['25A', None, None], "03A": ['03A', None, None], "14X": ['14X', None, None], "25X": ['25X', None, None],
    "03X": ['03X', None, None], "14A": ['14A', None, None], "Z": ['Z', None, None], "Y": ['Y', None, None],
    "B": ['B', None, None], "C": ['C', None, None]
}

domon_costello = {'B', 'C', 'Z', 'Y', '04X', '15A', '02A', '13X', '24X', '35X', '04A', '15X', '02X', '13A', '24A', '35A', '25A', '03A', '14X', '25X', '03X', '14A'}
SUBSTITUENT_PATTERN = re.compile(r'(?:^|[^1-9])([0-9]+)(?:Substituent|Subst)')
_BOND_ALPHA = re.compile(r"^a\d")
_BOND_BETA = re.compile(r"^b\d")
_BOND_DIGIT = re.compile(r"^\d-\d")
_CONF_PATTERN = re.compile(r'^L-|^D-|(\d,\d+lactone)')
_LABEL_PATTERN = re.compile(r'<!--\s*(.*?)\s*-->')
_TRANSFORM_PATTERN = re.compile(r'transform\s*=\s*"([^"]*)"')
_CONF_DISPLAY = {'L-': 'L', 'D-': 'D', '1,7lactone': 'on'}
_SEGMENT_PREFIXES = {'04', '15', '02', '13', '24', '35', '25', '03', '14'}
_SVG_NUMBER = re.compile(r'-?\d+(?:\.\d+)?(?:e-?\d+)?')
_SVG_LINE_PATH = re.compile(r'<path d="M(-?[\d.eE+-]+),(-?[\d.eE+-]+) L(-?[\d.eE+-]+),(-?[\d.eE+-]+)"[^>]*?id="([^"]+)"')
_SVG_TEXT_PATH = re.compile(r'<text([^>]*)><textPath xlink:href="#([^"]+)" startOffset="([^"]+)">\s*(?:<tspan dy="([^"]+)">(.*?)</tspan>)?\s*</textPath></text>', re.S)


def _get_glycorender():
    from glycorender.render import convert_svg_to_pdf, convert_svg_to_png
    return convert_svg_to_pdf, convert_svg_to_png


def _flatten_text_paths(
        data: str,  # SVG code as emitted by drawsvg
        turn: float = 0  # Rotation the drawing gets on top of the labels, in degrees (90 in vertical mode)
) -> str:  # SVG code with every label as plainly positioned text
    "Rewrites text-on-a-path as absolutely positioned, rotated text, since vector editors such as Affinity Designer silently drop <textPath>"
    lines = {m[4]: [float(k) for k in m[:4]] for m in _SVG_LINE_PATH.findall(data)}

    def _place(m):
        attrs, ref, offset, dy, label = m.group(1), m.group(2), m.group(3), m.group(4) or '0em', m.group(5) or ''
        if ref not in lines:
            return m.group(0)
        x0, y0, x1, y1 = lines[ref]
        length = np.hypot(x1 - x0, y1 - y0)
        frac = float(offset[:-1]) / 100 if offset.endswith('%') else (float(offset) / length if length else 0)
        size = float(re.search(r'font-size="([\d.eE+-]+)"', attrs).group(1))
        angle, shift = np.degrees(np.arctan2(y1 - y0, x1 - x0)), float(dy[:-2]) * size
        if np.cos(np.radians(angle + turn)) < -1e-6:
            # Runs right-to-left once drawn, so turn it upright on the same side of the line; 0.75 em is the cap height of the label fonts
            angle, shift = angle + 180, 0.75 * size - shift
        # Match glycorender: no gap in linkages, bold modifications, italic furanose f
        label = label.replace(' ', '')
        if dy == '0.5em' and label.endswith('f'):
            label = label[:-1] + '<tspan font-style="italic">f</tspan>'
        attrs += ' font-weight="bold"' if dy == '-3.15em' else ''
        x, y = x0 + frac * (x1 - x0), y0 + frac * (y1 - y0)
        return (f'<text{attrs} transform="translate({x:.4f},{y:.4f}) rotate({angle:.4f})" '
                f'x="0" y="{shift:.4f}">{label}</text>')

    # glycorender draws #000000 as charcoal, so nothing in a GlycoDraw figure is ever pure black; the SVG must match
    return _SVG_TEXT_PATH.sub(_place, data).replace('<text ',
                                                    "<text font-family=\"'Century Gothic', Comfortaa, sans-serif\" ").replace(
        '"#000000"', '"#1C1917"')


def _drawn_extent(
        element: Any, # drawsvg element or container to measure
        acc: list # Accumulator of (x0, y0, x1, y1) boxes in user space
) -> list: # The accumulator, so the caller can fold it in one expression
    "Collects the bounding boxes of everything visible in a drawsvg tree, so a drawing can be cropped to what it actually contains"
    from glycorender.render import pdfmetrics, font_to_use
    a = getattr(element, 'args', {}) or {}
    if isinstance(element, draw.Circle) and not a.get('fill') == a.get('stroke') == 'none':
        acc.append((a['cx'] - a['r'], a['cy'] - a['r'], a['cx'] + a['r'], a['cy'] + a['r']))
    elif isinstance(element, draw.Rectangle):
        acc.append((a['x'], a['y'], a['x'] + a['width'], a['y'] + a['height']))
    elif isinstance(element, draw.Text):
        size, shift, bold, track = a.get('font-size', 10), 0, False, 0
        text = element.escaped_text or ''.join(str(getattr(k, 'escaped_text', '') or '') for c in (element.children or []) for k in (c.children or []))
        if not text:
            # Every symbol carries a modification label, empty when it has none, which must not pad the bottom of the crop
            return acc
        if a.get('x') is None:
            # Modification, conformation and linkage labels ride an invisible carrier path, anchored by startOffset along it and displaced by a dy in em
            carrier = element.children[0]
            pts = [float(k) for k in
                   _SVG_NUMBER.findall(re.sub(r'[A-DF-Za-df-z]', ' ', carrier.args['xlink:href'].args['d']))]
            offset = str(carrier.args.get('startOffset', '0'))
            frac = float(offset[:-1]) / 100 if offset.endswith('%') else 0.0
            x, y = pts[0] + frac * (pts[-2] - pts[0]), pts[1] + frac * (pts[-1] - pts[1])
            for tspan in carrier.children or []:
                shift = float(str(tspan.args.get('dy', '0em')).rstrip('em')) * size
                bold = tspan.args.get('dy') == '-3.15em'
            # glycorender drops the spaces of a label on a path and tracks it by 0.05 em
            text, track = text.replace(' ', ''), 0.05 * size
        else:
            x, y = a['x'], a['y']
        # Measured in the font glycorender renders with, as a long modification label like 2Me3Me4Me6Me outgrows any
        # mean glyph width
        text_width = pdfmetrics.stringWidth(text, font_to_use + ('-Bold' if bold else ''), size) + track * (
                len(text) - 1)
        x -= text_width / 2 if a.get('text-anchor') == 'middle' else text_width if a.get('text-anchor') == 'end' else 0
        # 0.8 em is the tallest ascender of the label fonts (digits, capitals, β)
        acc.append((x, y + shift - 0.8 * size, x + text_width, y + shift + 0.3 * size))
    elif 'd' in a and (a.get('stroke-width') or a.get('fill', 'none') not in (None, 'none')):
        # An invisible carrier path is not ink and must not enlarge the crop; its text is measured above instead
        pts = [float(k) for k in _SVG_NUMBER.findall(re.sub(r'[A-DF-Za-df-z]', ' ', a['d']))]
        acc.append((min(pts[0::2]), min(pts[1::2]), max(pts[0::2]), max(pts[1::2])))
    if not isinstance(element, draw.Text):
        start = len(acc)
        for child in getattr(element, 'children', []) or []:
            _drawn_extent(child, acc)
        if rot := re.match(r'rotate\((\S+) (\S+) (\S+)\)', str(a.get('transform', ''))):
            # Upright symbols in vertical mode, brackets and fragment markers sit in rotated groups; turn their boxes along
            deg, rx, ry = (float(k) for k in rot.groups())
            c, s = cos(radians(deg)), sin(radians(deg))
            for i in range(start, len(acc)):
                pts = [(rx + (x - rx) * c - (y - ry) * s, ry + (x - rx) * s + (y - ry) * c)
                       for x in acc[i][0::2] for y in acc[i][1::2]]
                acc[i] = (min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts))
    return acc


def draw_hex(
        x_pos: float, # X coordinate of hexagon center
        y_pos: float, # Y coordinate of hexagon center
        dim: float, # Base dimension for scaling
        col_dict: dict[str, str], # Color mapping dictionary
        drawing: draw.Drawing, # Glycan drawing to be modified
        color: str = 'white', # Fill color
        outline_only: bool = False # Whether to draw only the circumference
) -> None:
    "Draws filled hexagon shape with border at specified position and scale"
    x_base = -x_pos * dim
    y_base = y_pos * dim
    half_dim = 0.5 * dim
    stroke_width = 0.04 * dim
    points = [v for a in (0, 60, 120, 180, 240, 300) for v in (x_base + half_dim * cos(radians(a)), y_base + half_dim * sin(radians(a)))]
    drawing.append(draw.Lines(*points, close = True, fill = 'none' if outline_only else color, stroke = col_dict['black'], stroke_width = stroke_width))


def add_customization(
        drawing, # Drawing object to modify
        x_base: float, # X coordinate of base position
        y_base: float, # Y coordinate of base position
        dim: float, # Base dimension for scaling
        modification: str, # Text annotation for modifications
        col_dict: dict[str, str], # Color mapping dictionary
        conf: str = None, # Ring configuration text
        furanose: bool = False, # Draw furanose indicator
        text_anchor: str = 'middle' # Text alignment
) -> None:
    "Adds text annotations and indicators to glycan symbol"
    half_dim = dim / 2
    # Text annotation
    p = draw.Path(stroke_width = 0)
    p.M(x_base-dim, y_base+half_dim)
    p.L(x_base+dim, y_base+half_dim)
    drawing.append(p)
    drawing.append(draw.Text(modification, dim*0.35, path = p, fill = col_dict['black'], text_anchor = text_anchor, line_offset = -3.15))
    if furanose or conf:
        conf_text = ""
        if conf:
            conf_text = _CONF_DISPLAY.get(conf, conf)
        if furanose:
            conf_text += "f"
        p = draw.Path(stroke_width = 0)
        p.M(x_base-dim, y_base)
        p.L(x_base+dim, y_base)
        drawing.append(p)
        drawing.append(draw.Text(conf_text, dim*0.3, path = p, fill = col_dict['black'], text_anchor = text_anchor, center = True))


def draw_shape(
        shape: str, # SNFG shape designation
        color: str, # SNFG color designation
        x_pos: float, # X coordinate of shape center
        y_pos: float, # Y coordinate of shape center
        col_dict: dict[str, str], # Color mapping dictionary
        drawing: draw.Drawing, # Glycan drawing to be modified
        modification: str = '', # Text annotation for modifications
        dim: float = 50, # Base dimension for scaling
        furanose: bool = False, # Draw furanose indicator
        conf: str = '', # Ring configuration text
        deg: float = 0, # Rotation angle in degrees
        text_anchor: str = 'middle', # Text alignment for postbiosynthetic modifications
        scalar: float = 0 # Intensity scaling factor for drawsvg output
) -> None:
    "Draws SNFG glycan symbol with specified shape, color, position, and optional annotations"
    x_base = -x_pos * dim
    y_base = y_pos * dim
    stroke_w = 0.04 * dim
    half_dim = dim / 2
    inside_hex_dim = ((sqrt(3))/2) * half_dim
    if scalar:
        radius = 2.35 if shape == "HexNAc" else 2.2
        gradient = draw.RadialGradient(x_base, y_base, half_dim * radius)
        opacity = max(0, min(1, scalar))*0.8  # Normalize opacity to [0, 1]
        opacity = opacity * 1.3 if color in ['snfg_yellow', 'snfg_light_blue', 'snfg_white'] else opacity
        gradient.add_stop(0, _SCALAR_COLS[color], opacity = opacity)
        gradient.add_stop(0.6, _SCALAR_COLS[color], opacity = opacity * 0.4)
        gradient.add_stop(1, 'white', opacity = 0)
        drawing.append(draw.Circle(x_base, y_base, half_dim * radius, fill = gradient))
    if shape == 'Hex':
        # Hexose - circle
        drawing.append(draw.Circle(x_base, y_base, half_dim, fill = col_dict[color], stroke_width = stroke_w, stroke = col_dict['black']))
    elif shape == 'HexNAc':
        # HexNAc - square
        drawing.append(draw.Rectangle(x_base-half_dim, y_base-half_dim, dim, dim, fill = col_dict[color], stroke_width = stroke_w, stroke = col_dict['black']))
    elif shape == 'HexN':
        # Hexosamine - crossed square
        drawing.append(draw.Rectangle(x_base-half_dim, y_base-half_dim, dim, dim, fill = 'white', stroke_width = stroke_w, stroke = col_dict['black']))
        drawing.append(draw.Lines(x_base-half_dim, y_base-half_dim,
                                  x_base+half_dim, y_base-half_dim,
                                  x_base+half_dim, y_base+half_dim,
                                  x_base-half_dim, y_base-half_dim,
                                  close = True, fill = col_dict[color], stroke = col_dict['black'], stroke_width = 0))
        p = draw.Path(stroke_width = stroke_w, stroke = col_dict['black'])
        p.M(x_base-half_dim, y_base-half_dim).L(x_base+half_dim, y_base-half_dim).M(x_base+half_dim, y_base-half_dim).L(x_base+half_dim, y_base+half_dim).M(x_base+half_dim, y_base+half_dim).L(x_base-half_dim, y_base-half_dim)
        drawing.append(p)
    elif shape in ['HexA_2', 'HexA']:
        # Hexuronate - divided diamond;  AltA / IdoA for HexA_2 and flipped colors for HexA
        drawing.append(draw.Lines(x_base,         y_base+half_dim,
                                  x_base+half_dim, y_base,
                                  x_base,         y_base-half_dim,
                                  x_base-half_dim, y_base,
                                  close = True, fill = 'white', stroke = col_dict['black'], stroke_width = stroke_w))
        y_direction = half_dim if shape == 'HexA_2' else -half_dim
        drawing.append(draw.Lines(x_base-half_dim, y_base,
                                  x_base, y_base+y_direction,
                                  x_base+half_dim, y_base,
                                  x_base-half_dim, y_base,
                                  close = True, fill = col_dict[color], stroke = col_dict['black'], stroke_width = 0))
        p = draw.Path(stroke_width = stroke_w, stroke = col_dict['black'], fill = 'none')
        p.M(x_base-half_dim, y_base).L(x_base, y_base+y_direction).L(x_base+half_dim, y_base).M(x_base-half_dim, y_base).L(x_base+half_dim, y_base)
        drawing.append(p)
    elif shape == 'dHex':
        # Deoxyhexose - triangle
        drawing.append(draw.Lines(x_base- half_dim, y_base+inside_hex_dim,  # -(dim*1/3)
                                  x_base, y_base-inside_hex_dim,
                                  x_base+half_dim, y_base+inside_hex_dim,
                                  close = True, fill = col_dict[color], stroke = col_dict['black'], stroke_width = stroke_w))
    elif shape == 'dHexNAc':
        # Deoxyhexnac - divided triangle
        drawing.append(draw.Lines(x_base-half_dim, y_base+inside_hex_dim,  # -(dim*1/3) for center of triangle
                                  x_base, y_base-inside_hex_dim,  # -(dim*1/3) for bottom alignment
                                  x_base+half_dim, y_base+inside_hex_dim,  # -(((3**0.5)/2)*dim*0.5) for half of triangle height
                                  close = True, fill = 'white', stroke = col_dict['black'], stroke_width = stroke_w))
        drawing.append(draw.Lines(x_base, y_base+inside_hex_dim,  # -(dim*1/3)
                                  x_base, y_base-inside_hex_dim,
                                  x_base+half_dim, y_base+inside_hex_dim,
                                  close = True, fill = col_dict[color], stroke = col_dict['black'], stroke_width = 0))
        p = draw.Path(stroke_width = stroke_w, stroke = col_dict['black'], fill = 'none')
        p.M(x_base, y_base+inside_hex_dim).L(x_base, y_base-inside_hex_dim).M(x_base, y_base+inside_hex_dim).L(x_base+half_dim, y_base+inside_hex_dim).M(x_base, y_base-inside_hex_dim).L(x_base+half_dim, y_base+inside_hex_dim)
        drawing.append(p)
    elif shape == 'ddHex':
        # Dideoxyhexose - flat rectangle
        drawing.append(draw.Lines(x_base-half_dim,         y_base+(dim*7/12*0.5),  # -(dim*0.5/12)
                                  x_base+half_dim,         y_base+(dim*7/12*0.5),
                                  x_base+half_dim,         y_base-(dim*7/12*0.5),
                                  x_base-half_dim,         y_base-(dim*7/12*0.5),
                                  close = True, fill = col_dict[color], stroke = col_dict['black'], stroke_width = stroke_w))
    elif shape == 'Pen':
        # Pentose - star
        cos18 = cos(radians(18))
        cos54 = cos(radians(54))
        sin18 = sin(radians(18))
        sin54 = sin(radians(54))
        base_r = half_dim/cos18
        small_r = (0.25*dim)/cos18
        drawing.append(draw.Lines(x_base, y_base-base_r,
                                  x_base+small_r*cos54, y_base-small_r*sin54,
                                  x_base+base_r*cos18, y_base-base_r*sin18,
                                  x_base+small_r*cos18, y_base+small_r*sin18,
                                  x_base+base_r*cos54, y_base+base_r*sin54,
                                  x_base, y_base+small_r,
                                  x_base-base_r*cos54, y_base+base_r*sin54,
                                  x_base-small_r*cos18, y_base+small_r*sin18,
                                  x_base-base_r*cos18, y_base-base_r*sin18,
                                  x_base-small_r*cos54, y_base-small_r*sin54,
                                  close = True, fill = col_dict[color], stroke = col_dict['black'], stroke_width = stroke_w))
    elif shape in ['dNon', 'ddNon']:
        # Deoxynonulosonate - diamond or Dideoxynonulosonate - flat diamond
        diamond_adjust = 0 if shape == 'dNon' else dim*1/8
        drawing.append(draw.Lines(x_base,         y_base+half_dim-diamond_adjust,
                                  x_base+half_dim+diamond_adjust, y_base,
                                  x_base,         y_base-half_dim+diamond_adjust,
                                  x_base-half_dim-diamond_adjust, y_base,
                                  close = True, fill = col_dict[color], stroke = col_dict['black'], stroke_width = stroke_w))
    elif shape == 'Unknown':
        # Unknown - flat hexagon
        flat_adjust = dim*1/8
        extra = dim*0.2
        drawing.append(draw.Lines(x_base-half_dim+flat_adjust, y_base+half_dim-flat_adjust,
                                  x_base+half_dim-flat_adjust, y_base+half_dim-flat_adjust,
                                  x_base+half_dim-flat_adjust+extra, y_base,
                                  x_base+half_dim-flat_adjust, y_base-half_dim+flat_adjust,
                                  x_base-half_dim+flat_adjust, y_base-half_dim+flat_adjust,
                                  x_base-half_dim+flat_adjust-extra, y_base,
                                  close = True, fill = col_dict[color], stroke = col_dict['black'], stroke_width = stroke_w))
    elif shape == 'Assigned':
        # Assigned - pentagon
        cos18 = cos(radians(18))
        cos54 = cos(radians(54))
        sin18 = sin(radians(18))
        sin54 = sin(radians(54))
        base_r = half_dim/cos18
        drawing.append(draw.Lines(x_base, y_base-base_r,
                                  x_base+base_r*cos18, y_base-base_r*sin18,
                                  x_base+base_r*cos54, y_base+base_r*sin54,
                                  x_base-base_r*cos54, y_base+base_r*sin54,
                                  x_base-base_r*cos18, y_base-base_r*sin18,
                                  close = True, fill = col_dict[color], stroke = col_dict['black'], stroke_width = stroke_w))
    elif shape == 'empty':
        drawing.append(draw.Circle(x_base, y_base, dim/2, fill = 'none', stroke_width = stroke_w, stroke = 'none'))
    elif shape == 'text':
        drawing.append(draw.Text(modification, dim*0.35, x_base, y_base, text_anchor = text_anchor, fill = col_dict['black']))
    elif shape in {'red_end', 'free'}:
        p = draw.Path(stroke_width = stroke_w, stroke = col_dict['black'], fill = 'none')
        p.M((x_base+0.1*dim), (y_base-0.4*dim))  # Start path at point (-10, 20)
        p.C((x_base-0.3*dim), (y_base-0.1*dim),
            (x_base+0.3*dim), (y_base+0.1*dim),
            (x_base-0.1*dim), (y_base+0.4*dim))
        drawing.append(p)
        if shape == 'red_end':
            drawing.append(draw.Circle(x_base, y_base, 0.15 * dim, fill = 'white', stroke_width = stroke_w, stroke = col_dict['black']))
    # Handle segmented Hex shapes (04X, 15A, etc.)
    elif shape[:2] in _SEGMENT_PREFIXES:
        use_grey_base = shape in {'04A', '15X', '02X', '13A', '24A', '35A', '14A'}
        segment_fill = 'white' if use_grey_base else col_dict['grey']
        # Define angle pairs for each shape type
        angles = {
            '04': (30, 150, [60, 120]), '15': (90, 330, [60, 0]),
            '02': (30, 270, [0, 300]), '13': (330, 210, [300, 240]),
            '24': (270, 150, [240, 180]), '35': (210, 90, [180, 120]),
            '25': (90, 270, [60, 0, 300]), '03': (30, 210, [0, 300, 240]),
            '14': (330, 150, [300, 240, 180])
        }
        start_angle, end_angle, mid_angles = angles[shape[:2]]
        draw_hex(x_pos, y_pos, dim, col_dict, drawing, color = col_dict['grey'] if use_grey_base else 'white')
        # Draw the segment
        points = [x_base, y_base]
        points.extend([x_base+inside_hex_dim*cos(radians(start_angle)), y_base-inside_hex_dim*sin(radians(start_angle))])
        for angle in mid_angles:
            points.extend([x_base+half_dim*cos(radians(angle)), y_base-half_dim*sin(radians(angle))])
        points.extend([x_base+inside_hex_dim*cos(radians(end_angle)), y_base-inside_hex_dim*sin(radians(end_angle))])
        drawing.append(draw.Lines(*points, close = True, fill = segment_fill, stroke = col_dict['black'], stroke_width = 0))
        # Draw the dividing line; either center-to-edge or edge-to-edge
        if shape[:2] in {'25', '03', '14'}:
            p = draw.Path(stroke_width = stroke_w, stroke = col_dict['black'])
            p.M(x_base+inside_hex_dim*cos(radians(start_angle)), y_base-inside_hex_dim*sin(radians(start_angle)))
            p.L(x_base+inside_hex_dim*cos(radians(end_angle)), y_base-inside_hex_dim*sin(radians(end_angle)))
            drawing.append(p)
        else:
            for angle in [start_angle, end_angle]:
                p = draw.Path(stroke_width = stroke_w, stroke = col_dict['black'])
                p.M(x_base, y_base)
                p.L(x_base+inside_hex_dim*cos(radians(angle)), y_base-inside_hex_dim*sin(radians(angle)))
                drawing.append(p)
        # Draw outline
        draw_hex(x_pos, y_pos, dim, col_dict, drawing, outline_only = True)
    elif shape in {'Z', 'Y'}:
        rot = f'rotate({deg} {-abs(x_pos)*dim} {-abs(y_pos)*dim})'
        g = draw.Group(transform = rot)
        p = draw.Path(stroke_width = stroke_w, stroke = col_dict['black'])
        p.M(x_base, y_base-half_dim)
        p.L(x_base, y_base+half_dim)
        p.M(x_base-0.02*dim, y_base-half_dim)
        p.L(x_base+0.4*dim, y_base-half_dim)
        g.append(p)
        if shape == 'Y':
            g.append(draw.Circle(x_base + 0.4 * dim, y_base, 0.15 * dim, fill = 'none', stroke_width = stroke_w, stroke = col_dict['black']))
        drawing.append(g)
    elif shape in {'B', 'C'}:
        p = draw.Path(stroke_width = stroke_w, stroke = col_dict['black'])
        p.M(x_base, y_base-half_dim)
        p.L(x_base, y_base+half_dim)
        p.M(x_base+0.02*dim, y_base+half_dim)
        p.L(x_base-0.4*dim, y_base+half_dim)
        drawing.append(p)
        if shape == 'C':
            drawing.append(draw.Circle(x_base - 0.4 * dim, y_base, 0.15 * dim, fill = 'none', stroke_width = stroke_w, stroke = col_dict['black']))
    if shape not in {'empty', 'text', 'red_end', 'free', 'Z', 'Y', 'B', 'C'} and shape[:2] not in _SEGMENT_PREFIXES:
        add_customization(drawing, x_base = x_base, y_base = y_base, dim = dim, modification = modification,
                          col_dict = col_dict, conf = conf, furanose = furanose, text_anchor = text_anchor)


def add_bond(
        x_start: float, # Starting X coordinate
        x_stop: float, # Ending X coordinate
        y_start: float, # Starting Y coordinate
        y_stop: float, # Ending Y coordinate
        drawing: draw.Drawing, # Glycan drawing to be modified
        label: str = '', # Bond label text
        dim: float = 50, # Base dimension for scaling
        compact: bool = False, # Use compact drawing style
        highlight: str = 'show', # Highlight state: 'show' or 'hide'
        color_highlight: bool = False,  # Whether to highlight this linkage in red
        dashed: bool = False  # Whether to draw the bond dashed, for uncertain attachment
) -> None:
    "Draws glycosidic bond line with optional label between specified coordinates"
    col_dict = col_dict_transparent if highlight == 'hide' else col_dict_base
    scaling_factor = 1.2 if compact else 2
    y_scaling = 0.6 if compact else 1
    x_start, x_stop = [-x * scaling_factor * dim for x in (x_start, x_stop)]
    y_start, y_stop = [y * y_scaling * dim for y in (y_start, y_stop)]
    if abs(x_start - x_stop) < 1e-9 and y_stop > y_start:
        # A vertical bond always runs upwards, so its label reads bottom-to-top on its left whether the branch hangs above or below, and turns upright in vertical mode
        y_start, y_stop = y_stop, y_start
    final_width = 0.12 * dim if color_highlight else 0.08 * dim
    if dashed:  # A fixed dash period vanishes on the short bonds of compact mode, so scale it to the bond
        length = ((x_stop - x_start) ** 2 + (y_stop - y_start) ** 2) ** 0.5
        segment = length / (2 * max(3, round(length / (0.4 * dim))))
    p = draw.Path(stroke_width = final_width, stroke = col_dict['snfg_red'] if color_highlight else col_dict['black'],
                  class_ = 'snfg-linkage',
                  **({'stroke_dasharray': f"{segment},{segment}"} if dashed else {}))
    p.M(x_start, y_start).L(x_stop, y_stop)
    drawing.append(p)
    if label and label != '-':
        # A wildcard linkage such as "β 2/4/6" is far wider than the bond it rides on, so shrink it to what fits
        # between the two symbols; along a diagonal bond a symbol reaches dim / 2 * (|cos| + |sin|), its corner
        # poking under the label
        length = ((x_stop - x_start) ** 2 + (y_stop - y_start) ** 2) ** 0.5
        span = length - dim * (abs(x_stop - x_start) + abs(y_stop - y_start)) / (length or 1)
        drawing.append(draw.Text(label, min(dim * 0.4, max(dim * 0.22, span / (0.6 * len(label)))), path = p,
                                 text_anchor = 'middle', fill = col_dict['black'], valign = 'middle',
                                 line_offset = -0.5))


def add_sugar(
        monosaccharide: str, # IUPAC monosaccharide name
        drawing: draw.Drawing, # Glycan drawing to be modified
        x_pos: float = 0, # X coordinate of sugar center
        y_pos: float = 0, # Y coordinate of sugar center
        modification: str = '', # Text annotation for modifications
        dim: float = 50, # Base dimension for scaling
        compact: bool = False, # Use compact drawing style
        conf: str = '', # Ring configuration text
        deg: float = 0, # Rotation angle in degrees
        text_anchor: str = 'middle', # Text alignment for postbiosynthetic modifications
        highlight: str = 'show', # Highlight state: 'show' or 'hide'
        scalar: float = 0 # Intensity scaling factor
) -> None:
    "Draws SNFG monosaccharide symbol with specified parameters at given position"
    col_dict = col_dict_transparent if highlight == 'hide' else col_dict_base
    x_pos = x_pos * (1.2 if compact else 2)
    y_pos = y_pos * (0.6 if compact else 1)
    shape = sugar_dict[monosaccharide][0] if monosaccharide in sugar_dict else (
        'Hex' if sum(p in sugar_dict for p in monosaccharide.split('/')) > 1 else 'empty')
    if shape not in {'empty', 'text', 'red_end', 'free', 'Z', 'Y', 'B', 'C'} and shape[:2] not in _SEGMENT_PREFIXES:
        # One group per symbol, so that vertical mode can turn it back upright about its own center
        symbol = draw.Group(class_ = 'snfg-symbol')
        symbol.center = (-x_pos * dim, y_pos * dim)
        drawing.append(symbol)
        drawing = symbol
    if monosaccharide in sugar_dict:
        shape, color, furanose = sugar_dict[monosaccharide]
        draw_shape(shape = shape, color = color, x_pos = x_pos, y_pos = y_pos, drawing = drawing, modification = modification,
                   conf = conf, furanose = furanose, dim = dim, deg = deg, text_anchor = text_anchor, col_dict = col_dict, scalar = scalar)
    elif '/' in monosaccharide:
        parts = monosaccharide.split('/')
        entries = [sugar_dict[p] for p in parts if p in sugar_dict]
        if len(entries) < 2:
            return
        shape, furanose = entries[0][0], any(e[2] for e in entries)
        colors = [col_dict[e[1]] for e in entries[:2]]
        x_base, y_base = -x_pos * dim, y_pos * dim
        stroke_w, half_dim = 0.04 * dim, dim / 2
        if shape == 'Hex':
            for sign, col in [(1, colors[0]), (-1, colors[1])]:
                pts = []
                for i in range(33):
                    a = radians(135 + sign * i * 180 / 32)
                    pts.extend([x_base + half_dim * cos(a), y_base - half_dim * sin(a)])
                drawing.append(draw.Lines(*pts, close = True, fill = col, stroke = 'none', stroke_width = 0))
            # A path, not a circle: glycorender paints circles before paths, so the halves would cover half its stroke
            k = 0.5523 * half_dim
            p = draw.Path(fill = 'none', stroke_width = stroke_w, stroke = col_dict['black'])
            p.M(x_base + half_dim, y_base)
            p.C(x_base + half_dim, y_base + k, x_base + k, y_base + half_dim, x_base, y_base + half_dim)
            p.C(x_base - k, y_base + half_dim, x_base - half_dim, y_base + k, x_base - half_dim, y_base)
            p.C(x_base - half_dim, y_base - k, x_base - k, y_base - half_dim, x_base, y_base - half_dim)
            p.C(x_base + k, y_base - half_dim, x_base + half_dim, y_base - k, x_base + half_dim, y_base).Z()
            drawing.append(p)
        elif shape == 'HexNAc':
            tl, tr = (x_base - half_dim, y_base - half_dim), (x_base + half_dim, y_base - half_dim)
            bl, br = (x_base - half_dim, y_base + half_dim), (x_base + half_dim, y_base + half_dim)
            drawing.append(draw.Lines(*tl, *bl, *br, close = True, fill = colors[0], stroke = 'none', stroke_width = 0))
            drawing.append(draw.Lines(*tl, *tr, *br, close = True, fill = colors[1], stroke = 'none', stroke_width = 0))
            # A path rather than a rect, so that glycorender paints it after the halves instead of under them
            drawing.append(draw.Lines(*tl, *tr, *br, *bl, close = True, fill = 'none', stroke = col_dict['black'],
                                      stroke_width = stroke_w))
        elif shape == 'dNon':
            drawing.append(
                draw.Lines(x_base, y_base + half_dim, x_base - half_dim, y_base, x_base, y_base - half_dim, close = True,
                           fill = colors[0], stroke = 'none', stroke_width = 0))
            drawing.append(
                draw.Lines(x_base, y_base + half_dim, x_base + half_dim, y_base, x_base, y_base - half_dim, close = True,
                           fill = colors[1], stroke = 'none', stroke_width = 0))
            drawing.append(
                draw.Lines(x_base, y_base + half_dim, x_base + half_dim, y_base, x_base, y_base - half_dim, x_base - half_dim,
                           y_base, close = True, fill = 'none', stroke = col_dict['black'], stroke_width = stroke_w))
        elif shape == 'dHex':
            ihd = ((sqrt(3)) / 2) * half_dim
            drawing.append(
                draw.Lines(x_base - half_dim, y_base + ihd, x_base, y_base - ihd, x_base, y_base + ihd, close = True,
                           fill = colors[0], stroke = 'none', stroke_width = 0))
            drawing.append(
                draw.Lines(x_base, y_base + ihd, x_base, y_base - ihd, x_base + half_dim, y_base + ihd, close = True,
                           fill = colors[1], stroke = 'none', stroke_width = 0))
            drawing.append(
                draw.Lines(x_base - half_dim, y_base + ihd, x_base, y_base - ihd, x_base + half_dim, y_base + ihd, close = True,
                           fill = 'none', stroke = col_dict['black'], stroke_width = stroke_w))
        else:
            draw_shape(shape = shape, color = entries[0][1], x_pos = x_pos, y_pos = y_pos, drawing = drawing,
                       modification = modification,
                       conf = conf, furanose = entries[0][2], dim = dim, deg = deg, text_anchor = text_anchor,
                       col_dict = col_dict, scalar = scalar)
            return
        # Dividing line
        p = draw.Path(stroke_width = stroke_w, stroke = col_dict['black'])
        if shape in ('HexNAc', 'Hex'):
            d = half_dim if shape == 'HexNAc' else half_dim * sqrt(2) / 2
            p.M(x_base - d, y_base - d).L(x_base + d, y_base + d)
        else:
            div_y = ((sqrt(3)) / 2) * half_dim if shape == 'dHex' else half_dim
            p.M(x_base, y_base - div_y).L(x_base, y_base + div_y)
        drawing.append(p)
        add_customization(drawing, x_base = x_base, y_base = y_base, dim = dim, modification = modification,
                          col_dict = col_dict, conf = conf, furanose = furanose, text_anchor = text_anchor)
    else:
        x_base = -x_pos * dim
        y_base = y_pos * dim
        half_dim = dim / 2
        p = draw.Path(stroke_width = 0.04 * dim, stroke = col_dict['black'])
        p.M(x_base - half_dim, y_base + half_dim)
        p.L(x_base + half_dim, y_base - half_dim)
        p.M(x_base + half_dim, y_base + half_dim)
        p.L(x_base - half_dim, y_base - half_dim)
        drawing.append(p)


def process_bonds(
        linkage_list: list[str] | list[list[str]] # Glycosidic linkages
) -> list[str] | list[list[str]]: # Formatted linkage text
    "Formats glycosidic linkage text for visualization"
    def process_single_linkage(linkage: str) -> str:
        if '-' in linkage:
            first, last = linkage[0], re.search(r"-(.*)", linkage).group(1)
        else:
            first, last = linkage[0], linkage[-1]
        if '?' in first and '?' in last: return '?'
        if '?' in first: return f' {last}'
        if '?' in last:
            if _BOND_ALPHA.match(linkage): return '\u03B1'
            if _BOND_BETA.match(linkage): return '\u03B2'
            return '-'
        if _BOND_ALPHA.match(linkage): return f'\u03B1 {last}'
        if _BOND_BETA.match(linkage): return f'\u03B2 {last}'
        if _BOND_DIGIT.match(linkage): return f'{first} - {last}'
        return '-'
    if linkage_list and isinstance(linkage_list[0], list):
        return [[process_single_linkage(linkage) for linkage in sub_list] for sub_list in linkage_list]
    return [process_single_linkage(linkage) for linkage in linkage_list]


def get_highlight_attribute(
        glycan_graph: nx.DiGraph, # NetworkX glycan graph
        motif_string: str, # Motif to highlight
        termini_list: list = [], # Terminal position specifications
        reverse_highlight: bool = False # Whether to highlight everything EXCEPT highlight_motif
) -> nx.DiGraph: # Graph with highlight attributes
    "Labels nodes in glycan graph based on presence in specified motif"
    if motif_string:
        motif = glycan_to_nxGraph(motif_string, termini = 'provided' if termini_list else None, termini_list = termini_list)
        _, mappings = subgraph_isomorphism(glycan_graph, motif, termini_list = termini_list, return_matches = True)
        matched = set(unwrap(mappings))
        if not matched:
            warnings.warn(
                f"highlight_motif '{motif_string}' does not occur in {graph_to_string(glycan_graph)}, so nothing is highlighted.")
        in_label, out_label = ('hide', 'show') if reverse_highlight else ('show', 'hide')
        mapping_show = {node: in_label if node in matched else out_label for node in glycan_graph.nodes()}
    else:
        mapping_show = {node: 'show' for node in glycan_graph.nodes()}
    nx.set_node_attributes(glycan_graph, dict(sorted(mapping_show.items())), 'highlight_labels')
    return glycan_graph


def get_branches_from_graph(graph: nx.DiGraph, main_chain: list, main_chain_sugars: list):
    """Extract branch structures based on paths to lowest node index tips"""
    main_chain_set = set(main_chain)
    all_nodes = main_chain_set.copy()
    main_chain_sugars = sorted(main_chain_sugars, reverse = True)

    def add_branch(branch_list: list, start_node: int, connection: tuple):
        # A branch always follows the lowest node index at a fork
        path = [start_node]
        while successors := [n for n in graph.successors(path[-1]) if n not in all_nodes]:
            path.append(min(successors))
        branch_list.append({'nodes': path, 'sugar_nodes': sorted([n for n in path if n % 2 == 0], reverse = True),
                            'connection': connection})
        all_nodes.update(path)

    def process_level(parent_level):
        level = []
        for i, branch in enumerate(parent_level):
            for j, node in enumerate(branch['sugar_nodes']):
                if graph.out_degree(node) > 1:
                    for succ in sorted(graph.successors(node)):
                        if succ not in all_nodes:
                            add_branch(level, succ, (i, j))
        return level

    first_level = []
    for node in main_chain:
        if node % 2 == 0 and graph.out_degree(node) > 1:  # Sugar with branches
            for succ in sorted(graph.successors(node)):
                if succ not in all_nodes:
                    add_branch(first_level, succ, (0, main_chain_sugars.index(node)))
    levels = [first_level]
    while levels[-1]:
        levels.append(process_level(levels[-1]))
    levels.pop()  # The loop only stops on an empty level
    return levels + [[]] * max(0, 3 - len(levels))


def get_coordinates_and_labels(
        draw_this: str, # IUPAC-condensed glycan sequence
        highlight_motif: str | None, # Motif to highlight
        termini_list: list = [], # Terminal position specifications (from 'terminal', 'internal', and 'flexible')
        reverse_highlight: bool = False # Whether to highlight everything EXCEPT highlight_motif
) -> list[list]: # Drawing coordinates and labels (monosaccharide label, x position, y position, modification, bond, conformation)
    "Calculates drawing coordinates and formats labels for glycan visualization"
    graph = glycan_to_nxGraph(draw_this, termini = 'calc' if termini_list else 'ignore').copy()
    graph = get_highlight_attribute(graph, highlight_motif, termini_list = termini_list, reverse_highlight = reverse_highlight)
    node_values = list(nx.get_node_attributes(graph, 'string_labels').values())
    highlight_values = list(nx.get_node_attributes(graph, 'highlight_labels').values())
    parsed_sugars = {}
    for idx, raw_label in enumerate(node_values):
        if idx % 2:
            continue
        negated = raw_label.startswith('!')
        if negated:
            highlight_values[idx] = 'hide'
            if idx + 1 < len(highlight_values): highlight_values[idx + 1] = 'hide'
            raw_label = raw_label[1:]
        if '/' in raw_label and raw_label not in domon_costello:
            cores = [get_core(p) for p in raw_label.split('/')]
            if all(c in sugar_dict for c in cores):
                parsed_sugars[idx] = ('/'.join(cores), '!' if negated else '')
                continue
        core_label = get_core(raw_label) if raw_label not in domon_costello else raw_label
        normalized_label = core_label if core_label in sugar_dict else 'Unknown'
        modification_text = get_modification(raw_label)
        # get_core knows only some furanoses, so Altf or Tagf would carry their ring 'f' as a modification label
        if normalized_label + 'f' in sugar_dict and re.match(r'(?:[DL]-)?f', modification_text):
            normalized_label, modification_text = normalized_label + 'f', modification_text.replace('f', '', 1)
        # Text right after an amine is its N-substituent (GlcNS, GlcNGc), unlike O-substituents or a uronic A: NS, NGc
        if sugar_dict.get(normalized_label, [''])[0] == 'HexN' and re.search(
                normalized_label + r'(?!A(?![a-z]))[A-NP-Z]', raw_label):
            modification_text = re.sub(r'^((?:[DL]-)?)', r'\1N', modification_text, count = 1)
        # An O marks an O-linked substituent (OMe, OS, OAc) and goes, but it stays inside a name such as Ole or Oct
        modification_text = re.sub(r'O(?=[A-Z])', '', modification_text).replace('-ol', '')
        if modification_text:
            modification_text = modification_text.replace('Substituent', 'Subst')
            match = SUBSTITUENT_PATTERN.search(modification_text) if 'Subst' in modification_text else None
            modification_text = f"{match.group(1)}Subst" if match else (
                modification_text if ('Subst' in modification_text or normalized_label != 'Unknown') else '')
        parsed_sugars[idx] = (normalized_label, ('!' + modification_text) if negated else modification_text)
    root = max(graph.nodes())
    leaves = [n for n in graph.nodes() if graph.out_degree(n) == 0 and n != root] if len(graph) > 1 else [0]
    main_chain = nx.shortest_path(graph.reverse(copy = False), leaves[0], root) if leaves else []
    main_label_sugar = [node for node in main_chain if node % 2 == 0]
    main_entries = [parsed_sugars[node] for node in main_chain if node % 2 == 0]
    main_sugar, main_sugar_modification = map(list, zip(*reversed(main_entries)))
    main_bond = [node_values[node] for node in main_chain if node % 2 == 1][::-1]  # Odd indices are bonds
    main_sugar_highlight = [highlight_values[node] for node in main_chain if node % 2 == 0][::-1]
    main_bond_highlight = [highlight_values[node] for node in main_chain if node % 2 == 1][::-1]
    main_sugar_x_pos = list(range(len(main_sugar)))
    branch_levels = get_branches_from_graph(graph, main_chain, main_label_sugar)

    def process_branch_data(branches: list):
        sugar, sugar_mod, bond, connection, sugar_label, bond_label = [], [], [], [], [], []
        for branch in branches:
            # Extract sugar and bond labels
            sugar_nodes = branch['sugar_nodes']
            bond_nodes = [m for m in branch['nodes'] if m % 2 == 1]
            sugar_entries = [parsed_sugars[n] for n in sugar_nodes]
            sugar.append([label for label, _ in sugar_entries])
            sugar_mod.append([mod for _, mod in sugar_entries])
            bond.append([node_values[n] for n in bond_nodes])
            connection.append(branch['connection'])
            sugar_label.append([highlight_values[n] for n in sugar_nodes])
            bond_label.append([highlight_values[n] for n in bond_nodes])
        return sugar, sugar_mod, bond, connection, sugar_label, bond_label

    # Get branch data for all levels
    lv_sugar, lv_sugar_modification, lv_bond, lv_connection, lv_sugar_label, lv_bond_label = map(list, zip(*[
        process_branch_data(b) for b in branch_levels]))
    # Process linkages
    main_bond = process_bonds(main_bond)
    lv_bond = [process_bonds(b) for b in lv_bond]
    # Main chain x
    tucked_end = (main_sugar[-1] == 'Fuc' and draw_this.count('(') > 1) or (
                main_sugar[-1] == 'Xyl' and len(main_bond) > 1 and main_bond[-1] == 'β 2')
    if tucked_end:
        main_sugar_x_pos[-1] -= 1

    # Calculate x positions for branches
    def calculate_x_positions(sugars: list, connections: list, parent_x_positions: list, level: int = 1):
        x_positions = []
        for j, branch in enumerate(sugars):
            if level == 1:
                start_x = parent_x_positions[connections[j][1]]
            else:
                parent_branch, parent_idx = connections[j]
                start_x = parent_x_positions[parent_branch][parent_idx]
            tmp = [start_x + 1 + k for k in range(len(branch))]
            if branch[-1] in {'Fuc', 'Xyl'}:
                tmp[-1] -= 1
            x_positions.append(tmp)
        return x_positions

    # Calculate x positions for all branch levels
    lv_x_pos = [calculate_x_positions(lv_sugar[0], lv_connection[0], main_sugar_x_pos)]
    for i in range(1, len(lv_sugar)):
        lv_x_pos.append(calculate_x_positions(lv_sugar[i], lv_connection[i], lv_x_pos[i - 1], level = i + 1))
    # Initialize y positions; ALL START AT Y=0 (except Fuc)
    main_sugar_y_pos = [2 if tucked_end and i == len(main_sugar) - 1 else 0 for i in range(len(main_sugar))]
    lv_y_pos = [[[2 if s == "Fuc" or (s == "Xyl" and i == len(sugars) - 1) else 0 for i, s in enumerate(sugars)] for
                 sugars in level] for level in lv_sugar]
    # A terminal Fuc or Xyl is tucked into its parent's column, so it has to leave its parent's row, even when that
    # parent is a Fuc lifted the same way
    for ys, sugars in zip(unwrap(lv_y_pos), unwrap(lv_sugar)):
        if len(sugars) > 1 and sugars[-1] in {'Fuc', 'Xyl'}:
            ys[-1] = ys[-2] + 2
    SPACING = 1
    # Main chain goes down, branches go up
    branch_points = {conn[1] for conn in lv_connection[0]}
    # For each branch point, main chain beyond it goes down
    for parent_idx in sorted(branch_points):
        # Core fucose special case (don't push down main chain)
        branch_indices = [j for j, conn in enumerate(lv_connection[0]) if conn[1] == parent_idx]
        core_branches = [lv_sugar[0][j] for j in branch_indices]
        is_core_fuc = all(j in [['Fuc'], ['Xyl']] for j in core_branches)
        is_fuc_partner = main_sugar[parent_idx + 1] == 'Fuc' or (tucked_end and parent_idx + 2 == len(main_sugar))
        if not is_core_fuc and not is_fuc_partner:
            branch_sugar = max(core_branches, key = len)
            connected, deeper_connected = branch_indices, []
            l1_main_chain_indices = [k for k, conn in enumerate(lv_connection[0]) if
                                     conn[1] > parent_idx and lv_sugar[0][k] not in [['Fuc'], ['Gal']]]
            l1_main_chain_branches = [lv_sugar[0][k] for k in l1_main_chain_indices]
            main_chain_indices, deeper_main_chain = l1_main_chain_indices, []
            for lvl in range(1, len(lv_sugar)):
                connected = [k for k, conn in enumerate(lv_connection[lvl]) if conn[0] in connected]
                deeper_connected.append([lv_sugar[lvl][k] for k in connected])
                main_chain_indices = [k for k, conn in enumerate(lv_connection[lvl]) if conn[0] in main_chain_indices]
                deeper_main_chain.append([lv_sugar[lvl][k] for k in main_chain_indices])
            l2_connected_branches = deeper_connected[0] if deeper_connected else []
            has_fuc = ('Fuc' in branch_sugar) or ('Fuc' in unwrap(unwrap(deeper_connected)))
            has_own_fuc = 'Fuc' in unwrap(l1_main_chain_branches) or 'Fuc' in unwrap(unwrap(deeper_main_chain))
            has_bisecting = parent_idx > 0 and any('GlcNAc' in b[0] for b in core_branches) and main_sugar[parent_idx] == 'Man' and main_bond[
                parent_idx - 1] == 'β 4'
            has_triple_branch = len(branch_indices) == 2 and not 'Xyl' in unwrap(core_branches)
            is_highly_branched = len(l2_connected_branches) > 1
            l2_connected_branches = [k for k in l2_connected_branches if k not in [['Fuc'], ['Gal']]]
            max_l2_len = max((len(b) for b in l2_connected_branches), default = 0)
            max_l1_len = max((len(b) for b in l1_main_chain_branches if b not in [['Fuc'], ['Gal']]), default = 0)
            spacing_spec = SPACING + has_fuc * SPACING + has_own_fuc * SPACING + (
                    max_l1_len > 0) * 0.25 * SPACING + is_highly_branched * 0.5 * SPACING
            if (max_l2_len > 0) and (spacing_spec < 2.5 or (has_fuc and has_own_fuc)):
                spacing_spec += 0.5 * SPACING
            if (has_bisecting or has_triple_branch) and spacing_spec < 1.1:
                spacing_spec += SPACING
            # Push main chain down after branch point
            for i in range(parent_idx + 1, len(main_sugar)):
                main_sugar_y_pos[i] += spacing_spec
    # All branches go up
    for j, conn in enumerate(lv_connection[0]):
        parent_idx = conn[1]
        branch_sugar = lv_sugar[0][j]
        # Special case for core fucose
        if parent_idx == 0 and branch_sugar == ['Fuc'] and lv_bond[0][j] == ['α 6']:
            # Core fucose goes up
            lv_y_pos[0][j] = [-2 * SPACING] * len(branch_sugar)
        else:
            is_bisecting = parent_idx > 0 and branch_sugar[0] in ['GlcNAc'] and main_sugar[parent_idx] == 'Man' and main_bond[
                parent_idx - 1] == 'β 4'
            is_leading_xyl = main_sugar[-1] == 'Xyl'
            is_fuc_partner = main_sugar[parent_idx + 1] == 'Fuc' or (tucked_end and parent_idx + 2 == len(main_sugar))
            parent_branches = [(k, c) for k, c in enumerate(lv_connection[0]) if
                               c[1] == parent_idx]  # + 1 from main chain
            is_triple_branch = len(parent_branches) == 2 and j == parent_branches[0][0]
            # All other branches go up by spacing amount
            if len(branch_sugar) == 1 and branch_sugar[0] in ['Fuc', 'Xyl']:
                lv_y_pos[0][j][0] = main_sugar_y_pos[parent_idx] + 2 * SPACING
            elif is_leading_xyl and j == 0:
                lv_y_pos[0][j] = [p + main_sugar_y_pos[parent_idx] + SPACING - lv_y_pos[0][j][0] for p in
                                  lv_y_pos[0][j]]
            elif len(branch_sugar) == 1 and (is_bisecting or is_fuc_partner or is_triple_branch):
                lv_y_pos[0][j][0] = main_sugar_y_pos[parent_idx]
            elif len(branch_sugar) == 1:
                lv_y_pos[0][j][0] = main_sugar_y_pos[parent_idx] - SPACING
            else:
                offset = main_sugar_y_pos[parent_idx + 1] - main_sugar_y_pos[parent_idx]
                shift_amount = main_sugar_y_pos[parent_idx] - offset
                lv_y_pos[0][j] = [p + shift_amount for p in lv_y_pos[0][j]]
    for parent_idx in branch_points:
        sibs = [j for j, conn in enumerate(lv_connection[0]) if conn[1] == parent_idx and len(lv_sugar[0][j]) == 1]
        order = sorted(sibs, key = lambda j: -lv_y_pos[0][j][0])
        for k in range(1, len(order)):
            room = lv_y_pos[0][order[k - 1]][0] - 2 * SPACING
            if lv_y_pos[0][order[k]][0] > room:
                lv_y_pos[0][order[k]][0] = room

    def process_branch_level(level_sugar, level_y_pos, level_connection, next_level_sugar, next_level_connection,
                             parent_level_y_pos, parent_level_sugar):
        # At each branch point, push remaining sugars down
        for idx, (parent_branch, parent_idx) in enumerate(level_connection):
            is_fuc_partner = parent_level_sugar[parent_branch][parent_idx+1] == 'Fuc' if parent_idx+1 < len(parent_level_sugar[parent_branch]) else False
            if not is_fuc_partner and level_sugar[idx][0] != 'Fuc':
                branch_indices = [j for j, conn in enumerate(level_connection) if conn[0] == parent_branch and conn[1] == parent_idx]
                core_branches = [level_sugar[j] for j in branch_indices]
                branch_sugar = max(core_branches, key = len)
                has_next_level = len(next_level_sugar) > 0
                next_level_connected_branches = [next_level_sugar[k] for k, conn in enumerate(next_level_connection) if conn[0] in branch_indices] if has_next_level else []
                has_fuc = ('Fuc' in branch_sugar) or ('Fuc' in unwrap(next_level_connected_branches) if has_next_level else False)
                spacing_spec = SPACING + has_fuc*SPACING
                for i in range(parent_idx + 1, len(parent_level_y_pos[parent_branch])):
                    parent_level_y_pos[parent_branch][i] += spacing_spec
        # Branches go up
        for j, (parent_branch, parent_idx) in enumerate(level_connection):
            parent_y = parent_level_y_pos[parent_branch][parent_idx]
            is_fuc_partner = parent_level_sugar[parent_branch][parent_idx+1] == 'Fuc' if parent_idx+1 < len(parent_level_sugar[parent_branch]) else False
            if len(level_sugar[j]) == 1 and level_sugar[j][0] in ['Fuc', 'Xyl']:
                nxt = parent_level_y_pos[parent_branch][parent_idx + 1] if parent_idx + 1 < len(
                    parent_level_y_pos[parent_branch]) else parent_y
                away = -1 if nxt > parent_y else 1
                level_y_pos[j][0] = parent_y + away * 2 * SPACING
            elif len(level_sugar[j]) == 1 and is_fuc_partner:
                level_y_pos[j][0] = parent_y
            else:
                offset = parent_level_y_pos[parent_branch][parent_idx+1] - parent_y if parent_idx+1 < len(parent_level_y_pos[parent_branch]) else 0
                shift_amount = parent_y - offset
                level_y_pos[j] = [p + shift_amount for p in level_y_pos[j]]
        return level_y_pos

    for i in range(1, len(lv_sugar)):
        next_sugar, next_connection = (lv_sugar[i + 1], lv_connection[i + 1]) if i + 1 < len(lv_sugar) else ([], [])
        lv_y_pos[i] = process_branch_level(lv_sugar[i], lv_y_pos[i], lv_connection[i], next_sugar, next_connection,
                                           lv_y_pos[i - 1], lv_sugar[i - 1])
    # The rules above fix the arrangement; this pass fixes the spacing, by pushing whole subtrees apart (or together) until every column has exactly the clearance its symbols need
    lanes_x = [[main_sugar_x_pos]] + lv_x_pos
    lanes_y = [[main_sugar_y_pos]] + lv_y_pos
    parent, children = {}, {}
    for i in range(1, len(main_sugar_y_pos)):
        parent[(0, 0, i)] = (0, 0, i - 1)
    for lane, conns in enumerate(lv_connection, start = 1):
        for b, conn in enumerate(conns):
            for i in range(len(lanes_y[lane][b])):
                parent[(lane, b, i)] = (lane, b, i - 1) if i else (
                    (0, 0, conn[1]) if lane == 1 else (lane - 1, conn[0], conn[1]))
    for node, par in parent.items():
        children.setdefault(par, []).append(node)

    def subtree(node: tuple):
        out, stack = [], [node]
        while stack:
            n = stack.pop()
            out.append(n)
            stack.extend(children.get(n, []))
        return out

    def contour(nodes: list):
        # Per column, how far this subtree's symbols reach up and down; modification labels ride inside the clearance band and are not measured
        c = {}
        for lane, b, i in nodes:
            x, y = lanes_x[lane][b][i], lanes_y[lane][b][i]
            lo, hi = c.get(x, (y - 0.5, y + 0.5))
            c[x] = (min(lo, y - 0.5), max(hi, y + 0.5))
        return c

    CLEARANCE, PARENT_SPAN = 1.0, 1.0
    for node in sorted(children, key = lambda n: len(subtree(n))):
        kids = children[node]
        if len(kids) < 2:
            continue
        px, py = lanes_x[node[0]][node[1]][node[2]], lanes_y[node[0]][node[1]][node[2]]
        chain, pins, free, taken = (node[0], node[1], node[2] + 1), [], [], {(px, py)}
        for k in sorted(kids, key = lambda k: abs(lanes_y[k[0]][k[1]][k[2]] - py)):
            slot = (lanes_x[k[0]][k[1]][k[2]], lanes_y[k[0]][k[1]][k[2]])
            # A Fuc tucked into its parent's column and a bisecting GlcNAc drawn level with it sit where SNFG convention put them; a second residue claiming the same slot cannot
            if (slot[0] == px or slot[1] == py) and slot not in taken:
                pins.append(k)
                taken.add(slot)
            else:
                free.append(k)
        acc = contour([node] + unwrap([subtree(k) for k in pins]))
        # The chain continues downwards and every side branch goes up; with no free chain residue to hold the lower side, the branches keep the side the rules above chose for them and straddle the parent instead of stacking above it
        sides = {k: 1 if (k == chain if chain in free else lanes_y[k[0]][k[1]][k[2]] > py) else -1 for k in free}
        for k in free:
            # A second residue tucked into the parent's column cannot stack beyond the pinned one, as its bond would run
            # through it; it takes the other side
            if lanes_x[k[0]][k[1]][k[2]] == px and any(lanes_x[p[0]][p[1]][p[2]] == px and (
                    lanes_y[p[0]][p[1]][p[2]] > py) == (sides[k] > 0) for p in pins):
                sides[k], lift = -sides[k], 2 * (py - lanes_y[k[0]][k[1]][k[2]])
                for lane, b, i in subtree(k):
                    lanes_y[lane][b][i] += lift
        seed, near = dict(acc), {}
        for side in (1, -1):
            for k in sorted([k for k in free if sides[k] == side],
                            key = lambda k: side * lanes_y[k[0]][k[1]][k[2]]):
                nodes = subtree(k)
                c = contour(nodes)
                shared = [x for x in c if x in acc]
                delta = py + side * PARENT_SPAN - lanes_y[k[0]][k[1]][k[2]]
                if shared:
                    gap = min((c[x][0] - acc[x][1]) if side > 0 else (acc[x][0] - c[x][1]) for x in shared)
                    delta = side * max(side * delta, CLEARANCE - gap)
                for lane, b, i in nodes:
                    lanes_y[lane][b][i] += delta
                near.setdefault(side, side * (lanes_y[k[0]][k[1]][k[2]] - py))
                for x, (lo, hi) in contour(nodes).items():
                    plo, phi = acc.get(x, (lo, hi))
                    acc[x] = (min(plo, lo), max(phi, hi))
        # Placing each side at its own minimum leaves one linkage of the branch point far longer than the other; sliding both sides together splits the separation evenly, at no cost in height
        if len(near) == 2 and near[1] != near[-1]:
            far = -1 if near[-1] > near[1] else 1
            moving = unwrap([subtree(k) for k in free])
            c = contour(unwrap([subtree(k) for k in free if sides[k] == far]))
            shared = [x for x in c if x in seed]
            room = min((c[x][0] - seed[x][1]) if far > 0 else (seed[x][0] - c[x][1]) for x in
                       shared) - CLEARANCE if shared else abs(near[1] - near[-1])
            shift = -far * min(abs(near[1] - near[-1]) / 2, max(0, room))
            for lane, b, i in moving:
                lanes_y[lane][b][i] += shift

    def extract_conformation(sugar_modifications: list):
        if sugar_modifications and isinstance(sugar_modifications[0], list):
            return [[k.group() if k is not None else '' for k in j] for j in [[re.search(_CONF_PATTERN, k) for k in j] for j in sugar_modifications]], \
                [[re.sub(_CONF_PATTERN, '', k) for k in j] for j in sugar_modifications]
        else:
            return [k.group() if k is not None else '' for k in [re.search(_CONF_PATTERN, k) for k in sugar_modifications]], \
                [re.sub(_CONF_PATTERN, '', k) for k in sugar_modifications]

    main_conf, main_sugar_modification = extract_conformation(main_sugar_modification)
    lv_conf, lv_sugar_modification = map(list, zip(*[extract_conformation(m) for m in lv_sugar_modification]))
    node_positions = {n: (0, 0, i) for i, n in enumerate(main_label_sugar[::-1])}
    for lane, branches in enumerate(branch_levels, start = 1):
        for b, branch in enumerate(branches):
            node_positions.update({n: (lane, b, i) for i, n in enumerate(branch['sugar_nodes'])})
    data_combined = [[main_sugar, main_sugar_x_pos, main_sugar_y_pos, main_sugar_modification, main_bond, main_conf,
                      main_sugar_highlight, main_bond_highlight]] + [
                        [lv_sugar[i], lv_x_pos[i], lv_y_pos[i], lv_sugar_modification[i], lv_bond[i], lv_connection[i],
                         lv_conf[i],
                         lv_sugar_label[i], lv_bond_label[i]] for i in range(len(lv_sugar))] + [node_positions]
    return data_combined


def draw_bracket(
        x: float, # X coordinate
        y_min_max: list[float], # [Min Y, Max Y] coordinates
        drawing: draw.Drawing, # Glycan drawing to be modified
        direction: str = 'right', # Bracket direction ("left", "right")
        dim: float = 50, # Base dimension for scaling
        highlight: str = 'show', # Highlight state
        deg: float = 0 # Rotation angle in degrees
) -> None:
    "Draws bracket shape at specified position and dimensions"
    col_dict = col_dict_transparent if highlight == 'hide' else col_dict_base
    x_common = -x * dim
    y_min = y_min_max[0] * dim - 0.75 * dim
    y_max = y_min_max[1] * dim + 0.75 * dim
    # Vertical
    offset = 0.25 * dim * (1 if direction == 'right' else -1)
    g = draw.Group(transform = f'rotate({deg} {x_common} {(y_min + y_max)/2})')
    p = draw.Path(stroke_width = 0.04 * dim, stroke = col_dict['black'])
    p.M(x_common, y_max).L(x_common, y_min)
    p.M(x_common - offset / 12.5, y_min).L(x_common + offset, y_min)
    p.M(x_common - offset / 12.5, y_max).L(x_common + offset, y_max)
    g.append(p)
    drawing.append(g)


def is_jupyter() -> bool:
    "Detects if code is running in Jupyter notebook environment"
    try:
        from IPython import get_ipython
        return 'IPKernelApp' in get_ipython().config  # Check if in IPython kernel
    except (AttributeError, ImportError):
        return False


def display_svg_with_matplotlib(
        svg_data: Any, # SVG drawing object
        chem: bool = False, # Whether svg_data comes from RDKit chemical
        shadow: bool = False,  # Draw a soft drop shadow under the monosaccharide symbols
        sticker: bool = False  # Cut the whole structure out as a die-cut sticker
) -> None:
    "Renders SVG using matplotlib for non-Jupyter environments"
    _, convert_svg_to_png = _get_glycorender()
    import matplotlib.pyplot as plt
    # Get original SVG dimensions and scale them up
    width, height = getattr(svg_data, 'width', 800), getattr(svg_data, 'height', 800)
    svg_data = svg_data if isinstance(svg_data, str) else svg_data.as_svg()
    # Convert to PNG with larger dimensions
    png_output = convert_svg_to_png(svg_data, output_width = width, background = (1.0, 1.0, 1.0), shadow = shadow,
                                    sticker = sticker, output_height = height, scale = 2.0, return_bytes = True, chem = chem)
    img = plt.imread(BytesIO(png_output), format = 'png')
    dpi = plt.rcParams['figure.dpi']
    fig = plt.figure(figsize = (img.shape[1] / dpi, img.shape[0] / dpi))
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    plt.close(fig)


def _written_positions(graph: nx.DiGraph # Glycan graph from glycan_to_nxGraph
                       ) -> dict[int, int]: # Node: its position in the sequence as written
    "glycan_to_nxGraph numbers the main part from 0 and the floating parts after it, although they are written before it"
    main = nx.node_connected_component(graph.to_undirected(as_view = True), 0)
    return {n: i for i, n in enumerate(sorted(graph, key = lambda n: (n in main, n)))}


def process_per_residue(
        draw_this: str, # reordered IUPAC-condensed glycan sequence
        per_residue: list[float], # Scalar values per residue
        glycan: str, # original IUPAC-condensed glycan sequence
) -> dict[int, float]: # Value per sugar node of the drawn sequence
    "Maps per-residue scalar values onto the sugar nodes of the drawn sequence"
    n_residues = (len(glycan_to_nxGraph(draw_this)) + 1) // 2
    if n_residues != len(per_residue):
        raise ValueError(
            f"per_residue has {len(per_residue)} values but {glycan} has {n_residues} monosaccharides to color")
    if glycan != draw_this:
        g1, g2 = glycan_to_nxGraph(glycan), glycan_to_nxGraph(draw_this)
        _, mappy = compare_glycans(g2, g1, return_matches = True)
        pos1, at2 = _written_positions(g1), {i: n for n, i in _written_positions(g2).items()}
        per_residue = [per_residue[pos1[mappy[at2[i * 2]]] // 2] for i in range(len(per_residue))]
    return {i * 2: v for i, v in enumerate(per_residue)}


def process_per_linkage(
        draw_this: str, # reordered IUPAC-condensed glycan sequence
        highlight_linkages: list[int], # Which linkages to highlight
        glycan: str, # original IUPAC-condensed glycan sequence
) -> dict[int, bool]: # Flag per linkage node of the drawn sequence
    "Maps which linkages to highlight onto the linkage nodes of the drawn sequence"
    n_linkages = len(glycan_to_nxGraph(glycan)) // 2
    if any(not 0 <= i < n_linkages for i in highlight_linkages):
        raise ValueError(
            f"highlight_linkages {highlight_linkages} has to index the {n_linkages} linkages of {glycan}, starting from 0")
    per_linkage = [i in highlight_linkages for i in range(n_linkages)]
    if glycan != draw_this:
        g1, g2 = glycan_to_nxGraph(glycan), glycan_to_nxGraph(draw_this)
        _, mappy = compare_glycans(g2, g1, return_matches = True)
        # A linkage moves with the residue it leaves, which precedes it in the written sequence
        pos1, at2 = _written_positions(g1), {i: n for n, i in _written_positions(g2).items()}
        per_linkage = [per_linkage[pos1[mappy[at2[i * 2]]] // 2] for i in range(len(per_linkage))]
    return {i * 2 + 1: v for i, v in enumerate(per_linkage)}


mono_list = ['Glc', 'GlcNAc', 'GlcA', 'Man', 'ManNAc', 'Gal', 'GalNAc', 'Gul', 'GulNAc',
             'Alt', 'AltNAc', 'All', 'AllNAc', 'Neu5Ac', 'Tal', 'TalNAc', 'Neu5Gc', 'Ido', 'IdoNAc', 'IdoA', 'Fuc']

chem_cols = ['#CDE7EF', '#CDE7EF', '#CDE7EF',     # blue
             '#CDE9DF', '#CDE9DF',                # green
             '#FFF6DE', '#FFF6DE',                # yellow
             '#FDE7E0', '#FDE7E0',                # orange
             '#FDF0F1', '#FDF0F1',                # pink
             '#F1E6ED', '#F1E6ED', '#F1E6ED',     # purple
             '#EEF8FB', '#EEF8FB', '#EEF8FB',     # light blue
             '#F1E9E5', '#F1E9E5', '#F1E9E5',     # brown
             '#F7E0E0']                           # red

chem_cols_alpha = ['#0385AE', '#0385AE', '#0385AE',     # blue
                   '#058F60', '#058F60',                # green
                   '#FCC326', '#FCC326',                # yellow
                   '#EF6130', '#EF6130',                # orange
                   '#F39EA0', '#F39EA0',                # pink
                   '#A15989', '#A15989', '#A15989',     # purple
                   '#91D3E3', '#91D3E3', '#91D3E3',     # light blue
                   '#9F6D55', '#9F6D55', '#9F6D55',     # brown
                   '#C23537']                           # red


def get_mono_atoms(
        draw_this: str, # IUPAC-condensed glycan sequence
        mono_list: str | list[str]  # Monosaccharide(s) to highlight
) -> tuple[str, dict[int, int]]:  # (SMILES, {atom index: index into mono_list})
    "Maps every atom of a glycan's SMILES onto the monosaccharide it was built from"
    mono_list = [mono_list] if isinstance(mono_list, str) else mono_list
    from glycowork.motif.smiles import glycan_to_smiles
    smiles, owners = glycan_to_smiles(draw_this, mapping = True)
    graph = glycan_to_nxGraph(draw_this)
    cores = {node: get_core(graph.nodes[node]['string_labels']) for node in set(owners)}
    return smiles, {atom: mono_list.index(cores[owner]) for atom, owner in enumerate(owners) if cores[owner] in mono_list}


def color_by_mono(
        mol: Any, # RDKit molecule object
        atom_monos: dict[int, int], # {atom index: index into mono_list}
        atom_colors: dict[int, list], # Color map to fill for atoms
        bond_colors: dict[int, list], # Color map to fill for bonds
        alpha: bool = True, # Use alpha-adjusted colors
        hex_codes: bool = True # Return hex color codes
) -> None:
    "Colours every atom by the monosaccharide it came from, and every bond whose two atoms agree"
    for atom, i in atom_monos.items():
        add_colors_to_map([atom], atom_colors, i, alpha = alpha, hex_codes = hex_codes)
    for bond in mol.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if atom_monos.get(begin, -1) == atom_monos.get(end, -2):
            add_colors_to_map([bond.GetIdx()], bond_colors, atom_monos[begin], alpha = alpha, hex_codes = hex_codes)


def add_colors_to_map(
        els: list[int], # Element indices
        cols: dict[int, list], # Color map dictionary
        col_num: int, # Color index
        alpha: bool = True, # Use alpha-adjusted colors
        hex_codes: bool = True # Return hex color codes
) -> None:
    "Adds color assignments to mapping dictionary for chemical structure visualization"
    from matplotlib.colors import ColorConverter
    color = chem_cols_alpha[col_num] if alpha else chem_cols[col_num]
    color = color if hex_codes else ColorConverter().to_rgb(color)
    for el in els:
        cols.setdefault(el, [])
        if color not in cols[el]: cols[el].append(color)


def draw_chem2d(
        draw_this: str, # IUPAC-condensed glycan sequence
        mono_list: str | list[str], # Monosaccharide(s) to highlight
        filepath: str | Path | None = None # Output file path
) -> Any: # IPython SVG display object
    "Creates 2D chemical structure drawing with highlighted monosaccharides using RDKit"
    # Adapted from https://github.com/rdkit/rdkit/blob/master/Docs/Book/data/test_multi_colours.py
    try:
        from rdkit.Chem import MolFromSmiles
        from rdkit.Chem.Draw import PrepareMolForDrawing
        from rdkit.Chem.Draw.rdMolDraw2D import MolDraw2DSVG
    except ImportError:
        raise ImportError(
            "You must install the 'chem' dependencies to use this feature. Try 'pip install glycowork[chem]'.")
    smiles, atom_monos = get_mono_atoms(draw_this, mono_list)
    mol = PrepareMolForDrawing(
        MolFromSmiles(smiles))  # only appends hydrogens, so the heavy-atom indices of atom_monos still hold
    atom_colors, bond_colors = {}, {}
    color_by_mono(mol, atom_monos, atom_colors, bond_colors, hex_codes = False)
    d = MolDraw2DSVG(250, 250)
    d.drawOptions().fillHighlights = True
    d.drawOptions().useBWAtomPalette()
    d.drawOptions().rotate = 180
    d.DrawMoleculeWithHighlights(mol, '', atom_colors, bond_colors, {}, {}, -1)
    d.FinishDrawing()
    svg_data = d.GetDrawingText()
    if filepath:
        filepath = Path(filepath)
        filepath = filepath.with_name(filepath.name.replace('?', '_'))
        suffix = filepath.suffix.lower()
        if suffix not in {'.svg', '.pdf', '.png'}:
            raise ValueError(f"Cannot save to '{filepath.name}': filepath has to end in .svg, .pdf, or .png")
        filepath.parent.mkdir(parents = True, exist_ok = True)
        if suffix == '.svg':
            with open(filepath, 'w', encoding = "utf-8") as f:
                f.write(svg_data)
        else:
            convert_svg_to_pdf, convert_svg_to_png = _get_glycorender()
            (convert_svg_to_pdf if suffix == '.pdf' else convert_svg_to_png)(svg_data, str(filepath), chem = True)
    if not is_jupyter():
        return display_svg_with_matplotlib(svg_data, chem = True)
    from IPython.display import SVG
    return SVG(svg_data)


def draw_chem3d(
        draw_this: str, # IUPAC-condensed glycan sequence
        mono_list: str | list[str], # Monosaccharide(s) to highlight
        filepath: str | Path | None = None, # Output file path for PDB
        pdb_file: str | Path | None = None  # already existing glycan structure
) -> None:
    "Generates 3D chemical structure model with highlighted monosaccharides using RDKit and py3Dmol"
    # Adapted from https://github.com/rdkit/rdkit/blob/master/Docs/Book/data/test_multi_colours.py and https://github.com/rdkit/rdkit/blob/master/Docs/Book/GettingStartedInPython.rst
    try:
        from rdkit.Chem import MolFromSmiles, AddHs, RemoveHs, MolToPDBFile, MolFromPDBFile
        from rdkit.Chem.AllChem import EmbedMolecule, MMFFOptimizeMolecule
        if is_jupyter():
            from rdkit.Chem.Draw import IPythonConsole
            import py3Dmol
        else:
            from rdkit.Chem.Draw import rdDepictor
            from rdkit.Chem.Draw.rdMolDraw2D import MolDraw2DSVG
    except ImportError:
        raise ImportError("You must install the 'chem' dependencies to use this feature. Try 'pip install glycowork[chem]'.")
    mono_list = [mono_list] if isinstance(mono_list, str) else mono_list
    smiles, atom_monos = get_mono_atoms(draw_this, mono_list)
    smiles_mol = MolFromSmiles(smiles)
    from_pdb = False
    if pdb_file:
        mol = MolFromPDBFile(str(pdb_file))
        from_pdb = True
    else:
        # Try glycontact for a realistic GlycoShape conformer; fall back to RDKit
        glycoshape_mol = None
        try:
            from glycontact.process import fetch_pdbs
            pdb_paths = fetch_pdbs(draw_this)
            if pdb_paths and isinstance(pdb_paths[0], Path):
                glycoshape_mol = MolFromPDBFile(str(pdb_paths[0]))
        except Exception:
            pass
        if glycoshape_mol is not None:
            mol = glycoshape_mol
            from_pdb = True
        else:
            mol = AddHs(smiles_mol)
            if EmbedMolecule(mol, randomSeed = 42) == -1 and EmbedMolecule(mol, randomSeed = 42,
                                                                           useRandomCoords = True) == -1:
                raise ValueError(f"RDKit could not embed a 3D conformer for {draw_this}")
            MMFFOptimizeMolecule(mol)
            mol = RemoveHs(mol)
            print("Disclaimer: The conformer generated using RDKit and MMFFOptimizeMolecule is not intended to be a replacement for a 'real' conformer analysis tool. Install glycontact and run this again for improved conformers.")
    # Color atoms by monosaccharide after mol is finalized
    atom_colors, bond_colors = {}, {}
    if from_pdb:
        for atom in mol.GetAtoms():
            info = atom.GetPDBResidueInfo()
            mono_name = PDB_TO_IUPAC.get(info.GetResidueName().strip(), '').split('(')[0].strip() if info else ''
            if mono_name and get_core(mono_name) in mono_list:
                add_colors_to_map([atom.GetIdx()], atom_colors, mono_list.index(get_core(mono_name)), alpha = False)
        # ROH reducing end oxygen belongs to adjacent monosaccharide
        for atom in mol.GetAtoms():
            info = atom.GetPDBResidueInfo()
            if info and info.GetResidueName().strip() == 'ROH' and atom.GetIdx() not in atom_colors:
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetIdx() in atom_colors:
                        atom_colors[atom.GetIdx()] = atom_colors[neighbor.GetIdx()]
                        break
    else:
        color_by_mono(mol, atom_monos, atom_colors, bond_colors, alpha = False)
    atom_colors = {k: ['#ECECEC'] if len(v) > 1 else v for k, v in atom_colors.items()}
    if filepath:
        filepath = Path(filepath)
        if filepath.suffix.lower() == '.pdb':
            MolToPDBFile(mol, filepath)
        else:
            print("3D structure can only be saved as .pdb file.")
    if is_jupyter():
        v = py3Dmol.view(width = 500, height = 300)
        v.removeAllModels()
        IPythonConsole.addMolToView(mol, v)
        for atom_idx, colors in atom_colors.items():
            v.setStyle({'serial': atom_idx}, {'stick': {'color': colors[0]}})
        v.zoomTo()
        v.show()
    else:
        rdDepictor.Compute2DCoords(mol, clearConfs = False)
        drawer = MolDraw2DSVG(500, 500)
        drawer.drawOptions().addStereoAnnotation = True
        drawer.drawOptions().addAtomIndices = False
        drawer.drawOptions().bondLineWidth = 2
        drawer.DrawMolecule(mol, highlightAtoms = list(atom_colors.keys()),
                            highlightAtomColors = {k: tuple(int(v[0].lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4)) for k, v in atom_colors.items()})
        drawer.FinishDrawing()
        display_svg_with_matplotlib(drawer.GetDrawingText(), chem = True)


class GlycanDrawing:
    def __init__(self, drawing_obj, shadow = False, sticker = False, vertical = False, alt_text = None):
        self.drawing_obj = drawing_obj
        self.shadow = shadow
        self.sticker = sticker
        self.vertical = vertical
        self.alt_text = alt_text

    def as_svg(self):
        data = self.drawing_obj.as_svg()
        return data.replace('<svg ', f'<svg aria-label="{self.alt_text}" role="img" ', 1) if self.alt_text else data

    def save_svg(self, filepath):
        data = self.as_svg()
        # drawsvg has no cut layer of its own, so route the SVG through glycorender as well
        if self.shadow or self.sticker:
            from glycorender.render import pdf_to_svg_bytes
            data = pdf_to_svg_bytes(data, shadow = self.shadow, sticker = self.sticker)
            data = data.replace('<svg ', f'<svg aria-label="{self.alt_text}" role="img" ', 1) if self.alt_text else data
        with open(filepath, 'w', encoding = "utf-8") as f:
            f.write(_flatten_text_paths(data, turn = 90 if self.vertical else 0))

    def save(self, filepath):
        "Saves as .svg, .pdf, or .png (300 dpi) and returns the drawing, so saves can be chained"
        filepath = Path(filepath)
        suffix = filepath.suffix.lower()
        if suffix not in {'.svg', '.pdf', '.png'}:
            raise ValueError(f"Cannot save to '{filepath.name}': filepath has to end in .svg, .pdf, or .png")
        filepath.parent.mkdir(parents = True, exist_ok = True)
        if suffix == '.svg':
            self.save_svg(filepath)
        elif suffix == '.pdf':
            convert_svg_to_pdf, _ = _get_glycorender()
            convert_svg_to_pdf(self.as_svg(), str(filepath), shadow = self.shadow, sticker = self.sticker)
        else:
            _, convert_svg_to_png = _get_glycorender()
            # print resolution; glycorender records the 300 dpi, so the physical size still matches the PDF
            convert_svg_to_png(self.as_svg(), str(filepath), scale = 300 / 72, shadow = self.shadow,
                               sticker = self.sticker)
        return self

    def _repr_png_(self):
        _, convert_svg_to_png = _get_glycorender()
        # Rendered at twice its size but displayed at its size, so high-DPI screens show it crisp
        png = convert_svg_to_png(self.as_svg(), None, scale = 2.0, return_bytes = True, shadow = self.shadow,
                                 sticker = self.sticker, background = (1.0, 1.0, 1.0))
        return png, {'width': self.drawing_obj.width, 'height': self.drawing_obj.height}


def _finish_drawing(
        d: draw.Group, # Drawn content, in user space
        in_glycan: str, # Glycan as passed by the caller, for filenames
        alt_text: str, # ALT text for accessibility
        id_key: tuple, # Everything that makes this drawing distinct, hashed into its element IDs
        vertical: bool = False, # Draw vertically
        dim: float = 50, # Base dimension for scaling
        filepath: str | Path | None = None, # Output file path
        suppress: bool = False, # Suppress display
        shadow: bool = False, # Draw a soft drop shadow under the monosaccharide symbols
        sticker: bool = False # Cut the whole structure out as a die-cut sticker
) -> Any: # Drawing object
    "Crops, saves, and returns a finished GlycoDraw canvas, shared by structures and compositions"
    alt_text = html.escape(alt_text)
    # A bond into an invisible placeholder dangles, so it stops half a symbol short of it; trimmed here rather than by the
    # renderer, so that the SVG, the PNG/PDF, the crop and the label centered on the bond all agree
    stack, blanks, bonds, carriers = [d], set(), [], {}
    while stack:
        el = stack.pop()
        a = getattr(el, 'args', {}) or {}
        stack.extend(getattr(el, 'children', []) or [])
        if isinstance(el, draw.Circle) and a.get('fill') == a.get('stroke') == 'none':
            blanks.add((round(a['cx'], 6), round(a['cy'], 6)))
        elif a.get('class') == 'snfg-linkage':
            bonds.append(el)
        elif 'startOffset' in a:
            carriers[id(a['xlink:href'])] = el
    rising = set()
    for bond in bonds:
        ends = [float(k) for k in _SVG_NUMBER.findall(re.sub(r'[A-DF-Za-df-z]', ' ', bond.args['d']))]
        (x0, y0), (x1, y1) = ends[:2], ends[-2:]
        if abs(x1 - x0) < 1e-6:
            rising.add((round(x0, 6), round(max(y0, y1), 6)))
        length = np.hypot(x1 - x0, y1 - y0)
        if length <= dim:
            continue
        t0 = dim / 2 / length if (round(x0, 6), round(y0, 6)) in blanks else 0
        t1 = 1 - dim / 2 / length if (round(x1, 6), round(y1, 6)) in blanks else 1
        bond.args['d'] = f'M{x0 + t0 * (x1 - x0)},{y0 + t0 * (y1 - y0)} L{x0 + t1 * (x1 - x0)},{y0 + t1 * (y1 - y0)}'
        if id(bond) in carriers and t1 - t0 < 1:
            # Its label stays where the whole bond centered it, clear of the symbol, instead of sliding onto it with the
            # trimmed end
            carriers[id(bond)].args['startOffset'] = f'{(0.5 - t0) / (t1 - t0) * 100:.4f}%'
    # SNFG says not to rotate symbols: in vertical mode each turns back upright about its center, its modification label
    # beside it; otherwise only a label that a bond rising straight up from its symbol would strike through steps right
    # of that bond
    stack = [d]
    while stack:
        el = stack.pop()
        stack.extend(getattr(el, 'children', []) or [])
        if (getattr(el, 'args', {}) or {}).get('class') != 'snfg-symbol':
            continue
        cx, cy = el.center
        if vertical:
            el.args['transform'] = f'rotate(-90 {cx} {cy})'
        elif (round(cx, 6), round(cy, 6)) not in rising:
            continue
        for label in el.children:
            if isinstance(label, draw.Text) and any(t.args.get('dy') == '-3.15em' for t in label.children[0].children):
                label.args['text-anchor'], label.children[0].args['startOffset'] = 'start', '0'
                # dy = -3.15em lifts the baseline off its carrier, so the carrier sits that far below it
                base_y = cy + 3.54 * label.args['font-size'] if vertical else cy + dim / 2
                start = cx + (0.7 if vertical else 0.12) * dim
                label.children[0].args['xlink:href'].args['d'] = f'M{start},{base_y} L{start + 2 * dim},{base_y}'
    if vertical:
        # Free-standing text (repeat counts, reducing-end labels, floating-bit counts) turns upright about its middle
        for i, el in enumerate(d.children):
            if isinstance(el, draw.Text) and el.args.get('x') is not None and (box := _drawn_extent(el, [])):
                bx0, by0, bx1, by1 = box[0]
                d.children[i] = draw.Group([el], transform = f'rotate(-90 {(bx0 + bx1) / 2} {(by0 + by1) / 2})')
                if el.args.get('text-anchor') == 'start':
                    # Repeat counts hang below the chain, which is its left once turned, so they grow away from it
                    el.args['x'] -= max(0, (bx1 - bx0) - (by1 - by0)) / 2
    # Labels are placed by local rules, blind to their neighbors. One that runs into a symbol, a bond or another label
    # tries a smaller size, and then other spots (sliding along its bond, or beside its symbol), keeping the first that
    # is clear; every label that does not collide stays exactly where it was. Geometry is in page orientation
    from glycorender.render import pdfmetrics, font_to_use
    turn = (lambda x, y: (-y, x)) if vertical else (lambda x, y: (x, y))
    tol, edge, ring = 0.02 * dim, np.linspace(-0.5, 0.5, 51) * dim, np.linspace(0, 2 * np.pi, 160)
    # Symbols are sampled across their inside and densely along their outline, a square or a circle
    square = np.vstack([np.stack(np.meshgrid(edge[::5], edge[::5]), -1).reshape(-1, 2)] + [
        np.column_stack([edge, np.full(51, k)])[:, ::i] for k in (-dim / 2, dim / 2) for i in (1, -1)])
    circle = np.vstack(
        [square[np.hypot(*square.T) < dim / 2], np.column_stack([np.cos(ring), np.sin(ring)]) * dim / 2])
    pts, owner, pad, labels = [], [], [], []
    stack = [(d, None)]
    while stack:
        el, sym = stack.pop()
        a = getattr(el, 'args', {}) or {}
        sym = el if a.get('class') == 'snfg-symbol' else sym
        if not isinstance(el, draw.Text):
            stack.extend((k, sym) for k in getattr(el, 'children', []) or [])
        if a.get('class') == 'snfg-symbol':
            pts.append((circle if isinstance(el.children[0], draw.Circle) else square) + turn(*el.center))
        elif a.get('class') == 'snfg-linkage':
            ends_xy = np.array([turn(*k) for k in np.reshape([float(k) for k in _SVG_NUMBER.findall(
                re.sub(r'[A-DF-Za-df-z]', ' ', a['d']))], (-1, 2))[[0, -1]]])
            k = np.linspace(0, 1, max(2, int(np.hypot(*(ends_xy[1] - ends_xy[0])) / tol / 2)))[:, None]
            pts.append(ends_xy[0] + k * (ends_xy[1] - ends_xy[0]))
        elif isinstance(el, draw.Text) and a.get('x') is not None and (box := _drawn_extent(el, [])):
            # Free-standing text (repeat counts, floating-bit counts) is upright on the page and never moves
            (bx0, by0, bx1, by1), grid = box[0], np.stack(np.meshgrid(np.linspace(-0.5, 0.5, 21),
                                                                      np.linspace(-0.5, 0.5, 5)),
                                                          -1).reshape(-1, 2)
            pts.append(grid * (bx1 - bx0, by1 - by0) + turn((bx0 + bx1) / 2, (by0 + by1) / 2))
        elif isinstance(el, draw.Text) and el.children and el.children[0].children:
            dy = el.children[0].children[0].args.get('dy')
            text = html.unescape(
                ''.join(str(k.escaped_text or '') for k in el.children[0].children)).replace(' ', '')
            if (dy == '-0.5em' or dy == '-3.15em' and sym is not None) and text:
                font = pdfmetrics.fonts[font_to_use + ('-Bold' if dy == '-3.15em' else '')]
                ink = [[k / font.ttf.units_per_em for _, args in font.ttf.outline(font.ttf.gid(ch)) for k in
                        args] or [
                           0, 0] for ch in text]
                ys = [k for g in ink for k in g[1::2]]
                # At unit size: advance with glycorender's 0.05 em tracking, the side bearings of the first and last
                # glyph, and the ink band about the baseline
                labels.append(
                    [el, sym, dy == '-3.15em', text, pdfmetrics.stringWidth(text, font.name, 1) + 0.05 * (
                            len(text) - 1), min(ink[0][0::2]),
                     font.ttf.width(font.ttf.gid(text[-1])) / 1000 - max(
                         ink[-1][0::2]), min(ys), max(ys), font.ttf.cap_height / font.ttf.units_per_em])
            continue
        else:
            continue
        owner.append(np.full(len(pts[-1]), id(el), dtype = np.int64))
        pad.append(np.full(len(pts[-1]), float(a.get('stroke-width', 0)) / 2 if 'd' in a else 0.0))

    def rect(label, carrier, frac, anchor, size):
        # Ink box of a label on a straight carrier: origin, direction along it, normal towards the glyph tops, extents
        el, sym, bold, text, width, lsb, rsb, low, high, cap = label[:10]
        c = sym.center if sym is not None else None
        place = (lambda x, y: (x - c[0] + turn(*c)[0], y - c[1] + turn(*c)[1])) if c else turn
        (x0, y0), (x1, y1) = [place(*k) for k in carrier]
        t = np.array([x1 - x0, y1 - y0]) / np.hypot(x1 - x0, y1 - y0)
        u0, lift = {'middle': -width * size / 2, 'end': -width * size}.get(anchor, 0), (
                                                                                           3.15 if bold else 0.5) * size
        # A label that would read right-to-left is turned upright by glycorender, same side but hanging from its cap
        low, high = (cap - high, cap - low) if t[0] < -1e-6 else (low, high)
        return (np.array([x0 + frac * (x1 - x0), y0 + frac * (y1 - y0)]), t, np.array([t[1], -t[0]]),
                u0 + lsb * size,
                u0 + (width - rsb) * size, lift + low * size, lift + high * size)

    def outline(box):
        # A placed label is an obstacle too, sampled along its edges and across its inside
        o, t, nrm, u0, u1, v0, v1 = box
        u, v = np.linspace(u0, u1, max(2, int((u1 - u0) / tol / 2))), np.linspace(v0, v1, 5)
        uv = np.column_stack([np.concatenate([np.tile(u, 5), np.repeat([u0, u1], 5)]),
                              np.concatenate([np.repeat(v, len(u)), np.tile(v, 2)])])
        return o + uv[:, :1] * t + uv[:, 1:] * nrm

    def clear(box, own, slack):
        # Points are kept sorted by x, so only the slice level with the label is tested; overlap within slack is a touch
        o, t, nrm, u0, u1, v0, v1 = box
        xs = o[0] + np.outer([u0, u1, u1, u0], t)[:, 0] + np.outer([v0, v0, v1, v1], nrm)[:, 0]
        lo, hi = np.searchsorted(pts[:, 0], [xs.min() - 0.1 * dim, xs.max() + 0.1 * dim])
        u, v, m = (pts[lo:hi] - o) @ t, (pts[lo:hi] - o) @ nrm, pad[lo:hi] - slack
        return not np.any(
            (u >= u0 - m) & (u <= u1 + m) & (v >= v0 - m) & (v <= v1 + m) & (owner[lo:hi] != own))

    for label in labels:
        carrier = label[0].children[0]
        offset = str(carrier.args.get('startOffset', '0'))
        label.append([np.reshape([float(k) for k in _SVG_NUMBER.findall(re.sub(
            r'[A-DF-Za-df-z]', ' ', carrier.args['xlink:href'].args['d']))], (-1, 2))[[0, -1]].tolist(),
                      float(offset[:-1]) / 100 if offset.endswith('%') else 0.0,
                      label[0].args.get('text-anchor', 'middle'), float(label[0].args['font-size'])])
        pts.append(outline(rect(label, *label[10])))
        owner.append(np.full(len(pts[-1]), id(label[0]), dtype = np.int64))
        pad.append(np.zeros(len(pts[-1])))
    pts, owner, pad = np.vstack(pts), np.concatenate(owner), np.concatenate(pad)
    order = np.argsort(pts[:, 0])
    pts, owner, pad = pts[order], owner[order], pad[order]
    for label in sorted(labels, key = lambda k: k[2]):
        el, sym, bold, text, width, spots = label[0], label[1], label[2], label[3], label[4], label[10]
        if clear(rect(label, *spots), id(el), tol):
            continue
        size = spots[3]
        if bold:
            cx, cy = sym.center
            # Beside its symbol, shrinking in place first: above, right, below, left (right first in vertical mode)
            sides = {'above': ([(cx - dim, cy), (cx + dim, cy)], 0.5, 'middle', lambda s: -0.6025 * dim),
                     'right': ([(cx + 0.62 * dim, cy), (cx + 2.62 * dim, cy)], 0, 'start',
                               lambda s: 0.35 * s),
                     'below': ([(cx - dim, cy), (cx + dim, cy)], 0.5, 'middle',
                               lambda s: 0.6 * dim + 0.78 * s),
                     'left': ([(cx - 2.62 * dim, cy), (cx - 0.62 * dim, cy)], 1, 'end', lambda s: 0.35 * s)}
            pref = ['right', 'above', 'below', 'left'] if vertical else ['above', 'right', 'below', 'left']
            tries = [
                ([(x, y + sides[k][3](s) + 3.15 * s) for x, y in sides[k][0]], sides[k][1], sides[k][2], s)
                for
                k, s in [(pref[0], 0.8 * size), (pref[0], 0.65 * size)] + [
                    (k, s) for k in pref[1:] for s in (size, 0.8 * size, 0.65 * size)]]
        else:
            length = np.hypot(*np.subtract(spots[0][1], spots[0][0]))
            # Slides along its bond, nearest spots first; the symbols at either end are obstacles like any other
            tries = [(spots[0], k / length, 'middle', s) for s in
                     (size, 0.8 * size, max(0.22 * dim, 0.65 * size))
                     for k in sorted(np.arange(width * s / 2, length - width * s / 2, tol),
                                     key = lambda k: abs(k / length - spots[1]))]
        for spot in tries:
            # A new spot has to keep a margin, so that the label does not just trade one touch for another
            if not clear(box := rect(label, *spot), id(el), tol / 2):
                continue
            keep, new = owner != id(el), outline(box)
            pts, owner = np.vstack([pts[keep], new]), np.concatenate(
                [owner[keep], np.full(len(new), id(el))])
            pad = np.concatenate([pad[keep], np.zeros(len(new))])
            order = np.argsort(pts[:, 0])
            pts, owner, pad = pts[order], owner[order], pad[order]
            el.args['font-size'], el.args['text-anchor'] = spot[3], spot[2]
            el.children[0].args['startOffset'] = f'{spot[1] * 100:.4f}%'
            if bold:
                (x0, y0), (x1, y1) = spot[0]
                el.children[0].args['xlink:href'].args['d'] = f'M{x0},{y0} L{x1},{y1}'
            break
    # Canvas: crop to what was actually drawn, since a formula over sugar positions cannot know how far labels, brackets and highlight halos reach
    boxes = _drawn_extent(d, [])
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    if vertical:
        # Rotating about the content center keeps the crop a plain transpose of the box, instead of forcing the square canvas a canvas-centred rotation would need
        c_x, c_y = (x0 + x1) / 2, (y0 + y1) / 2
        d.args['transform'] = f'rotate(90 {c_x} {c_y})'
        x0, y0, x1, y1 = c_x - (y1 - y0) / 2, c_y - (x1 - x0) / 2, c_x + (y1 - y0) / 2, c_y + (x1 - x0) / 2
    margin = dim * (0.45 if sticker else 0.35 if shadow else 0.2)
    # Namespace the element IDs per drawing, so that several GlycoDraw SVGs inlined into one HTML document do not resolve each other's <use> references
    tag = hashlib.blake2s(repr(id_key).encode(), digest_size = 4).hexdigest()
    d2 = draw.Drawing(x1 - x0 + 2 * margin, y1 - y0 + 2 * margin, origin = (x0 - margin, y0 - margin),
                      id_prefix = f'g{tag}_')
    d2.append(d)
    drawing = GlycanDrawing(d2, shadow = shadow, sticker = sticker, vertical = vertical, alt_text = alt_text)
    if filepath:
        drawing.save(str(filepath).replace(in_glycan, re.sub(r'[<>:"/\\|?*]', '_', in_glycan)))
    return drawing if is_jupyter() or suppress or filepath else display_svg_with_matplotlib(
        d2, shadow = shadow, sticker = sticker)


@rescue_glycans
def GlycoDraw(
        glycan: str | list[str] | dict[str, int], # IUPAC-condensed glycan sequence or composition (e.g., H5N4F1A2, Hex5HexNAc4Fuc1Neu5Ac2, {'Hex': 5}); a list is drawn as a grid
        vertical: bool = False, # Draw vertically
        compact: bool = False, # Use compact style
        show_linkage: bool = True, # Show linkage labels
        dim: float = 50, # Base dimension for scaling
        highlight_motif: str | None = None, # Motif to highlight
        highlight_termini_list: list = [], # Terminal positions (from 'terminal', 'internal', and 'flexible')
        highlight_linkages: list[int] | None = None, # Which linkages to highlight in a different color; indices, starting from 0, in glycan
        reverse_highlight: bool = False, # Whether to highlight everything EXCEPT highlight_motif
        repeat: bool | int | str | None = None, # Repeat unit specification (True: n units, int: # of units, str: range of units)
        repeat_range: list[int] | None = None, # Repeat unit range
        draw_method: str | None = None, # Drawing method: None, 'chem2d', 'chem3d'
        filepath: str | Path | None = None, # Output file path
        suppress: bool = False, # Suppress display
        per_residue: list = [], # Per-residue intensity values (order should be the same as the monosaccharides in glycan string)
        pdb_file: str | Path | None = None,  # only used when draw_method='chem3d'; already existing glycan structure
        alt_text: str | None = None,  # Custom ALT text for accessibility
        libr: dict | None = None,  # Can be modified for drawing too exotic monosaccharides
        reducing_end_label: str | None = None,  # Label to be drawn connected to the reducing end
        restrict_vocab: bool = False,  # Whether only tokens present in libr can be drawn
        shadow: bool = False,  # Draw a soft drop shadow under the monosaccharide symbols
        sticker: bool = False,  # Cut the whole structure out as a die-cut sticker: flat border hugging the outline, with a drop shadow
        highlight_residues: list[int] | None = None,  # Residues to highlight (indices, starting from 0, of the monosaccharides in glycan, as per_residue); the others, and linkages not between two highlighted residues or in highlight_linkages, are faded as outside highlight_motif
) -> Any:  # Drawing object
    "Renders glycan structure using SNFG symbols or chemical structure representation"
    if isinstance(glycan, dict):
        # A composition dict, e.g. {'Hex': 5, 'HexNAc': 4}, as returned by glycan_to_composition
        glycan = canonicalize_composition(glycan, as_string = True)
    elif not isinstance(glycan, str):
        # Several glycans (a list or a pandas column) are drawn side by side at one shared scale
        return plot_glycans_grid(list(glycan), filepath = filepath, suppress = suppress, vertical = vertical,
                                 compact = compact, show_linkage = show_linkage, dim = dim,
                                 highlight_motif = highlight_motif, highlight_termini_list = highlight_termini_list,
                                 reverse_highlight = reverse_highlight, libr = libr, restrict_vocab = restrict_vocab,
                                 shadow = shadow, sticker = sticker)
    if any(k in glycan for k in (';', 'β', 'α', 'RES', '=')):
        raise Exception
    in_glycan = glycan  # motif names, repeat units, and trailing linkages all rewrite glycan below, while a caller-built filename still carries what was passed in
    if libr is None:
        libr = lib
    if glycan.lower().startswith('terminal') and resolve_motif_name(glycan) is None:
        glycan = glycan.split('_')[-1]
    motif_hit = resolve_motif_name(glycan)
    if motif_hit:
        if motif_hit[0].startswith('r'):
            raise ValueError(f"'{in_glycan}' is the glyco-regex {motif_hit[0][1:]}, matching a family of structures "
                             f"rather than one to draw; draw a glycan with highlight_motif = '{in_glycan}' instead.")
        glycan = motif_hit[0]
    elif is_composition(glycan) and not sugar_dict.keys().isdisjoint(comp := canonicalize_composition(glycan)):
        # Compositions have no topology to lay out, so each monosaccharide gets its symbol followed by its count, and substituents like S or P are spelled out
        d, cursor, row = draw.Group(), 0.0, 0.0
        comp = dict(sorted(comp.items(), key = lambda x: (_COMP_ORDER.get(x[0], len(_COMP_ORDER)), x[0])))
        for mono, count in comp.items():
            if mono in sugar_dict:
                shape, color, furanose = sugar_dict[mono]
                draw_shape(shape, color, x_pos = -(cursor + 0.5), y_pos = row, col_dict = col_dict_base, drawing = d, furanose = furanose, dim = dim)
                cursor += 1.2
            else:
                # Centered in the room it takes, like a symbol, so a long name like Sulfate clears the previous count
                step = max(1.2, 0.31 * len(mono) + 0.2)
                d.append(draw.Text(mono, dim * 0.5, (cursor + (step - 0.2) / 2) * dim, (row + 0.18) * dim,
                                   text_anchor = 'middle', fill = col_dict_base['black']))
                cursor += step
            d.append(draw.Text(str(count), dim * 0.5, cursor * dim, (row + 0.18) * dim, text_anchor = 'start', fill = col_dict_base['black']))
            # Vertical compositions are stacked row by row instead of rotated, so the counts stay upright and readable
            cursor, row = (0.0, row + 1.25) if vertical else (cursor + 0.28 * len(str(count)) + 0.45, row)
        if alt_text is None:
            alt_text = f"SNFG composition diagram of {glycan}: " + ", ".join(f"{count} {mono}" for mono, count in comp.items()) + "."
        return _finish_drawing(d, in_glycan, alt_text, (in_glycan, compact, vertical, dim), dim = dim, filepath = filepath, suppress = suppress, shadow = shadow, sticker = sticker)
    # Values often arrive as an array or Series (attributions, a dataframe column), whose truth value is ambiguous
    per_residue = list(per_residue)
    highlight_linkages = [] if highlight_linkages is None else list(highlight_linkages)
    highlight_residues = None if highlight_residues is None else {int(k) for k in highlight_residues}
    if repeat and not repeat_range:
        _backbone = re.findall(r'.*\((?!.*\()', glycan)[0]
        _conn = re.sub(r'\)(.*)', '', re.sub(r'.*\((?!.*\()', '', glycan))
        glycan = f'blank(?1-{_conn[-1]}){_backbone}{_conn[:2]}-?)'
        if per_residue:
            per_residue = [0] + per_residue
        if highlight_linkages:
            highlight_linkages = [k + 1 for k in highlight_linkages]
    if glycan.endswith(')'):
        glycan += 'blank'
        if per_residue:
            per_residue = per_residue + [0]
    cut = glycan.rfind('}') + 1 if '^' in glycan else 0
    draw_this = glycan[:cut] + (
        graph_to_string(glycan_to_nxGraph(glycan[cut:]), order_by = "linkage") if not glycan[cut:].startswith(
            '[') else glycan[cut:])
    if per_residue:
        per_residue_by_node = process_per_residue(draw_this, per_residue, glycan)
    if highlight_linkages:
        per_linkage_by_node = process_per_linkage(draw_this, highlight_linkages, glycan)
    if highlight_residues is not None:
        # Mapped like per_residue, with the leading repeat placeholder and the trailing blank as residues that are never highlighted
        shift = 1 if repeat and not repeat_range else 0
        shown_by_node = process_per_residue(draw_this, [k - shift in highlight_residues for k in range((len(glycan_to_nxGraph(glycan)) + 1) // 2)], glycan)
    if compact:
        show_linkage = False
    if isinstance(highlight_motif, str):
        highlight_hit = resolve_motif_name(highlight_motif)
        if highlight_hit:
            highlight_motif = highlight_hit[0]
            if not highlight_motif.startswith('r') and not highlight_termini_list:
                highlight_termini_list = highlight_hit[1]
        if highlight_motif and highlight_motif.startswith('r'):
            temp = get_match(highlight_motif[1:], draw_this)
            if not temp:
                from glycowork.motif.regex import explain_match
                ex = explain_match(highlight_motif[1:], draw_this)
                warnings.warn(
                    f"'{highlight_motif[1:]}' does not match {draw_this}; chunks without a hit: {ex.loc[ex.hits_in_glycan == 0, 'chunk'].tolist()}")
            highlight_motif, highlight_termini_list = (temp[0], []) if temp else (None, highlight_termini_list)
    # toggle SNFG vs 2D/3D chem
    if draw_method:
        if draw_method == 'chem2d':
            return draw_chem2d(draw_this = draw_this, mono_list = mono_list, filepath = filepath)
        elif draw_method == 'chem3d':
            return draw_chem3d(draw_this = draw_this, mono_list = mono_list, filepath = filepath, pdb_file = pdb_file)
        else:
            raise ValueError('Method not supported. Please choose between "chem2d" and "chem3d".')
    # Handle floaty bits if present
    floaty_bits, anchored_bits, node_shift = [], [], 0
    for openpos, closepos, _ in get_matching_indices(draw_this, opendelim = '{', closedelim = '}'):
        bit = draw_this[openpos:closepos]
        if '^' in bit:
            fragment, bit_anchors = parse_floating_bit(bit)
            anchored_bits.append((f"{fragment}blank", bit_anchors, node_shift))
        else:
            fragment = bit
            floaty_bits.append((f"{bit}blank", node_shift))  # with the node its per-residue and per-linkage values start at
        # The values were indexed against the string that still carried these bits, so the nodes they contributed have to be added back when reading them; an anchored bit contributes its merged fragment, not every alternative
        node_shift += 2 * fragment.count('(')
        draw_this = draw_this[:openpos-1] + len(draw_this[openpos-1:closepos+1])*'*' + draw_this[closepos+1:]
    draw_this = draw_this.replace('*', '')
    if anchored_bits:  # An anchor matching nothing must not silently delete its residue from the drawing
        anchor_graph = glycan_to_nxGraph(draw_this)
        placeable = [any(resolve_anchor(anchor_graph, anchor) for anchor in bit_anchors.values()) for _, bit_anchors, _ in
                     anchored_bits]
        floaty_bits += [(bit, off) for (bit, _, off), ok in zip(anchored_bits, placeable) if not ok]
        anchored_bits = [entry for entry, ok in zip(anchored_bits, placeable) if ok]
    if restrict_vocab and not _drawable(draw_this, libr):
        if "!" in draw_this:
            draw_this = re.sub(r'\[!.*?\)\]|!.*?\)', '', draw_this)
        else:
            raise Exception('Did you enter a real glycan or motif?')
    data = get_coordinates_and_labels(draw_this, highlight_motif = highlight_motif, termini_list = highlight_termini_list, reverse_highlight  = reverse_highlight)
    main_sugar, main_sugar_x_pos, main_sugar_y_pos, main_sugar_modification, main_bond, main_conf, main_sugar_label, main_bond_label = data[0]
    # Branch levels are ordered by graph traversal but per-residue/per-linkage values arrive in sequence order, so they are placed by node rather than by position in a level
    node_positions = data[-1]
    node_at = {v: k + node_shift for k, v in node_positions.items()}
    lv_sugar, lv_x_pos, lv_y_pos, lv_sugar_modification, lv_bond, lv_connection, lv_conf, lv_sugar_label, lv_bond_label = map(
        list, zip(*data[1:-1]))
    if highlight_residues is not None:
        tree, shown = glycan_to_nxGraph(draw_this), {node for node, flag in shown_by_node.items() if flag}

        def bond_label(child):
            # A drawn residue's linkage stays visible when both of its residues are highlighted, or when it is one of highlight_linkages
            parent = next(iter(tree.pred[next(iter(tree.pred[child - node_shift]))])) + node_shift
            return 'show' if (child in shown and parent in shown) or (highlight_linkages and per_linkage_by_node.get(child + 1, False)) else 'hide'

        main_sugar_label = ['show' if node_at[(0, 0, k)] in shown else 'hide' for k in range(len(main_sugar))]
        main_bond_label = [bond_label(node_at[(0, 0, k + 1)]) for k in range(len(main_sugar) - 1)]
        lv_sugar_label = [[['show' if node_at[(lvl + 1, b, s)] in shown else 'hide' for s in range(len(sugars))] for b, sugars in enumerate(level)]
                          for lvl, level in enumerate(lv_sugar)]
        lv_bond_label = [[[bond_label(node_at[(lvl + 1, b, s)]) for s in range(len(sugars))] for b, sugars in enumerate(level)] for lvl, level in
                         enumerate(lv_sugar)]
    if not show_linkage:
        main_bond = ['-'] * len(main_bond)
        lv_bond = [[['-' for _ in y] for y in level] for level in lv_bond]

    # Calculate angles for main chain Y, Z fragments
    def calculate_degree(y1, y2, x1, x2):
        return degrees(atan((y1-y2) / (2*(x2-x1))))

    main_deg = [calculate_degree(main_sugar_y_pos[k], main_sugar_y_pos[k - 1], main_sugar_x_pos[k], main_sugar_x_pos[k - 1])
        if sugar in {'Z', 'Y'} and k > 0 else 0 for k, sugar in enumerate(main_sugar)]
    # Calculate angles for branch Y, Z fragments, at every branch level
    lv_deg = []
    for lvl, level_sugar in enumerate(lv_sugar):
        parent_x = [main_sugar_x_pos] if not lvl else lv_x_pos[lvl - 1]
        parent_y = [main_sugar_y_pos] if not lvl else lv_y_pos[lvl - 1]
        lv_deg.append([[
            calculate_degree(lv_y_pos[lvl][k][j],
                             parent_y[0 if not lvl else lv_connection[lvl][k][0]][lv_connection[lvl][k][1]],
                             lv_x_pos[lvl][k][j],
                             parent_x[0 if not lvl else lv_connection[lvl][k][0]][lv_connection[lvl][k][1]])
            if sugar in {'Z', 'Y'} and len(sugars) == 1 else
            calculate_degree(lv_y_pos[lvl][k][j], lv_y_pos[lvl][k][j - 1], lv_x_pos[lvl][k][j], lv_x_pos[lvl][k][j - 1])
            if sugar in {'Z', 'Y'} else 0 for j, sugar in enumerate(sugars)
        ] for k, sugars in enumerate(level_sugar)])
    # Adjust drawing dimensions
    all_y = unwrap(unwrap(lv_y_pos)) + main_sugar_y_pos
    all_x = unwrap(unwrap(lv_x_pos)) + main_sugar_x_pos
    max_y, min_y = max(all_y), min(all_y)
    max_x = max(all_x)
    y_span = max_y - min_y
    # Floaty bits are spread over the full height of their own lane, so they need vertical room of their own
    if len(floaty_bits) + len(anchored_bits) > y_span:
        y_span += 1.0
        max_y += 0.5
        min_y -= 0.5
    # Generate default ALT text if not provided
    if alt_text is None:
        orientation = "vertical" if vertical else "horizontal"
        style = "compact" if compact else "standard"
        linkage_info = "with" if show_linkage else "without"
        alt_text = f"SNFG diagram of {in_glycan} drawn in {orientation} {style} style {linkage_info} linkage labels."
        if highlight_motif:
            alt_text += f" The motif {highlight_motif} is highlighted."
        if repeat:
            alt_text += f" Contains repeat unit{f' (n={repeat})' if repeat is not True else ''}."
    # Draw
    d = draw.Group()
    if reducing_end_label:
        bond_start_x = main_sugar_x_pos[0] - 0.5
        label_x = main_sugar_x_pos[0] - 0.55 - (len(reducing_end_label) * 0.1)
        label_y = main_sugar_y_pos[0]
        add_bond(bond_start_x, main_sugar_x_pos[0], label_y, main_sugar_y_pos[0], d, label = '-', dim = dim, compact = compact, highlight = main_sugar_label[0])
        col_dict = col_dict_transparent if main_sugar_label[0] == 'hide' else col_dict_base
        x_base = -label_x * dim * (1.2 if compact else 2)
        y_base = label_y * dim * (0.6 if compact else 1) + 5
        d.append(draw.Text(reducing_end_label, dim * 0.35, x_base, y_base, text_anchor = 'end', fill = col_dict['black'], dominant_baseline = 'middle'))
    # Bond main chain
    [add_bond(main_sugar_x_pos[k+1], main_sugar_x_pos[k], main_sugar_y_pos[k + 1], main_sugar_y_pos[k], d, label = main_bond[k], dim = dim, compact = compact, highlight = main_bond_label[k], color_highlight = per_linkage_by_node.get(node_at[(0, 0, k + 1)] + 1, False) if highlight_linkages else False) for k in range(len(main_sugar) - 1)]
    # Bond within each branch, at every branch level; level 1 also connects to the main chain
    for lvl in range(len(lv_sugar)):
        [add_bond(lv_x_pos[lvl][b_idx][s_idx + 1], lv_x_pos[lvl][b_idx][s_idx], lv_y_pos[lvl][b_idx][s_idx + 1],
                  lv_y_pos[lvl][b_idx][s_idx], d, label = lv_bond[lvl][b_idx][s_idx + 1], dim = dim, compact = compact,
                  highlight = lv_bond_label[lvl][b_idx][s_idx + 1],
                  color_highlight = per_linkage_by_node.get(node_at[(lvl + 1, b_idx, s_idx + 1)] + 1,
                                                            False) if highlight_linkages else False) for b_idx in
         range(len(lv_sugar[lvl])) for s_idx in range(len(lv_sugar[lvl][b_idx]) - 1)]
        if not lvl:
            [add_bond(lv_x_pos[0][k][0], main_sugar_x_pos[lv_connection[0][k][1]], lv_y_pos[0][k][0],
                      main_sugar_y_pos[lv_connection[0][k][1]], d, label = lv_bond[0][k][0], dim = dim,
                      compact = compact, highlight = lv_bond_label[0][k][0],
                      color_highlight = per_linkage_by_node.get(node_at[(1, k, 0)] + 1,
                                                                False) if highlight_linkages else False) for
             k in range(len(lv_sugar[0]))]
    # Bond each deeper branch to the branch it sits on
    for lvl in range(1, len(lv_sugar)):
        [add_bond(lv_x_pos[lvl][k][0], lv_x_pos[lvl - 1][lv_connection[lvl][k][0]][lv_connection[lvl][k][1]],
                  lv_y_pos[lvl][k][0], lv_y_pos[lvl - 1][lv_connection[lvl][k][0]][lv_connection[lvl][k][1]], d,
                  label = lv_bond[lvl][k][0], dim = dim, compact = compact, highlight = lv_bond_label[lvl][k][0],
                  color_highlight = per_linkage_by_node.get(node_at[(lvl + 1, k, 0)] + 1,
                                                            False) if highlight_linkages else False) for k in
         range(len(lv_sugar[lvl]))]
    # Sugar main chain
    [add_sugar(main_sugar[k], d, x_pos = main_sugar_x_pos[k], y_pos = main_sugar_y_pos[k],
               modification = main_sugar_modification[k], conf = main_conf[k], compact = compact, dim = dim,
               deg = main_deg[k], highlight = main_sugar_label[k],
               scalar = per_residue_by_node.get(node_at[(0, 0, k)], 0) if per_residue else 0)
     for k in range(len(main_sugar))]
    # Sugar of every branch level
    for lvl in range(len(lv_sugar)):
        [add_sugar(lv_sugar[lvl][b_idx][s_idx], d, x_pos = lv_x_pos[lvl][b_idx][s_idx],
                   y_pos = lv_y_pos[lvl][b_idx][s_idx], modification = lv_sugar_modification[lvl][b_idx][s_idx],
                   conf = lv_conf[lvl][b_idx][s_idx], compact = compact, dim = dim, deg = lv_deg[lvl][b_idx][s_idx],
                   highlight = lv_sugar_label[lvl][b_idx][s_idx],
                   scalar = per_residue_by_node.get(node_at[(lvl + 1, b_idx, s_idx)], 0) if per_residue else 0) for
         b_idx in range(len(lv_sugar[lvl])) for s_idx in range(len(lv_sugar[lvl][b_idx]))]
    # Floating bits, brackets, and repeat labels are never part of the motif, so reverse_highlight shows them
    highlight = 'show' if highlight_motif is None or reverse_highlight else 'hide'
    if floaty_bits != []:
        # Identical bits are drawn once with a count, unless their per-residue values or linkage highlights tell them apart
        fb_keys = [(bit, tuple((per_residue_by_node.get(off + n, 0) if per_residue else 0, per_linkage_by_node.get(off + n + 1, False) if highlight_linkages else False)
                               for n in range(0, 2 * bit.count('('), 2))) for bit, off in floaty_bits]
        fb_count, fb_offset = {i: fb_keys.count(i) for i in fb_keys}, {}
        for key, (_, off) in zip(fb_keys, floaty_bits):
            fb_offset.setdefault(key, off)
        fb_keys = list(fb_offset)
        floaty_bits = [key[0] for key in fb_keys]
        floaty_data = []
        for k, k_val in enumerate(floaty_bits):
            if in_lib(min_process_glycans([k_val])[0][0], libr):
                floaty_data.append(get_coordinates_and_labels(k_val, highlight_motif = None))
            else:
                floaty_data.append(get_coordinates_and_labels('blank(-)blank', highlight_motif = None))
        n_floats = len(floaty_bits)
        y_spacing = (y_span / (n_floats - 1)) if n_floats > 1 else 0
        # How far each bit reaches above and below its root row, so that bits with a tucked Fuc or a side branch are
        # stacked clear of each other
        reach = [(min(ys) - j_val[0][2][0], max(ys) - j_val[0][2][0]) for j_val in floaty_data for ys in
                 [j_val[0][2] + unwrap(unwrap([lv[2] for lv in j_val[1:-1]]))]]
        for j, j_val in enumerate(floaty_data):
            floaty_sugar, floaty_sugar_x_pos, floaty_sugar_y_pos, floaty_sugar_modification, floaty_bond, floaty_conf, _, _ = j_val[0]
            # Nodes of the bit as drawn on its own, moved to where the bit sits in the sequence; its trailing blank stands in for the acceptor and carries no value
            off, n_bit = fb_offset[fb_keys[j]], 2 * floaty_bits[j].count('(')
            at = {v: off + n if n < n_bit else None for n, v in j_val[-1].items()}
            scalar = lambda pos: per_residue_by_node.get(at[pos], 0) if per_residue else 0
            linked = lambda pos: per_linkage_by_node.get(at[pos] + 1, False) if highlight_linkages and at[pos] is not None else False
            floaty_sugar_label = [highlight] * len(floaty_sugar)
            floaty_bond_label = [highlight] * len(floaty_bond)
            floaty_sugar_x_pos = [k + max_x + 1 for k in floaty_sugar_x_pos]
            current_y = (min_y + j * y_spacing + sum(r[1] - r[0] for r in reach[:j]) - reach[j][0]) if (
                    n_floats > 1) else (min_y + max_y - reach[j][0] - reach[j][1]) / 2
            # The bit keeps its own layout, moved onto its row; flattened, a floating H antigen or Lewis X stacked its
            # Fuc onto the Gal or GlcNAc, and side branches were lost
            shift = current_y - floaty_sugar_y_pos[0]
            floaty_sugar_y_pos = [k + shift for k in floaty_sugar_y_pos]
            lanes = [([floaty_sugar_x_pos], [floaty_sugar_y_pos])] + [([[k + max_x + 1 for k in xs] for xs in lv[1]],
                                                                        [[k + shift for k in ys] for ys in lv[2]])
                                                                       for lv in j_val[1:-1]]
            if floaty_sugar != ['blank', 'blank']:
                [add_bond(floaty_sugar_x_pos[k + 1], floaty_sugar_x_pos[k], floaty_sugar_y_pos[k + 1], floaty_sugar_y_pos[k], d, label = floaty_bond[k] if show_linkage else '-', dim = dim, compact = compact, highlight = floaty_bond_label[k], color_highlight = linked((0, 0, k + 1))) for k in range(len(floaty_sugar) - 1)]
                for lvl, lv in enumerate(j_val[1:-1]):
                    (xs, ys), (pxs, pys) = lanes[lvl + 1], lanes[lvl]
                    for b, (conn, bonds) in enumerate(zip(lv[5], lv[4])):
                        parent = conn[0] if lvl else 0
                        add_bond(xs[b][0], pxs[parent][conn[1]], ys[b][0], pys[parent][conn[1]], d,
                                 label = bonds[0] if show_linkage else '-', dim = dim, compact = compact,
                                 highlight = highlight, color_highlight = linked((lvl + 1, b, 0)))
                        [add_bond(xs[b][k + 1], xs[b][k], ys[b][k + 1], ys[b][k], d,
                                  label = bonds[k + 1] if show_linkage else '-', dim = dim, compact = compact,
                                  highlight = highlight, color_highlight = linked((lvl + 1, b, k + 1))) for k in range(len(bonds) - 1)]
                [add_sugar(floaty_sugar[k], d, x_pos = floaty_sugar_x_pos[k], y_pos = floaty_sugar_y_pos[k], modification = floaty_sugar_modification[k], conf = floaty_conf[k], compact = compact, dim = dim, highlight = floaty_sugar_label[k], scalar = scalar((0, 0, k))) for k in range(len(floaty_sugar))]
                for lvl, lv in enumerate(j_val[1:-1]):
                    [add_sugar(lv[0][b][k], d, x_pos = lanes[lvl + 1][0][b][k], y_pos = lanes[lvl + 1][1][b][k],
                               modification = lv[3][b][k], conf = lv[6][b][k], compact = compact, dim = dim,
                               highlight = highlight, scalar = scalar((lvl + 1, b, k))) for b in range(len(lv[0])) for k in range(len(lv[0][b]))]
            else:
                add_sugar('text', d, x_pos = min(floaty_sugar_x_pos) - 0.3, y_pos = floaty_sugar_y_pos[-1], modification = floaty_bits[j].replace('blank', ''), compact = compact, dim = dim, text_anchor = 'end', highlight = highlight)
            if fb_count[fb_keys[j]] > 1:
                x_offset = 0.5 if not compact else 0.75
                add_sugar('text', d, x_pos = max(floaty_sugar_x_pos) + x_offset, y_pos = current_y, modification = f"{fb_count[fb_keys[j]]}x", compact = compact, dim = dim, highlight = highlight)
        bracket_x = max_x * (2 if not compact else 1.2) + 1
        bracket_y = (min_y, max_y) if not compact else ((min_y * 0.5) * 1.2, (max_y * 0.5) * 1.2)
        draw_bracket(bracket_x, bracket_y, d, direction = 'right', dim = dim, highlight = highlight)
    if anchored_bits:
        # Dashed connectors point at symbols that are already on the canvas, so collect them separately and splice them in underneath, instead of letting them paint over the monosaccharides
        anchor_layer = draw.Group()
        lanes = [(main_sugar_x_pos, main_sugar_y_pos)] + list(zip(lv_x_pos, lv_y_pos))
        occupied = {(round(x), round(y)) for x, y in zip(main_sugar_x_pos, main_sugar_y_pos)}
        occupied |= {(round(x), round(y)) for xs, ys in
                     zip(unwrap(lv_x_pos), unwrap(lv_y_pos)) for x, y in zip(xs, ys)}
        for bit, bit_anchors, off in anchored_bits:
            a_data = get_coordinates_and_labels(bit, highlight_motif = None)
            a_sugar, a_x_pos, _, a_modification, a_bond, a_conf, _, _ = a_data[0]
            # Every ghost copy carries the values of the residues it stands for; index 0 is the trailing blank
            a_at = {v: off + n for n, v in a_data[-1].items()}
            a_scalar = [per_residue_by_node.get(a_at[(0, 0, k)], 0) if per_residue and k else 0 for k in range(len(a_sugar))]
            a_linked = [per_linkage_by_node.get(a_at[(0, 0, k)] + 1, False) if highlight_linkages and k else False for k in range(len(a_sugar))]
            for linkage, anchor in bit_anchors.items():
                for n in resolve_anchor(anchor_graph, anchor):
                    lane, b, i = node_positions[n]
                    x_pos, y_pos = lanes[lane]
                    target_x, target_y = (x_pos[i], y_pos[i]) if lane == 0 else (x_pos[b][i], y_pos[b][i])
                    # A ghost copy beside every candidate acceptor beats one distant copy with lines crossing the structure
                    ghost_y = next(
                        (target_y + offset for offset in ((-3, 3, -4, 4, -5, 5) if compact else (-2, 2, -3, 3, -4, 4))
                         if all(
                            (round(target_x + a_x_pos[k]), round(target_y + offset)) not in occupied for k in
                            range(1, len(a_sugar)))), target_y - (3 if compact else 2))
                    occupied.update((round(target_x + a_x_pos[k]), round(ghost_y)) for k in range(1, len(a_sugar)))
                    [add_bond(target_x + a_x_pos[k + 1], target_x + a_x_pos[k], ghost_y, ghost_y, d,
                              label = a_bond[k] if show_linkage else '-', dim = dim,
                              compact = compact, highlight = highlight, color_highlight = a_linked[k + 1]) for k in range(1, len(a_sugar) - 1)]
                    [add_sugar(a_sugar[k], d, x_pos = target_x + a_x_pos[k], y_pos = ghost_y, modification = a_modification[k],
                               conf = a_conf[k],
                               compact = compact, dim = dim, highlight = highlight, scalar = a_scalar[k]) for k in range(1, len(a_sugar))]
                    add_bond(target_x + a_x_pos[1], target_x, ghost_y, target_y, anchor_layer,
                             label = process_bonds([linkage])[0] if show_linkage else '-', dim = dim, compact = compact,
                             highlight = highlight, dashed = True, color_highlight = a_linked[1])
        d.children.insert(0, anchor_layer)
    # add brackets around repeating unit
    if repeat:
        # process annotation
        repeat_annot = 'n' + (' = ' + str(repeat) if isinstance(repeat, (str, int)) and repeat is not True else '')
        # repeat range code block
        if repeat_range:
            # Between the linkage label and the symbol, not through the label at the middle of the bond
            bracket_open = (main_sugar_x_pos[repeat_range[1]] * 2) + 0.62 if not compact else (main_sugar_x_pos[repeat_range[1]] * 1.2) + 0.6
            bracket_close = (main_sugar_x_pos[repeat_range[0]] * 2) - 0.62 if not compact else (main_sugar_x_pos[repeat_range[0]] * 1.2) - 0.6
            bracket_y_open =  (main_sugar_y_pos[repeat_range[1]], main_sugar_y_pos[repeat_range[1]]) if not compact else (((np.mean(main_sugar_y_pos[repeat_range[1]]) * 0.5) * 1.2), ((np.mean(main_sugar_y_pos[repeat_range[1]]) * 0.5) * 1.2))
            bracket_y_close = (main_sugar_y_pos[repeat_range[0]], main_sugar_y_pos[repeat_range[0]]) if not compact else (((np.mean(main_sugar_y_pos[repeat_range[0]]) * 0.5) * 1.2), ((np.mean(main_sugar_y_pos[repeat_range[0]]) * 0.5) * 1.2))
            text_x = main_sugar_x_pos[repeat_range[0]] - (0.31 if not compact else 0.5)
            text_y = main_sugar_y_pos[0] + 1.05 if not compact else (main_sugar_y_pos[0] + 1.03) / 0.6
            draw_bracket(bracket_close, bracket_y_close, d, direction = 'left', dim = dim, highlight = highlight, deg = 0)
            draw_bracket(bracket_open, bracket_y_open, d, direction = 'right', dim = dim, highlight = highlight, deg = 0)
            add_sugar('text', d, x_pos = text_x, y_pos = text_y, modification = repeat_annot, compact = compact, dim = dim, text_anchor = 'start', highlight = highlight)
        # repeat unit code block
        else:
            open_deg = calculate_degree(main_sugar_y_pos[-1], main_sugar_y_pos[-2], main_sugar_x_pos[-1], main_sugar_x_pos[-2])
            if open_deg == 0:
                bracket_open = np.mean([k * 2 for k in main_sugar_x_pos][-2:]) + 0.2 if not compact else np.mean([k * 1.2 for k in main_sugar_x_pos][-2:]) + 0.15
                bracket_y_open = (np.mean(main_sugar_y_pos[-2:]), np.mean(main_sugar_y_pos[-2:])) if not compact else (((np.mean(main_sugar_y_pos[-2:]) * 0.5) * 1.2), ((np.mean(main_sugar_y_pos[-2:]) * 0.5) * 1.2))
                bracket_y_close = (main_sugar_y_pos[0], main_sugar_y_pos[0]) if not compact else (((np.mean(main_sugar_y_pos[0]) * 0.5) * 1.2), ((np.mean(main_sugar_y_pos[0]) * 0.5) * 1.2))
            else:
                bracket_open = np.mean([k * 2 for k in main_sugar_x_pos][-2:]) if not compact else np.mean([k * 1.2 for k in main_sugar_x_pos][-2:])
                bracket_y_open = (np.mean(main_sugar_y_pos[-2:]), np.mean(main_sugar_y_pos[-2:])) if not compact else (((np.mean(main_sugar_y_pos[-2:]) * 0.5) * 1.2) + 0.3, ((np.mean(main_sugar_y_pos[-2:]) * 0.5) * 1.2) - 0.3)
                bracket_y_close = (main_sugar_y_pos[0], main_sugar_y_pos[0]) if not compact else (((np.mean(main_sugar_y_pos[0]) * 0.5) * 1.2) + 0.3, ((np.mean(main_sugar_y_pos[0]) * 0.5) * 1.2) - 0.3)
            bracket_close = np.mean([k * 2 for k in main_sugar_x_pos][:2]) - 0.2 if not compact else np.mean([k * 1.2 for k in main_sugar_x_pos][:2]) - 0.15
            text_x = bracket_close - 0.42 if not compact else bracket_close - 0.13
            text_y = main_sugar_y_pos[0] + 1.05 if not compact else (main_sugar_y_pos[0] + 1.03) / 0.6
            draw_bracket(bracket_open, bracket_y_open, d, direction = 'right', dim = dim, highlight = highlight, deg = open_deg)
            draw_bracket(bracket_close, bracket_y_close, d, direction = 'left', dim = dim, highlight = highlight, deg = 0)
            add_sugar('text', d, x_pos = text_x, y_pos = text_y, modification = repeat_annot, compact = compact, dim = dim, text_anchor = 'start', highlight = highlight)
    return _finish_drawing(d, in_glycan, alt_text, (in_glycan, highlight_motif, highlight_termini_list, compact, vertical, dim, per_residue, repeat, reducing_end_label, show_linkage, highlight_linkages, reverse_highlight, repeat_range, None if highlight_residues is None else sorted(highlight_residues)), vertical = vertical, dim = dim, filepath = filepath, suppress = suppress, shadow = shadow, sticker = sticker)


def _drawable(glycan: str, # Candidate label
              libr: dict | None = None # Vocabulary to check against
              ) -> bool: # Whether GlycoDraw could render this
    "Mirrors GlycoDraw's own restrict_vocab test, so annotate_figure never rejects a label GlycoDraw can draw"
    if libr is None:
        libr = lib
    return in_lib(glycan, expand_lib(libr, list(sugar_dict.keys())
                                     + [k for k in min_process_glycans([glycan])[0] if '/' in k]))


def plot_glycans_grid(
        glycans: str | list[str], # Glycans to draw (IUPAC-condensed, compositions, motif names)
        ncols: int = 4, # Glycans per row
        labels: list[str] | None = None, # Caption under each glycan, such as a name, ID, or fold change
        filepath: str | Path | None = None, # Output file path (.svg, .pdf, or .png)
        suppress: bool = False, # Suppress display
        **kwargs # Passed on to GlycoDraw for every glycan, e.g. compact, vertical, dim, show_linkage, highlight_motif
) -> Any: # Drawing object
    "Draws several glycans at one shared scale into a single captioned figure, row by row, for side-by-side comparison"
    glycans = [glycans] if isinstance(glycans, (str, dict)) else list(glycans)
    # A composition dict, as glycan_to_composition returns it, is captioned in the alt text by its string form
    glycans = [canonicalize_composition(g, as_string = True) if isinstance(g, dict) else g for g in glycans]
    # A filtered pandas Series keeps its index, so labels[i] would look up index labels instead of positions
    labels = None if labels is None else [labels] if isinstance(labels, str) else list(labels)
    if labels is not None and len(labels) != len(glycans):
        raise ValueError(f"labels has {len(labels)} entries but there are {len(glycans)} glycans to caption")
    dim = kwargs.get('dim', 50)
    drawings = [GlycoDraw(g, suppress = True, **kwargs) for g in glycans]
    boxes = [k.drawing_obj.view_box for k in drawings]
    # Every column is as wide as its widest glycan and every row as tall as its tallest, so all symbols keep one size
    # A gutter between cells, or a composition or a Fuc at a cell's edge reads as part of its neighbor
    cap, gap = dim * 0.55 if labels is not None else 0, dim * 0.8
    col_w = [max(b[2] for b in boxes[c::ncols]) + gap for c in range(min(ncols, len(boxes)))]
    row_h = [max(b[3] for b in boxes[r:r + ncols]) + cap + gap for r in range(0, len(boxes), ncols)]
    tag = hashlib.blake2s(repr((glycans, labels, ncols, sorted(kwargs.items(), key = str))).encode(),
                          digest_size = 4).hexdigest()
    grid = draw.Drawing(sum(col_w) - gap, sum(row_h) - gap, id_prefix = f'g{tag}_')
    for i, (k, (x0, y0, w, h)) in enumerate(zip(drawings, boxes)):
        r, c = divmod(i, ncols)
        left, top = sum(col_w[:c]), sum(row_h[:r])
        # Vertical glycans stand on their reducing end, so they share a baseline right above their captions
        shift = (left + (col_w[c] - gap - w) / 2 - x0,
                 top + (row_h[r] - gap - cap - h) / (1 if kwargs.get('vertical') else 2) - y0)
        grid.append(draw.Group(k.drawing_obj.elements, transform = f'translate({shift[0]} {shift[1]})'))
        if labels is not None:
            grid.append(draw.Text(str(labels[i]), dim * 0.35, left + (col_w[c] - gap) / 2,
                                  top + row_h[r] - gap - cap * 0.35,
                                  text_anchor = 'middle', fill = col_dict_base['black']))
    alt_text = html.escape(f"SNFG diagrams of {len(glycans)} glycans: " + "; ".join(glycans) + ".")
    drawing = GlycanDrawing(grid, shadow = kwargs.get('shadow', False), sticker = kwargs.get('sticker', False),
                            vertical = kwargs.get('vertical', False), alt_text = alt_text)
    if filepath:
        drawing.save(filepath)
    return drawing if is_jupyter() or suppress or filepath else display_svg_with_matplotlib(
        grid, shadow = drawing.shadow, sticker = drawing.sticker)


def _spread_glycans(placements: list, # (x, y, w, h, anchor_x, anchor_y) per glycan, (x, y) being its natural corner
                    canvas: tuple, # (width, height) of the figure
                    iterations: int = 300, # Relaxation passes
                    pad: float = 4.0 # Minimum gap to open between boxes
                    ) -> list: # Settled top-left corners
    "Nudges overlapping glycan boxes apart while keeping them near their anchors"
    pos = [[x, y] for x, y, _w, _h, _ax, _ay in placements]
    n = len(placements)
    if n < 2:
        return pos
    cw, ch = canvas
    for _ in range(iterations):
        hot = [False] * n
        for i in range(n):
            for j in range(i + 1, n):
                (xi, yi), (xj, yj) = pos[i], pos[j]
                wi, hi, wj, hj = placements[i][2], placements[i][3], placements[j][2], placements[j][3]
                ox = min(xi + wi, xj + wj) - max(xi, xj) + pad
                oy = min(yi + hi, yj + hj) - max(yi, yj) + pad
                if ox <= 0 or oy <= 0:
                    continue
                hot[i] = hot[j] = True
                if ox < oy:  # separate along whichever axis needs the smaller push
                    d = ox / 2.0 * (1 if xi < xj else -1)
                    pos[i][0] -= d
                    pos[j][0] += d
                else:
                    d = oy / 2.0 * (1 if yi < yj else -1)
                    pos[i][1] -= d
                    pos[j][1] += d
        for i, (x, y, w, h, _ax, _ay) in enumerate(placements):
            if not hot[i]:  # only drift home once this box has stopped colliding
                pos[i][0] += (x - pos[i][0]) * 0.05
                pos[i][1] += (y - pos[i][1]) * 0.05
            pos[i][0] = max(0.0, min(cw - w, pos[i][0]))
            pos[i][1] = max(0.0, min(ch - h, pos[i][1]))
        if not any(hot):
            break
    return pos


def _leader_line(anchor: tuple, # (x, y) of the data point
                 box: tuple # (x, y, w, h) of the placed glycan
                 ) -> str: # SVG path, or '' when the anchor sits inside the box
    "Draws a thin line from a data point to the glycan that was moved away from it"
    ax, ay = anchor
    x, y, w, h = box
    cx, cy = max(x, min(x + w, ax)), max(y, min(y + h, ay))
    if abs(cx - ax) < 1 and abs(cy - ay) < 1:
        return ''
    return ('<path d="M%.2f,%.2f L%.2f,%.2f" stroke="#7a7a7a" stroke-width="0.8" '
            'fill="none" stroke-linecap="round" />' % (ax, ay, cx, cy))


def annotate_figure(
        svg_input: str, # Input SVG file path
        scale_range: tuple[int, int] = (25, 80), # Min/max glycan dimensions
        compact: bool = False, # Use compact style
        glycan_size: str = 'medium', # Glycan size preset ('small', 'medium', 'large')
        filepath: str | Path = '', # Output file path
        scale_by_DE_res: pd.DataFrame | None = None, # Differential expression results (motif_analysis.get_differential_expression)
        x_thresh: float = 1, # X metric threshold
        y_thresh: float = 0.05, # P-value threshold
        x_metric: str = 'Log2FC' # X axis metric ('Log2FC', 'Effect size')
) -> str | None: # Modified SVG code
    "Replaces text labels with glycan drawings in SVG figure"
    from glycorender.render import pdf_to_svg_bytes
    glycan_size_dict = {'small': (0.1, -74), 'medium': (0.2, -55), 'large': (0.3, -49)}
    glyc_scale, glyc_offset = glycan_size_dict[glycan_size]
    glycan_scale = ''
    if scale_by_DE_res is not None:
        label_col = 'Glycan' if 'Glycan' in scale_by_DE_res.columns else 'Glycosite'
        if missing := {label_col, 'corr p-val', x_metric} - set(scale_by_DE_res.columns):
            raise ValueError(
                f"scale_by_DE_res is missing the column(s) {', '.join(sorted(missing))}; pass the output of get_differential_expression.")
        res_df = scale_by_DE_res.loc[
            (abs(scale_by_DE_res[x_metric]) > x_thresh) & (scale_by_DE_res['corr p-val'] < y_thresh)]
        labels = res_df[label_col].values.tolist()
        if labels:  # with nothing above the thresholds there is no scale to build, so everything is drawn at default size instead of crashing
            y = -np.log10(res_df['corr p-val'].values.tolist())
            glycan_scale = [y, labels]
            _y_min, _y_max = min(y), max(y)
            _y_range = max(_y_max - _y_min, 1e-6)
    # Get svg code
    svg_tmp = svg_input if isinstance(svg_input, str) and '<svg' in svg_input else Path(svg_input).read_text(
        encoding = "utf-8")
    # Get all text labels
    matches = re.findall(r"<!--.*-->[\s\S]*?<\/g>", svg_tmp)
    # Prepare for appending
    svg_tmp = svg_tmp.replace('</svg>', '')
    element_id = 0
    edit_svg = False
    drawn = []
    for match in matches:
        # Keep track of current label and position in figure
        current_label = _LABEL_PATTERN.findall(match)[0]
        if current_label.lower().startswith('terminal') and resolve_motif_name(current_label) is None:
            if _drawable(current_label.split('_')[-1]):
                edit_svg = True
        # Check if label is glycan
        if _drawable(current_label):
            edit_svg = True
        try:
            glycan = resolve_motif_name(current_label)[0]
            if not glycan.startswith('r') and (_drawable(
                    glycan) or "!" in glycan):  # glyco-regex motifs (r-prefixed) have no structure to draw, and their '!' is a lookbehind, not a negation
                edit_svg = True
        except Exception:
            pass
        # Delete text label, collect the glycan for placement once every label is known
        if edit_svg:
            transform_val = _TRANSFORM_PATTERN.findall(match)
            if not transform_val:
                edit_svg = False
                continue
            translate_part = re.search(r'translate\(([^)]+)\)', transform_val[0])
            if not translate_part:
                edit_svg = False
                continue
            anchor = [float(v) for v in re.split(r'[,\s]+', translate_part.group(1).strip())[:2]]
            # matplotlib defines each glyph once, inside the first text block that uses it, so dropping the
            # block wholesale would break every later <use> of those glyphs (silently blanking characters
            # in the title and axis labels); keep the definitions and delete only the drawn text
            svg_tmp = svg_tmp.replace(match, ''.join(re.findall(r'<defs>[\s\S]*?</defs>', match)))
            if glycan_scale == '' or current_label not in glycan_scale[
                1]:  # a label the DE table does not rank still gets drawn, just unscaled
                d = GlycoDraw(current_label, compact = compact, suppress = True, restrict_vocab = True)
            else:
                _dim = (scale_range[1] - scale_range[0]) * (
                        (glycan_scale[0][glycan_scale[1].index(current_label)] - _y_min) / _y_range) + scale_range[0]
                d = GlycoDraw(current_label, compact = compact, dim = _dim, suppress = True, restrict_vocab = True)
            svg_from_pdf = pdf_to_svg_bytes(d.as_svg())
            data = svg_from_pdf.replace('<?xml version="1.0" encoding="UTF-8"?>', '').replace('<?xml version="1.0"?>', '')
            id_matches = re.findall(r'(?:font_\d+_\d+|d\d+)', data)
            for idx in id_matches:
                data = data.replace(idx, 'd' + str(element_id))
                element_id += 1
            size = re.search(r'<svg[^>]*?width="([\d.]+)"[^>]*?height="([\d.]+)"', data)
            gw, gh = (float(size.group(1)) * glyc_scale, float(size.group(2)) * glyc_scale) if size else (0.0, 0.0)
            drawn.append((anchor[0], anchor[1] + glyc_offset * glyc_scale, gw, gh, anchor[0], anchor[1], data))
        edit_svg = False
    canvas = re.search(r'<svg[^>]*?width="([\d.]+)[a-z]*"[^>]*?height="([\d.]+)[a-z]*"', svg_tmp)
    canvas = (float(canvas.group(1)), float(canvas.group(2))) if canvas else (1000.0, 1000.0)
    for (_x, _y, gw, gh, ax, ay, data), (gx, gy) in zip(drawn, _spread_glycans([d[:6] for d in drawn], canvas)):
        svg_tmp += '\n' + _leader_line((ax, ay), (gx, gy, gw, gh))
        svg_tmp += '\n<g transform="translate(%.2f %.2f) scale(%s %s)">\n%s\n</g>' % (gx, gy, glyc_scale,
                                                                                      glyc_scale, data)
    svg_tmp += '</svg>'
    if filepath:
        filepath = Path(filepath)
        suffix = filepath.suffix.lower()
        if suffix not in {'.pdf', '.svg', '.png'}:
            raise ValueError(f"Cannot save to '{filepath.name}': filepath has to end in .svg, .pdf, or .png")
        filepath.parent.mkdir(parents = True, exist_ok = True)
        if suffix == '.pdf':
            from glycorender.render import simple_svg_to_pdf
            simple_svg_to_pdf(svg_tmp, str(filepath))
        elif suffix == '.svg':
            with open(filepath, 'w', encoding = "utf-8") as f:
                f.write(svg_tmp)
        else:
            from glycorender.render import simple_svg_to_png
            simple_svg_to_png(svg_tmp, str(filepath))
    else:
        return svg_tmp


def plot_glycans_excel(
        df: pd.DataFrame | pd.Series | list[str] | str | Path, # DataFrame or filepath with glycans, or the glycans themselves
        folder_filepath: str | Path, # Output folder path
        glycan_col_num: int | str = 0, # Glycan column index, or its name
        scaling_factor: float = 0.2, # Image scaling
        compact: bool = False, # Use compact style
        **kwargs # Passed on to GlycoDraw for every glycan, e.g., vertical, show_linkage, highlight_motif
) -> None:
    "Creates Excel file with SNFG glycan images in a new column"
    _, convert_svg_to_png = _get_glycorender()
    from glycorender.render import pdf_to_svg_bytes
    from openpyxl.drawing.image import Image as OpenpyxlImage
    from openpyxl.utils import get_column_letter
    import zipfile

    class _PngImage(OpenpyxlImage):
        "openpyxl only calls Pillow to learn a PNG's size and to hand its bytes back, both of which we already have"

        def __init__(self, data, width, height):
            self.ref = data
            self.width, self.height = width, height
            self.format = 'png'

        def _data(self):
            self.ref.seek(0)
            return self.ref.read()

    if isinstance(df, (str, Path)):
        df = pd.read_csv(df) if Path(df).suffix.lower() == ".csv" else pd.read_csv(df, sep = "\t") if Path(
            df).suffix.lower() == ".tsv" else pd.read_excel(df)
    elif not isinstance(df, pd.DataFrame):
        # A bare column or list of glycans becomes a one-column sheet; a Series would otherwise gain 'SNFG' as a row
        df = pd.DataFrame({getattr(df, 'name', None) or 'glycan': list(df)})
    else:
        df = df.copy()
    df["SNFG"] = np.nan
    image_column_number = df.columns.tolist().index("SNFG") + 1
    # Convert df_out to Excel; a directory gets the workbook as 'output.xlsx', an .xlsx path names it
    out = Path(folder_filepath)
    if out.suffix and out.suffix.lower() != '.xlsx' and not out.is_dir():  # an existing folder such as 'run.v2' is still a folder
        raise ValueError(
            f"folder_filepath has to be a directory (the workbook is then written as 'output.xlsx' inside it) "
            f"or an .xlsx file (got '{folder_filepath}').")
    out = out if out.suffix and not out.is_dir() else out / "output.xlsx"
    out.parent.mkdir(parents = True, exist_ok = True)
    writer = pd.ExcelWriter(out, engine = "openpyxl")
    df.to_excel(writer, index = False)
    # Get the active sheet
    sheet = writer.sheets["Sheet1"]
    column = df[glycan_col_num] if isinstance(glycan_col_num, str) else df.iloc[:, glycan_col_num]
    column_letter, column_width, svgs = get_column_letter(image_column_number), 0, {}
    for i, glycan_structure in enumerate(column):
        if isinstance(glycan_structure, (list, tuple)) and glycan_structure:
            glycan_structure = glycan_structure[0] if isinstance(glycan_structure[0], str) else glycan_structure[0][0]
        if isinstance(glycan_structure, str) and glycan_structure:
            # Generate glycan image using GlycoDraw
            try:
                drawing = GlycoDraw(glycan_structure, compact = compact, suppress = True, restrict_vocab = True, **kwargs)
            except Exception as e:
                raise ValueError(f"Could not draw the glycan in row {i + 2} of the sheet: {glycan_structure}") from e
            svg_data = drawing.as_svg()
            # Excel 2016+ draws the vector twin added below; other readers show the PNG, rasterized at 4x the
            # display size so it stays sharp on high-DPI screens and up to 400% zoom
            png_bytes = convert_svg_to_png(svg_data, scale = 8.0 * scaling_factor, return_bytes = True,
                                           shadow = drawing.shadow, sticker = drawing.sticker)
            # Office's SVG renderer ignores filters, so shadowed glycans keep only their PNG
            if not drawing.shadow:
                svgs[png_bytes] = pdf_to_svg_bytes(svg_data, sticker = drawing.sticker).encode('utf-8')
            # PNG IHDR carries the dimensions; the picture is displayed at a quarter of them
            img_width, img_height = [round(v / 4) for v in struct.unpack('>II', png_bytes[16:24])]
            img_for_excel = _PngImage(BytesIO(png_bytes), img_width, img_height)
            # Find the cell to insert the image
            cell = sheet.cell(row = i + 2,
                              column = image_column_number)  # +2 because Excel is 1-indexed and there's a header row
            # Insert the image into the cell
            sheet.add_image(img_for_excel, cell.coordinate)
            # Resize the cell to fit the image; a column width unit is 7 px (Calibri 11) and has to fit the widest
            column_width = max(column_width, img_width / 7)
            sheet.column_dimensions[column_letter].width = column_width
            sheet.row_dimensions[cell.row].height = img_height * 0.75
    # Save the workbook; closing the writer saves it and releases the file handle it has held open since creation
    writer.close()
    if not svgs:
        return
    # openpyxl cannot write Office's SVG picture extension, so each PNG gets its vector twin in the saved package
    with zipfile.ZipFile(out) as z:
        parts = {name: z.read(name) for name in z.namelist()}
    rels, xml = parts['xl/drawings/_rels/drawing1.xml.rels'].decode(), parts['xl/drawings/drawing1.xml'].decode()
    for media, rid in re.findall(r'Target="/(xl/media/image\d+)\.png" Id="(rId\d+)"', rels):
        parts[media + '.svg'] = svgs[parts[media + '.png']]
        rels = rels.replace('</Relationships>', '<Relationship Type="http://schemas.openxmlformats.org/officeDocument/'
                            f'2006/relationships/image" Target="/{media}.svg" Id="{rid}s"/></Relationships>')
        # lxml writes '"/>', the stdlib serializer openpyxl falls back to without it '" />'
        xml = re.sub(f'(<a:blip [^>]*?r:embed="{rid}"[^>]*?)\\s*/>',
                     '\\1><a:extLst><a:ext uri="{96DAC541-7B7A-43D3-8B79-37D633B846F1}"><asvg:svgBlip xmlns:asvg='
                     f'"http://schemas.microsoft.com/office/drawing/2016/SVG/main" r:embed="{rid}s"/>'
                     '</a:ext></a:extLst></a:blip>', xml)
    parts['xl/drawings/_rels/drawing1.xml.rels'], parts['xl/drawings/drawing1.xml'] = rels.encode(), xml.encode()
    parts['[Content_Types].xml'] = parts['[Content_Types].xml'].replace(
        b'<Default Extension="png"', b'<Default Extension="svg" ContentType="image/svg+xml"/><Default Extension="png"')
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, data in parts.items():
            z.writestr(name, data)
