from glycowork.glycan_data.loader import lib, unwrap, motif_list
from glycowork.motif.graph import glycan_to_nxGraph
from glycowork.motif.tokenization import get_core, get_modification
from glycowork.motif.processing import expand_lib, min_process_glycans

import cairo, cairosvg, cairocffi
import drawSvg as draw
import networkx as nx
import numpy as np
import sys
import re
from math import sin, cos, radians

def matches(line, opendelim='(', closedelim=')'):
  """
  Find matching pairs of delimiters in a given string.\n
  | Arguments:
  | :-
  | line (str): The string to search for delimiter pairs.
  | opendelim (str, optional): The character to use as the opening delimiter. Defaults to '('.
  | closedelim (str, optional): The character to use as the closing delimiter. Defaults to ')'.\n
  | Returns:
  | :-
  | tuple: A tuple containing the start and end positions of the matched delimiter pair, and the current depth of the stack.\n
  ref: https://stackoverflow.com/questions/5454322/python-how-to-match-nested-parentheses-with-regex
  """
  
  stack = []

  for m in re.finditer(r'[{}{}]'.format(opendelim, closedelim), line):
      pos = m.start()

      if line[pos-1] == '\\':
          # skip escape sequence
          continue

      c = line[pos]

      if c == opendelim:
          stack.append(pos+1)

      elif c == closedelim:
          if len(stack) > 0:
              prevpos = stack.pop()
              # print("matched", prevpos, pos, line[prevpos:pos])
              yield (prevpos, pos, len(stack))
          else:
              # error
              print("encountered extraneous closing quote at pos {}: '{}'".format(pos, line[pos:] ))
              pass

  if len(stack) > 0:
      for pos in stack:
          print("expecting closing quote to match open quote starting at: '{}'"
                .format(line[pos-1:]))
            
class RecordingSurface(cairosvg.surface.Surface):
  # https://github.com/cduck/drawSvg/issues/25
    """A surface that records draw commands."""
    def _create_surface(self, width, height):
        cairo_surface = cairocffi.RecordingSurface(
                cairocffi.CONTENT_COLOR_ALPHA, None)
        return cairo_surface, width, height

def get_bounding_box(d, pad=0, resolution=1/256, max_size=10000):
  # https://github.com/cduck/drawSvg/issues/25
    rbox = (-max_size, -max_size, 2*max_size, 2*max_size)
    # Hack, add an argument to asSvg instead
    svg_lines = d.asSvg().split('\n')
    svg_lines[2] = f'viewBox="{rbox[0]}, {rbox[1]}, {rbox[2]}, {rbox[3]}">'
    svg_code = '\n'.join(svg_lines)
    
    t = cairosvg.parser.Tree(bytestring=svg_code)
    s = RecordingSurface(t, None, 72, scale=1/resolution)
    b = s.cairo.ink_extents()
    
    return (
        rbox[0] + b[0]*resolution - pad,
        -(rbox[1]+b[1]*resolution)-b[3]*resolution - pad,
        b[2]*resolution + pad*2,
        b[3]*resolution + pad*2,
    )

def fit_to_contents(d, pad=0, resolution=1/256, max_size=10000):
  # https://github.com/cduck/drawSvg/issues/25
    bb = get_bounding_box(d, pad=pad, resolution=resolution, max_size=max_size)
    d.viewBox = (bb[0], -bb[1]-bb[3], bb[2], bb[3])
    d.width, d.height = bb[2], bb[3]
    
    # Debug: Draw bounding rectangle
    # d.append(draw.Rectangle(*bb, fill='none', stroke_width=2,
                            # stroke='red', stroke_dasharray='5 2'))

# Adjusted SNFG color palette
snfg_white = '#ffffff'
snfg_alt_blue = '#0385ae'
snfg_green = '#058f60'
snfg_yellow = '#fcc326'
snfg_light_blue = '#91d3e3'
snfg_pink = '#f39ea0'
snfg_purple = '#a15989'
snfg_brown = '#9f6d55'
snfg_orange = '#ef6130'
snfg_red = '#C23537'

draw_lib = expand_lib(lib, ['Re'])

sugar_dict = {
  "Hex": ['Hex', snfg_white, False],
  "Glc": ['Hex', snfg_alt_blue, False],
  "Glcf": ['Hex', snfg_alt_blue, True],
  "Man": ['Hex', snfg_green, False],
  "Manf": ['Hex', snfg_green, True],
  "Gal": ['Hex', snfg_yellow, False],
  "Galf": ['Hex', snfg_yellow, True],
  "Gul": ['Hex', snfg_orange, False],
  "Alt": ['Hex', snfg_pink, False],
  "All": ['Hex', snfg_purple, False],
  "Tal": ['Hex', snfg_light_blue, False],
  "Ido": ['Hex', snfg_brown, False],

  "HexNAc": ['HexNAc', snfg_white, False],
  "GlcNAc": ['HexNAc', snfg_alt_blue, False],
  "GlcfNAc": ['HexNAc', snfg_alt_blue, True],
  "ManNAc": ['HexNAc', snfg_green, False],
  "ManfNAc": ['HexNAc', snfg_green, True],
  "GalNAc": ['HexNAc', snfg_yellow, False],
  "GalfNAc": ['HexNAc', snfg_yellow, True],
  "GulNAc": ['HexNAc', snfg_orange, False],
  "AltNAc": ['HexNAc', snfg_pink, False],
  "AllNAc": ['HexNAc', snfg_purple, False],
  "TalNAc": ['HexNAc', snfg_light_blue, False],
  "IdoNAc": ['HexNAc', snfg_brown, False],

  "HexN": ['HexN', snfg_white, False],
  "GlcN": ['HexN', snfg_alt_blue, False],
  "ManN": ['HexN', snfg_green, False],
  "GalN": ['HexN', snfg_yellow, False],
  "GulN": ['HexN', snfg_orange, False],
  "AltN": ['HexN', snfg_pink, False],
  "AllN": ['HexN', snfg_purple, False],
  "TalN": ['HexN', snfg_light_blue, False],
  "IdoN": ['HexN', snfg_brown, False],

  "HexA": ['HexA', snfg_white, False],
  "GlcA": ['HexA', snfg_alt_blue, False],
  "ManA": ['HexA', snfg_green, False],
  "GalA": ['HexA', snfg_yellow, False],
  "GulA": ['HexA', snfg_orange, False],
  "AltA": ['HexA_2', snfg_pink, False],
  "AllA": ['HexA', snfg_purple, False],
  "TalA": ['HexA', snfg_light_blue, False],
  "IdoA": ['HexA_2', snfg_brown, False],

  "dHex": ['dHex', snfg_white, False],
  "Qui": ['dHex', snfg_alt_blue, False],
  "Rha": ['dHex', snfg_green, False],
  "6dGul": ['dHex', snfg_orange, False],
  "6dAlt": ['dHex', snfg_pink, False],
  "6dAltf": ['dHex', snfg_pink, True],
  "6dTal": ['dHex', snfg_light_blue, False],
  "Fuc": ['dHex', snfg_red, False],
  "Fucf": ['dHex', snfg_red, True],

  "dHexNAc": ['dHexNAc', snfg_white, False],
  "QuiNAc": ['dHexNAc', snfg_alt_blue, False],
  "RhaNAc": ['dHexNAc', snfg_green, False],
  "6dAltNAc": ['dHexNAc', snfg_pink, False],
  "6dTalNAc": ['dHexNAc', snfg_light_blue, False],
  "FucNAc": ['dHexNAc', snfg_red, False],
  "FucfNAc": ['dHexNAc', snfg_red, True],

  "ddHex": ['ddHex', snfg_white, False],
  "Oli": ['ddHex', snfg_alt_blue, False],
  "Tyv": ['ddHex', snfg_green, False],
  "Abe": ['ddHex', snfg_orange, False],
  "Par": ['ddHex', snfg_pink, False],
  "Dig": ['ddHex', snfg_purple, False],
  "Col": ['ddHex', snfg_light_blue, False],

  "Pen": ['Pen', snfg_white, False],
  "Ara": ['Pen', snfg_green, False],
  "Araf": ['Pen', snfg_green, True],
  "Lyx": ['Pen', snfg_yellow, False],
  "Lyxf": ['Pen', snfg_yellow, True],
  "Xyl": ['Pen', snfg_orange, False],
  "Xylf": ['Pen', snfg_orange, True],
  "Rib": ['Pen', snfg_pink, False],
  "Ribf": ['Pen', snfg_pink, True],

  "dNon": ['dNon', snfg_white, False],
  "Kdn": ['dNon', snfg_green, False],
  "Neu5Ac": ['dNon', snfg_purple, False],
  "Neu5Gc": ['dNon', snfg_light_blue, False],
  "Neu": ['dNon', snfg_brown, False],
  "Sia": ['dNon', snfg_red, False],

  "ddNon": ['ddNon', snfg_white, False],
  "Pse": ['ddNon', snfg_green, False],
  "Leg": ['ddNon', snfg_yellow, False],
  "Aci": ['ddNon', snfg_pink, False],
  "4eLeg": ['ddNon', snfg_light_blue, False],
  
  "Unknown": ['Unknown', snfg_white, False],
  "Bac": ['Unknown', snfg_alt_blue, False],
  "LDManHep": ['Unknown', snfg_green, False],
  "Kdo": ['Unknown', snfg_yellow, False],
  "Kdof": ['Unknown', snfg_yellow, True],
  "Dha": ['Unknown', snfg_orange, False],
  "DDManHep": ['Unknown', snfg_pink, False],
  "MurNAc": ['Unknown', snfg_purple, False],
  "MurNGc": ['Unknown', snfg_light_blue, False],
  "Mur": ['Unknown', snfg_brown, False],

  "Assigned": ['Assigned', snfg_white, False],
  "Api": ['Assigned', snfg_alt_blue, False],
  "Apif": ['Assigned', snfg_alt_blue, True],
  "Fru": ['Assigned', snfg_green, False],
  "Fruf": ['Assigned', snfg_green, True],
  "Tag": ['Assigned', snfg_yellow, False],
  "Sor": ['Assigned', snfg_orange, False],
  "Psi": ['Assigned', snfg_pink, False],

  "Re": ['empty', snfg_white, False],
}   

def draw_shape(shape, color, x_pos, y_pos, modification = '', dim = 50, furanose = False, conf = ''): 
  """draw individual monosaccharides in shapes & colors according to the SNFG nomenclature\n
  | Arguments:
  | :-
  | shape (string): monosaccharide class; shape of icon
  | color (string): monosaccharide identity; color of icon
  | x_pos (int): x coordinate of icon on drawing canvas
  | y_pos (int): y coordinate of icon on drawing canvas
  | modification (string): icon text annotation; used for post-biosynthetic modifications
  | dim (int): arbitrary dimention unit; necessary for drawsvg; inconsequential when drawing is exported as svg graphics\n
  | Returns:
  | :-  
  | 
  """

  if shape == 'Hex':
    # xexose - circle
    d.append(draw.Circle(0-x_pos*dim, 0+y_pos*dim, dim/2, fill=color, stroke_width=0.04*dim, stroke='black'))
    # text annotation
    p = draw.Path(stroke_width=0)
    p.M(0-x_pos*dim-dim, 0+y_pos*dim+0.5*dim)
    p.L(0-x_pos*dim+dim, 0+y_pos*dim+0.5*dim)  
    d.append(p)
    d.append(draw.Text(modification, dim*0.35, path=p, text_anchor='middle', valign='middle', lineOffset=-0.75))
    if furanose == True:
      p = draw.Path(stroke_width=0)
      p.M(0-x_pos*dim-dim, 0+y_pos*dim)
      p.L(0-x_pos*dim+dim, 0+y_pos*dim)  
      d.append(p)
      d.append(draw.Text('f', dim*0.30, path=p, text_anchor='middle', valign='middle'))
    # ring configuration
    p = draw.Path(stroke_width=0)
    p.M(0-x_pos*dim-dim, 0+y_pos*dim)
    p.L(0-x_pos*dim+dim, 0+y_pos*dim)  
    d.append(p)
    d.append(draw.Text(conf, dim*0.30, path=p, text_anchor='middle', valign='middle'))

  if shape == 'HexNAc':
    # hexnac - square
    d.append(draw.Rectangle((0-x_pos*dim)-(dim/2),(0+y_pos*dim)-(dim/2),dim,dim, fill=color, stroke_width=0.04*dim, stroke = 'black'))
    # text annotation
    p = draw.Path(stroke_width=0)
    p.M(0-x_pos*dim-dim, 0+y_pos*dim+0.5*dim)
    p.L(0-x_pos*dim+dim, 0+y_pos*dim+0.5*dim)  
    d.append(p)
    d.append(draw.Text(modification, dim*0.35, path=p, text_anchor='middle', valign='middle', lineOffset=-0.75))
    if furanose == True:
      p = draw.Path(stroke_width=0)
      p.M(0-x_pos*dim-dim, 0+y_pos*dim)
      p.L(0-x_pos*dim+dim, 0+y_pos*dim)  
      d.append(p)
      d.append(draw.Text('f', dim*0.30, path=p, text_anchor='middle', valign='middle'))
    # ring configuration
    p = draw.Path(stroke_width=0)
    p.M(0-x_pos*dim-dim, 0+y_pos*dim)
    p.L(0-x_pos*dim+dim, 0+y_pos*dim)  
    d.append(p)
    d.append(draw.Text(conf, dim*0.30, path=p, text_anchor='middle', valign='middle'))    
  
  if shape == 'HexN':
    # hexosamine - crossed square
    d.append(draw.Rectangle((0-x_pos*dim)-(dim/2),(0+y_pos*dim)-(dim/2),dim,dim, fill='white', stroke_width=0.04*dim, stroke = 'black'))
    d.append(draw.Lines((0-x_pos*dim)-(dim/2), (0+y_pos*dim)+(dim/2),
                        (0-x_pos*dim)+(dim/2), (0+y_pos*dim)+(dim/2),
                        (0-x_pos*dim)+(dim/2), (0+y_pos*dim)-(dim/2),
                        (0-x_pos*dim)-(dim/2), (0+y_pos*dim)+(dim/2),
            close=True,
            fill=color,
            stroke='black', stroke_width = 0))
    p = draw.Path(stroke_width=0.04*dim, stroke='black',)
    p.M((0-x_pos*dim)-(dim/2), (0+y_pos*dim)+(dim/2))
    p.L((0-x_pos*dim)+(dim/2), (0+y_pos*dim)+(dim/2))  
    d.append(p)
    p = draw.Path(stroke_width=0.04*dim, stroke='black',)
    p.M((0-x_pos*dim)+(dim/2), (0+y_pos*dim)+(dim/2))
    p.L((0-x_pos*dim)+(dim/2), (0+y_pos*dim)-(dim/2))  
    d.append(p)
    p = draw.Path(stroke_width=0.04*dim, stroke='black',)
    p.M((0-x_pos*dim)+(dim/2), (0+y_pos*dim)-(dim/2))
    p.L((0-x_pos*dim)-(dim/2), (0+y_pos*dim)+(dim/2))  
    d.append(p)
    # text annotation
    p = draw.Path(stroke_width=0)
    p.M(0-x_pos*dim-dim, 0+y_pos*dim+0.5*dim)
    p.L(0-x_pos*dim+dim, 0+y_pos*dim+0.5*dim)  
    d.append(p)
    d.append(draw.Text(modification, dim*0.35, path=p, text_anchor='middle', valign='middle', lineOffset=-0.75))
    # ring configuration
    p = draw.Path(stroke_width=0)
    p.M(0-x_pos*dim-dim, 0+y_pos*dim)
    p.L(0-x_pos*dim+dim, 0+y_pos*dim)  
    d.append(p)
    d.append(draw.Text(conf, dim*0.30, path=p, text_anchor='middle', valign='middle'))    
  
  if shape == 'HexA':
    # hexuronate - divided diamond
    d.append(draw.Lines((0-x_pos*dim),         (0+y_pos*dim)+(dim/2),
                        (0-x_pos*dim)+(dim/2), (0+y_pos*dim),
                        (0-x_pos*dim),         (0+y_pos*dim)-(dim/2),
                        (0-x_pos*dim)-(dim/2), (0+y_pos*dim),
            close=True,
            fill='white',
            stroke='black', stroke_width = 0.04*dim))

    d.append(draw.Lines((0-x_pos*dim)-(dim/2), (0+y_pos*dim),
                        (0-x_pos*dim), (0+y_pos*dim)+(dim/2),
                        (0-x_pos*dim)+(dim/2), (0+y_pos*dim),
                        (0-x_pos*dim)-(dim/2), (0+y_pos*dim),
            close=True,
            fill=color,
            stroke='black', stroke_width = 0))
    p = draw.Path(stroke_width=0.04*dim, stroke='black',)
    p.M((0-x_pos*dim)-(dim/2), (0+y_pos*dim))
    p.L((0-x_pos*dim), (0+y_pos*dim)+(dim/2))  
    d.append(p)
    p = draw.Path(stroke_width=0.04*dim, stroke='black',)
    p.M((0-x_pos*dim), (0+y_pos*dim)+(dim/2))
    p.L((0-x_pos*dim)+(dim/2), (0+y_pos*dim))  
    d.append(p)
    p = draw.Path(stroke_width=0.04*dim, stroke='black',)
    p.M((0-x_pos*dim)+(dim/2), (0+y_pos*dim))
    p.L((0-x_pos*dim)-(dim/2), (0+y_pos*dim))  
    d.append(p)
    # text annotation
    p = draw.Path(stroke_width=0)
    p.M(0-x_pos*dim-dim, 0+y_pos*dim+0.5*dim)
    p.L(0-x_pos*dim+dim, 0+y_pos*dim+0.5*dim)  
    d.append(p)
    d.append(draw.Text(modification, dim*0.35, path=p, text_anchor='middle', valign='middle', lineOffset=-0.75))
    # ring configuration
    p = draw.Path(stroke_width=0)
    p.M(0-x_pos*dim-dim, 0+y_pos*dim)
    p.L(0-x_pos*dim+dim, 0+y_pos*dim)  
    d.append(p)
    d.append(draw.Text(conf, dim*0.30, path=p, text_anchor='middle', valign='middle'))    
  
  if shape == 'HexA_2':
    # hexuronate - divided diamond (colors flipped)
    # AltA / IdoA
    d.append(draw.Lines((0-x_pos*dim),         (0+y_pos*dim)+(dim/2),
                        (0-x_pos*dim)+(dim/2), (0+y_pos*dim),
                        (0-x_pos*dim),         (0+y_pos*dim)-(dim/2),
                        (0-x_pos*dim)-(dim/2), (0+y_pos*dim),
            close=True,
            fill='white',
            stroke='black', stroke_width = 0.04*dim))

    d.append(draw.Lines((0-x_pos*dim)-(dim/2), (0+y_pos*dim),
                        (0-x_pos*dim), (0+y_pos*dim)-(dim/2),
                        (0-x_pos*dim)+(dim/2), (0+y_pos*dim),
                        (0-x_pos*dim)-(dim/2), (0+y_pos*dim),
            close=True,
            fill=color,
            stroke='black', stroke_width = 0))
    p = draw.Path(stroke_width=0.04*dim, stroke='black',)
    p.M((0-x_pos*dim)-(dim/2), (0+y_pos*dim))
    p.L((0-x_pos*dim), (0+y_pos*dim)-(dim/2))  
    d.append(p)
    p = draw.Path(stroke_width=0.04*dim, stroke='black',)
    p.M((0-x_pos*dim), (0+y_pos*dim)-(dim/2))
    p.L((0-x_pos*dim)+(dim/2), (0+y_pos*dim))  
    d.append(p)
    p = draw.Path(stroke_width=0.04*dim, stroke='black',)
    p.M((0-x_pos*dim)+(dim/2), (0+y_pos*dim))
    p.L((0-x_pos*dim)-(dim/2), (0+y_pos*dim))  
    d.append(p)
    # text annotation
    p = draw.Path(stroke_width=0)
    p.M(0-x_pos*dim-dim, 0+y_pos*dim+0.5*dim)
    p.L(0-x_pos*dim+dim, 0+y_pos*dim+0.5*dim)  
    d.append(p)
    d.append(draw.Text(modification, dim*0.35, path=p, text_anchor='middle', valign='middle', lineOffset=-0.75))
    # ring configuration
    p = draw.Path(stroke_width=0)
    p.M(0-x_pos*dim-dim, 0+y_pos*dim)
    p.L(0-x_pos*dim+dim, 0+y_pos*dim)  
    d.append(p)
    d.append(draw.Text(conf, dim*0.30, path=p, text_anchor='middle', valign='middle'))    
  
  if shape == 'dHex':
    # deoxyhexose - triangle
    d.append(draw.Lines((0-x_pos*dim)-(0.5*dim), (0+y_pos*dim)-(((3**0.5)/2)*dim*0.5), #-(dim*1/3)
                    (0-x_pos*dim)+(dim/2)-(0.5*dim), (0+y_pos*dim)+(((3**0.5)/2)*dim)-(((3**0.5)/2)*dim*0.5),
                    (0-x_pos*dim)+(dim)-(0.5*dim), (0+y_pos*dim)-(((3**0.5)/2)*dim*0.5),
            close=True,
            fill=color,
            stroke='black', stroke_width = 0.04*dim))
    # text annotation
    p = draw.Path(stroke_width=0)
    p.M(0-x_pos*dim-dim, 0+y_pos*dim+0.5*dim)
    p.L(0-x_pos*dim+dim, 0+y_pos*dim+0.5*dim)  
    d.append(p)
    d.append(draw.Text(modification, dim*0.35, path=p, text_anchor='middle', valign='middle', lineOffset=-0.75))
    # ring configuration
    p = draw.Path(stroke_width=0)
    p.M(0-x_pos*dim-dim, 0+y_pos*dim)
    p.L(0-x_pos*dim+dim, 0+y_pos*dim)  
    d.append(p)
    d.append(draw.Text(conf, dim*0.30, path=p, text_anchor='middle', valign='middle'))    
  
  if shape == 'dHexNAc':
    # deoxyhexnac - divided triangle
    d.append(draw.Lines((0-x_pos*dim)-(0.5*dim), (0+y_pos*dim)-(((3**0.5)/2)*dim*0.5), #-(dim*1/3) for center of triangle
                    (0-x_pos*dim)+(dim/2)-(0.5*dim), (0+y_pos*dim)+(((3**0.5)/2)*dim)-(((3**0.5)/2)*dim*0.5), # -(dim*1/3) for bottom alignment
                    (0-x_pos*dim)+(dim)-(0.5*dim), (0+y_pos*dim)-(((3**0.5)/2)*dim*0.5), # -(((3**0.5)/2)*dim*0.5) for half of triangle height
            close=True,
            fill='white',
            stroke='black', stroke_width = 0.04*dim))
    d.append(draw.Lines((0-x_pos*dim), (0+y_pos*dim)-(((3**0.5)/2)*dim*0.5), #-(dim*1/3)
                    (0-x_pos*dim)+(dim/2)-(0.5*dim), (0+y_pos*dim)+(((3**0.5)/2)*dim)-(((3**0.5)/2)*dim*0.5),
                    (0-x_pos*dim)+(dim)-(0.5*dim), (0+y_pos*dim)-(((3**0.5)/2)*dim*0.5),
            close=True,
            fill=color,
            stroke='black', stroke_width = 0))
    p = draw.Path(stroke_width=0.04*dim, stroke='black',)
    p.M((0-x_pos*dim), (0+y_pos*dim)-(((3**0.5)/2)*dim*0.5))
    p.L((0-x_pos*dim)+(dim/2)-(0.5*dim), (0+y_pos*dim)+(((3**0.5)/2)*dim)-(((3**0.5)/2)*dim*0.5))  
    d.append(p)
    p = draw.Path(stroke_width=0.04*dim, stroke='black',)
    p.M((0-x_pos*dim)+(dim/2)-(0.5*dim), (0+y_pos*dim)+(((3**0.5)/2)*dim)-(((3**0.5)/2)*dim*0.5))
    p.L((0-x_pos*dim)+(dim)-(0.5*dim), (0+y_pos*dim)-(((3**0.5)/2)*dim*0.5))  
    d.append(p)
    p = draw.Path(stroke_width=0.04*dim, stroke='black',)
    p.M((0-x_pos*dim)+(dim)-(0.5*dim), (0+y_pos*dim)-(((3**0.5)/2)*dim*0.5))
    p.L((0-x_pos*dim), (0+y_pos*dim)-(((3**0.5)/2)*dim*0.5))  
    d.append(p)
    # text annotation
    p = draw.Path(stroke_width=0)
    p.M(0-x_pos*dim-dim, 0+y_pos*dim+0.5*dim)
    p.L(0-x_pos*dim+dim, 0+y_pos*dim+0.5*dim)  
    d.append(p)
    d.append(draw.Text(modification, dim*0.35, path=p, text_anchor='middle', valign='middle', lineOffset=-0.75))
    # ring configuration
    p = draw.Path(stroke_width=0)
    p.M(0-x_pos*dim-dim, 0+y_pos*dim)
    p.L(0-x_pos*dim+dim, 0+y_pos*dim)  
    d.append(p)
    d.append(draw.Text(conf, dim*0.30, path=p, text_anchor='middle', valign='middle'))    
  
  if shape == 'ddHex':
    # dideoxyhex - flat rectangle
    d.append(draw.Lines((0-x_pos*dim)-(dim/2),         (0+y_pos*dim)+(dim*7/12*0.5), #-(dim*0.5/12)
                        (0-x_pos*dim)+(dim/2),         (0+y_pos*dim)+(dim*7/12*0.5),
                        (0-x_pos*dim)+(dim/2),         (0+y_pos*dim)-(dim*7/12*0.5),
                        (0-x_pos*dim)-(dim/2),         (0+y_pos*dim)-(dim*7/12*0.5),
                        close=True,
                        fill=color,
                        stroke='black', stroke_width = 0.04*dim))
    # text annotation
    p = draw.Path(stroke_width=0)
    p.M(0-x_pos*dim-dim, 0+y_pos*dim+0.5*dim)
    p.L(0-x_pos*dim+dim, 0+y_pos*dim+0.5*dim)  
    d.append(p)
    d.append(draw.Text(modification, dim*0.35, path=p, text_anchor='middle', valign='middle', lineOffset=-0.75))
    # ring configuration
    p = draw.Path(stroke_width=0)
    p.M(0-x_pos*dim-dim, 0+y_pos*dim)
    p.L(0-x_pos*dim+dim, 0+y_pos*dim)  
    d.append(p)
    d.append(draw.Text(conf, dim*0.30, path=p, text_anchor='middle', valign='middle'))    
  
  if shape == 'Pen':
    # pentose - star
    d.append(draw.Lines((0-x_pos*dim)+0 ,         (0+y_pos*dim)+(0.5*dim)/cos(radians(18)),
                    (0-x_pos*dim)+((0.25*dim)/cos(radians(18)))*cos(radians(54)) ,         (0+y_pos*dim)+((0.25*dim)/cos(radians(18)))*sin(radians(54)),
                    (0-x_pos*dim)+((0.5*dim)/cos(radians(18)))*cos(radians(18)) ,         (0+y_pos*dim)+((0.5*dim)/cos(radians(18)))*sin(radians(18)),
                    (0-x_pos*dim)+((0.25*dim)/cos(radians(18)))*cos(radians(18)) ,         (0+y_pos*dim)-((0.25*dim)/cos(radians(18)))*sin(radians(18)),
                    (0-x_pos*dim)+((0.5*dim)/cos(radians(18)))*cos(radians(54)) ,         (0+y_pos*dim)-((0.5*dim)/cos(radians(18)))*sin(radians(54)),
                    (0-x_pos*dim)+0 ,         (0+y_pos*dim)-(0.25*dim)/cos(radians(18)),
                    (0-x_pos*dim)-((0.5*dim)/cos(radians(18)))*cos(radians(54)) ,         (0+y_pos*dim)-((0.5*dim)/cos(radians(18)))*sin(radians(54)),
                    (0-x_pos*dim)-((0.25*dim)/cos(radians(18)))*cos(radians(18)) ,         (0+y_pos*dim)-((0.25*dim)/cos(radians(18)))*sin(radians(18)),
                    (0-x_pos*dim)+-((0.5*dim)/cos(radians(18)))*cos(radians(18)) ,         (0+y_pos*dim)+((0.5*dim)/cos(radians(18)))*sin(radians(18)),
                    (0-x_pos*dim)-((0.25*dim)/cos(radians(18)))*cos(radians(54)) ,         (0+y_pos*dim)+((0.25*dim)/cos(radians(18)))*sin(radians(54)),
                        close=True,
                        fill= color,
                        stroke='black', stroke_width = 0.04*dim))
    # text annotation
    p = draw.Path(stroke_width=0)
    p.M(0-x_pos*dim-dim, 0+y_pos*dim+0.5*dim)
    p.L(0-x_pos*dim+dim, 0+y_pos*dim+0.5*dim)  
    d.append(p)
    d.append(draw.Text(modification, dim*0.35, path=p, text_anchor='middle', valign='middle', lineOffset=-0.75))
    # ring configuration
    p = draw.Path(stroke_width=0)
    p.M(0-x_pos*dim-dim, 0+y_pos*dim)
    p.L(0-x_pos*dim+dim, 0+y_pos*dim)  
    d.append(p)
    d.append(draw.Text(conf, dim*0.30, path=p, text_anchor='middle', valign='middle'))    
  
  if shape == 'dNon':
    # deoxynonulosonate - diamond
    d.append(draw.Lines((0-x_pos*dim),         (0+y_pos*dim)+(dim/2),
                        (0-x_pos*dim)+(dim/2), (0+y_pos*dim),
                        (0-x_pos*dim),         (0+y_pos*dim)-(dim/2),
                        (0-x_pos*dim)-(dim/2), (0+y_pos*dim),
            close=True,
            fill=color,
            stroke='black', stroke_width = 0.04*dim))
    # text annotation
    p = draw.Path(stroke_width=0)
    p.M(0-x_pos*dim-dim, 0+y_pos*dim+0.5*dim)
    p.L(0-x_pos*dim+dim, 0+y_pos*dim+0.5*dim)  
    d.append(p)
    d.append(draw.Text(modification, dim*0.35, path=p, text_anchor='middle', valign='middle', lineOffset=-0.75))
    # ring configuration
    p = draw.Path(stroke_width=0)
    p.M(0-x_pos*dim-dim, 0+y_pos*dim)
    p.L(0-x_pos*dim+dim, 0+y_pos*dim)  
    d.append(p)
    d.append(draw.Text(conf, dim*0.30, path=p, text_anchor='middle', valign='middle'))    
  
  if shape == 'ddNon':
    # dideoxynonulosonate - flat diamond
    d.append(draw.Lines((0-x_pos*dim),         (0+y_pos*dim)+(dim/2)-(dim*1/8),
                        (0-x_pos*dim)+(dim/2)+(dim*1/8), (0+y_pos*dim),
                        (0-x_pos*dim),         (0+y_pos*dim)-(dim/2)+(dim*1/8),
                        (0-x_pos*dim)-(dim/2)-(dim*1/8), (0+y_pos*dim),
                        close=True,
                        fill=color,
                        stroke='black', stroke_width = 0.04*dim))
    # text annotation
    p = draw.Path(stroke_width=0)
    p.M(0-x_pos*dim-dim, 0+y_pos*dim+0.5*dim)
    p.L(0-x_pos*dim+dim, 0+y_pos*dim+0.5*dim)  
    d.append(p)
    d.append(draw.Text(modification, dim*0.35, path=p, text_anchor='middle', valign='middle', lineOffset=-0.75))
    # ring configuration
    p = draw.Path(stroke_width=0)
    p.M(0-x_pos*dim-dim, 0+y_pos*dim)
    p.L(0-x_pos*dim+dim, 0+y_pos*dim)  
    d.append(p)
    d.append(draw.Text(conf, dim*0.30, path=p, text_anchor='middle', valign='middle'))    
   
  if shape == 'Unknown':
    # unknown - flat hexagon
    d.append(draw.Lines((0-x_pos*dim)-(dim/2)+(dim*1/8),         (0+y_pos*dim)+(dim/2)-(dim*1/8),
                        (0-x_pos*dim)+(dim/2)-(dim*1/8),         (0+y_pos*dim)+(dim/2)-(dim*1/8),
                        (0-x_pos*dim)+(dim/2)-(dim*1/8)+(dim*0.2),         (0+y_pos*dim),
                        (0-x_pos*dim)+(dim/2)-(dim*1/8),         (0+y_pos*dim)-(dim/2)+(dim*1/8),
                        (0-x_pos*dim)-(dim/2)+(dim*1/8),         (0+y_pos*dim)-(dim/2)+(dim*1/8),
                        (0-x_pos*dim)-(dim/2)+(dim*1/8)-(dim*0.2),         (0+y_pos*dim),
                        close=True,
                        fill=color,
                        stroke='black', stroke_width = 0.04*dim))
    # text annotation
    p = draw.Path(stroke_width=0)
    p.M(0-x_pos*dim-dim, 0+y_pos*dim+0.5*dim)
    p.L(0-x_pos*dim+dim, 0+y_pos*dim+0.5*dim)  
    d.append(p)
    d.append(draw.Text(modification, dim*0.35, path=p, text_anchor='middle', valign='middle', lineOffset=-0.75))
    # ring configuration
    p = draw.Path(stroke_width=0)
    p.M(0-x_pos*dim-dim, 0+y_pos*dim)
    p.L(0-x_pos*dim+dim, 0+y_pos*dim)  
    d.append(p)
    d.append(draw.Text(conf, dim*0.30, path=p, text_anchor='middle', valign='middle'))    
  
  if shape == 'Assigned':
    # assigned - pentagon
    d.append(draw.Lines((0-x_pos*dim)+0 ,         (0+y_pos*dim)+(0.5*dim)/cos(radians(18)),
                        (0-x_pos*dim)+((0.5*dim)/cos(radians(18)))*cos(radians(18)) ,         (0+y_pos*dim)+((0.5*dim)/cos(radians(18)))*sin(radians(18)),
                        (0-x_pos*dim)+((0.5*dim)/cos(radians(18)))*cos(radians(54)) ,         (0+y_pos*dim)-((0.5*dim)/cos(radians(18)))*sin(radians(54)),
                        (0-x_pos*dim)-((0.5*dim)/cos(radians(18)))*cos(radians(54)) ,         (0+y_pos*dim)-((0.5*dim)/cos(radians(18)))*sin(radians(54)),
                        (0-x_pos*dim)+-((0.5*dim)/cos(radians(18)))*cos(radians(18)) ,         (0+y_pos*dim)+((0.5*dim)/cos(radians(18)))*sin(radians(18)),
                        close=True,
                        fill= color,
                        stroke='black', stroke_width = 0.04*dim))
    # text annotation
    p = draw.Path(stroke_width=0)
    p.M(0-x_pos*dim-dim, 0+y_pos*dim+0.5*dim)
    p.L(0-x_pos*dim+dim, 0+y_pos*dim+0.5*dim)  
    d.append(p)
    d.append(draw.Text(modification, dim*0.35, path=p, text_anchor='middle', valign='middle', lineOffset=-0.75))
    # ring configuration
    p = draw.Path(stroke_width=0)
    p.M(0-x_pos*dim-dim, 0+y_pos*dim)
    p.L(0-x_pos*dim+dim, 0+y_pos*dim)  
    d.append(p)
    d.append(draw.Text(conf, dim*0.30, path=p, text_anchor='middle', valign='middle'))    
  if shape == 'empty':
    d.append(draw.Circle(0-x_pos*dim, 0+y_pos*dim, dim/2, fill='none', stroke_width=0.04*dim, stroke='none'))
    # text annotation
    p = draw.Path(stroke_width=0)
    p.M(0-x_pos*dim-dim, 0+y_pos*dim+0.5*dim)
    p.L(0-x_pos*dim+dim, 0+y_pos*dim+0.5*dim)  
    d.append(p)
    d.append(draw.Text(modification, dim*0.35, path=p, text_anchor='middle', valign='middle', lineOffset=-0.75))

def add_bond(x_start, x_stop, y_start, y_stop, label = '', dim = 50, compact = False):
  """drawing lines (bonds) between start/stop coordinates\n
  | Arguments:
  | :-
  | x_start (int): x1 coordinate
  | x_stop (int): x2 coordinate
  | y_start (int): y1 coordinate
  | y_stop (int): y2 coordinate
  | label (string): bond text annotation to specify linkage
  | dim (int): arbitrary dimention unit; necessary for drawsvg; inconsequential when drawing is exported as svg graphics
  | compact (bool): drawing style; normal or compact\n
  | Returns:
  | :-  
  | 
  """
  if compact == False:
    x_start = x_start * 2
    x_stop = x_stop * 2
  else:
    y_start = (y_start *0.5)* 1.2
    y_stop = (y_stop *0.5)* 1.2
    x_start = x_start * 1.2
    x_stop = x_stop * 1.2
  p = draw.Path(stroke_width=0.08*dim, stroke='black',)
  p.M(0-x_start*dim, 0+y_start*dim)
  p.L(0-x_stop*dim, 0+y_stop*dim)  
  d.append(p)
  d.append(draw.Text(label, dim*0.4, path=p, text_anchor='middle', valign='middle', lineOffset=-1))

def add_sugar(monosaccharide, x_pos = 0, y_pos = 0, modification = '', dim = 50, compact = False, conf = ''):
  """wrapper function for drawing monosaccharide at specified position\n
  | Arguments:
  | :-
  | monosaccharide (string): simplified IUPAC nomenclature
  | x_pos (int): x coordinate of icon on drawing canvas
  | y_pos (int): y coordinate of icon on drawing canvas
  | modification (string): icon text annotation; used for post-biosynthetic modifications
  | dim (int): arbitrary dimention unit; necessary for drawsvg; inconsequential when drawing is exported as svg graphics
  | compact (bool): drawing style; normal or compact\n
  | Returns:
  | :-  
  | 
  """
  if compact == False:
    x_pos = x_pos * 2
    y_pos = y_pos * 1
  else:
    x_pos = x_pos * 1.2
    y_pos = (y_pos * 0.5) *1.2
  if monosaccharide in list(sugar_dict.keys()):
    draw_shape(shape = sugar_dict[monosaccharide][0], color = sugar_dict[monosaccharide][1], x_pos = x_pos, y_pos = y_pos, modification = modification, conf = conf, furanose = sugar_dict[monosaccharide][2], dim = dim)
  else:
    p = draw.Path(stroke_width=0.04*dim, stroke = 'black')
    p.M(0-x_pos*dim-1/2*dim, 0+y_pos*dim+1/2*dim)
    p.L(0-x_pos*dim+1/2*dim, 0+y_pos*dim-1/2*dim)  
    d.append(p)
    p = draw.Path(stroke_width=0.04*dim, stroke = 'black')
    p.M(0-x_pos*dim+1/2*dim, 0+y_pos*dim+1/2*dim)
    p.L(0-x_pos*dim-1/2*dim, 0+y_pos*dim-1/2*dim)  
    d.append(p)

def multiple_branches(glycan):
  if ']' in glycan:
    tmp = glycan.replace('(', '*')
    tmp = tmp.replace(')', '*')
    tmp = tmp.replace('[', '(')
    tmp = tmp.replace(']', ')')

    open_close = []
    for openpos, closepos, level in matches(tmp):
      if level == 0 and bool(re.search('^Fuc\S{6}$', tmp[openpos:closepos])) == False:
        open_close.append((openpos, closepos))

    for k in range(len(open_close)-1):
      # print(open_close[k][1], open_close[k+1][0])
      if open_close[k+1][0] - open_close[k][1] == 2:
        branch1 = glycan[open_close[k][0]:open_close[k][1]]
        branch2 = glycan[open_close[k+1][0]:open_close[k+1][1]]
        # print(branch1[-2], branch2[-2])
        if branch1[-2] > branch2[-2]:
          glycan = glycan[:open_close[k][0]] + branch2 + '][' + branch1 + glycan[open_close[k+1][1]:]
  return glycan

def multiple_branch_branches(glycan):
  if ']' in glycan:
    tmp = glycan.replace('(', '*')
    tmp = tmp.replace(')', '*')
    tmp = tmp.replace('[', '(')
    tmp = tmp.replace(']', ')')
    for openpos, closepos, level in matches(tmp):
      if level == 0 and bool(re.search('^Fuc\S{6}$', tmp[openpos:closepos])) == False:
          glycan = glycan[:openpos] + multiple_branches(glycan[openpos:closepos]) + glycan[closepos:]
  return glycan

def reorder_for_drawing(glycan, by = 'linkage'):
  """order level 0 branches by linkage, ascending\n
  | Arguments:
  | :-
  | glycan (string): glycan in IUPAC-condensed format
  | Returns:
  | :-
  | Returns re-ordered glycan
  """
  if ']' in glycan:
    tmp = glycan.replace('(', '*')
    tmp = tmp.replace(')', '*')
    tmp = tmp.replace('[', '(')
    tmp = tmp.replace(']', ')')

    for openpos, closepos, level in matches(tmp):
      # if level == 0 and bool(re.search('^Fuc\S{6}$', tmp[openpos:closepos])) == False:
      if level == 0 and bool(re.search('^Fuc\S{6}$|^Xyl\S{6}$', tmp[openpos:closepos])) == False:

        # nested branches
        glycan = glycan[:openpos] + branch_order(glycan[openpos:closepos]) + glycan[closepos:]

        # branches
        group1 = glycan[:openpos-1]
        group2 = glycan[openpos:closepos]
        branch_end = [j[-2] for j in [group1, group2]]
        branch_end = [k.replace('z', '9') for k in branch_end]
        branch_len = [len(k) for k in min_process_glycans([group1, group2])]

        if by == 'length':
          
          if branch_len[0] == branch_len[1]:
            if branch_end[0] == branch_end[1]:
              glycan = group1 + '[' + group2 + ']' + glycan[closepos+1:]
            else:
              glycan = [group1, group2][np.argmin(branch_end)] + '[' + [group1, group2][np.argmax(branch_end)] + ']' + glycan[closepos+1:]
          else:
            glycan = [group1, group2][np.argmax(branch_len)] + '[' + [group1, group2][np.argmin(branch_len)] + ']' + glycan[closepos+1:]  

        elif by == 'linkage':

          if branch_end[0] == branch_end[1]:
            glycan = group1 + '[' + group2 + ']' + glycan[closepos+1:]
          else:
            glycan = [group1, group2][np.argmin(branch_end)] + '[' + [group1, group2][np.argmax(branch_end)] + ']' + glycan[closepos+1:]
  return glycan

def branch_order(glycan, by = 'linkage'):

  tmp = glycan.replace('(', '*')
  tmp = tmp.replace(')', '*')
  tmp = tmp.replace('[', '(')
  tmp = tmp.replace(']', ')')

  for openpos, closepos, level in matches(tmp):
    if level == 0 and bool(re.search('^Fuc\S{6}$|^Xyl\S{6}$', tmp[openpos:closepos])) == False:
      group1 = glycan[:openpos-1]
      group2 = glycan[openpos:closepos]
      branch_end = [j[-2] for j in [group1, group2]]
      branch_end = [k.replace('z', '9') for k in branch_end]
      branch_len = [len(k) for k in min_process_glycans([group1, group2])]

      if by == 'length':
        
        if branch_len[0] == branch_len[1]:
          if branch_end[0] == branch_end[1]:
            glycan = group1 + '[' + group2 + ']' + glycan[closepos+1:]
          else:
            glycan = [group1, group2][np.argmin(branch_end)] + '[' + [group1, group2][np.argmax(branch_end)] + ']' + glycan[closepos+1:]
        else:
          glycan = [group1, group2][np.argmax(branch_len)] + '[' + [group1, group2][np.argmin(branch_len)] + ']' + glycan[closepos+1:]  

      elif by == 'linkage':

        if branch_end[0] == branch_end[1]:
          glycan = group1 + '[' + group2 + ']' + glycan[closepos+1:]
        else:
          glycan = [group1, group2][np.argmin(branch_end)] + '[' + [group1, group2][np.argmax(branch_end)] + ']' + glycan[closepos+1:]
  return glycan

def split_node(G, node):
  """split graph at node\n
  | Arguments:
  | :-
  | G (networkx object): glycan graph
  | node (int): where to split the graph object
  | Returns:
  | :-
  | glycan graph (networkx object), consisting of two disjointed graphs\n
  ref: https://stackoverflow.com/questions/65853641/networkx-how-to-split-nodes-into-two-effectively-disconnecting-edges
"""
  edges = G.edges(node, data=True)
  
  new_edges = []
  new_nodes = []

  H = G.__class__()
  H.add_nodes_from(G.subgraph(node))
  
  for i, (s, t, data) in enumerate(edges):
      
      new_node = '{}_{}'.format(node, i)
      I = nx.relabel_nodes(H, {node:new_node})
      new_nodes += list(I.nodes(data=True))
      new_edges.append((new_node, t, data))
  
  G.remove_node(node)
  G.add_nodes_from(new_nodes)
  G.add_edges_from(new_edges)
  
  return G

def unique(sequence):
  """remove duplicates but keep order\n
  | Arguments:
  | :-
  | sequence (list): 
  | Returns:
  | :-
  | deduplicated list, preserving original order\n
  ref: https://stackoverflow.com/questions/480214/how-do-i-remove-duplicates-from-a-list-while-preserving-order
  """
  seen = set()
  return [x for x in sequence if not (x in seen or seen.add(x))]

def get_indices(x, y):
  """for each element in list x, get index (indices if present multiple times) in list y\n
  | Arguments:
  | :-
  | x (list):
  | y (list):
  | Returns:
  | :-
  | list of list of indices\n
  """
  res = [([idx for idx, val in enumerate(x) if val == sub] if sub in x else [None]) for sub in y]
  return res

def process_bonds(linkage_list):
  """prepare linkages for printing to figure\n
  | Arguments:
  | :-
  | linkage_list (list): list (or list of lists) of linkages
  | Returns:
  | :-
  | processed linkage_list\n
  """
  tmp = []
  bonds = []
  if any(isinstance(el, list) for el in linkage_list):
    for j in linkage_list:
      for k in j:
        if '?' in k:
          tmp.append('?')
        elif bool(re.compile("^a\d").match(k)):
          tmp.append('\u03B1 '+k[-1])
        elif bool(re.compile("^b\d").match(k)):
          tmp.append('\u03B2 '+k[-1])
        elif bool(re.compile("^\d-\d").match(k)):
          tmp.append(k[0]+' - '+k[-1])
      bonds.append(tmp)
      tmp = []
    return bonds
  else:
    for k in linkage_list:
      if '?' in k:
        bonds.append('?')
      elif bool(re.compile("^a\d").match(k)):
        bonds.append('\u03B1 '+k[-1])
      elif bool(re.compile("^b\d").match(k)):
        bonds.append('\u03B2 '+k[-1])
      elif bool(re.compile("^\d-\d").match(k)):
        bonds.append(k[0]+' - '+k[-1])
    return bonds

def split_monosaccharide_linkage(label_list):

  if any(isinstance(el, list) for el in label_list):
    sugar = [k[::2][::-1] for k in label_list]
    sugar_modification = [[get_modification(k) for k in y] for y in sugar]
    sugar_modification = [[multireplace(['O', '-ol'], '', k) for k in y] for y in sugar_modification]
    sugar = [[get_core(k) for k in y] for y in sugar]
    bond = [k[1::2][::-1] for k in label_list]
  else:
    sugar = label_list[::2][::-1]
    sugar_modification = [get_modification(k) if k != 'Re' else '' for k in sugar]
    sugar_modification = [multireplace(['O', '-ol'], '', k) for k in sugar_modification]
    sugar = [get_core(k) if k != 'Re' else 'Re' for k in sugar]
    bond = label_list[1::2][::-1]

  return sugar, sugar_modification, bond

def multireplace(list, replacement, string):
  for k in list:
    string = string.replace(k, replacement)
  return string

def get_coordinates_and_labels(draw_this, show_linkage = True):
  if bool(re.search('^\[', draw_this)) == False:
    draw_this = multiple_branch_branches(draw_this)
    draw_this = multiple_branches(draw_this)
    draw_this = reorder_for_drawing(draw_this)
    draw_this = multiple_branches(draw_this)
    draw_this = multiple_branch_branches(draw_this)

  graph = glycan_to_nxGraph(draw_this, libr = draw_lib)
  node_labels = nx.get_node_attributes(graph, 'string_labels')
  edges = graph.edges()
  branch_points = [e[1] for e in edges if abs(e[0]-e[1]) > 1]
  skeleton = [']'+str(k) if k in branch_points else str(k) for k in node_labels.keys()]
  
  for k in range(len(skeleton)):
    #multibranch situation on reducing end
    if skeleton[k] == skeleton[-1] and graph.degree()[k] == 3:
      idx = np.where(['[' in m for m in skeleton[:k]])[0][-1]
      skeleton[idx-1] = skeleton[idx-1] + ']'
    #note whether a multibranch situation exists
    if graph.degree()[k] == 4:
      idx = np.where(['[' in m for m in skeleton[:k]])[0][-1]
      if any(']]' in s for s in skeleton[idx:]):
        idx = np.where(['[' in m for m in skeleton[:k]])[0][-2]
      else:
        idx = np.where(['[' in m for m in skeleton[:k]])[0][-1]
      skeleton[idx-1] = skeleton[idx-1] + ']'
      #note whether a branch separates neighbors
    elif graph.degree()[k] > 2:
      skeleton[k] = ']' + skeleton[k]
      #note whether a branch starts
    elif graph.degree()[k] == 1 and k > 0:
      skeleton[k] = '[' + skeleton[k]
  # fix cases of extra brackets at reducing end monosaccharide
  if skeleton[-1] == '['+str(len(node_labels)-1):
      skeleton[-1] = str(len(node_labels)-1)

  glycan = '-'.join(skeleton)#[:-1]#+str(len(node_labels)-1)
  glycan = re.sub('(\([^\()]*)\(', r'\1)', glycan)
  glycan = glycan.replace('[)', ')[')
  glycan = glycan.replace('])', ')]')
  while ']]' in glycan:
    glycan = glycan.replace(']]', ']')
  while '[[' in glycan:
    glycan = glycan.replace('[[', '[')

  ## split main & branches, get labels
  branch_branch_branch_node = []
  branch_branch_node = []
  branch_node = []

  levels = [2, 1, 0]
  parts = [branch_branch_branch_node, branch_branch_node, branch_node]

  if ']' in glycan:
    
    glycan = glycan.replace('[', '(')
    glycan = glycan.replace(']', ')')

    for k in range(len(levels)):
      for openpos, closepos, level in matches(glycan):
        if level == levels[k]:
          parts[k].append(glycan[openpos:closepos])
          glycan = glycan[:openpos-1]+ len(glycan[openpos-1:closepos+1])*'*' + glycan[closepos+1:]
      glycan = glycan.replace('*', '')
      parts[k] = [ [i for i in k if i != ''] for k in [k.split('-') for k in parts[k]] ]

  main_node = glycan.split('-')
  main_node = [k for k in main_node if k != '' ]
  branch_branch_branch_node, branch_branch_node, branch_node = parts

  branch_label = []
  for k in range(len(branch_node)):
    branch_label.append([list(node_labels.values())[j] for j in range(len(list(graph.nodes()))) if list(graph.nodes())[j] in list(map(int, branch_node[k]))])

  branch_branch_label = []
  for k in range(len(branch_branch_node)):
    branch_branch_label.append([list(node_labels.values())[j] for j in range(len(list(graph.nodes()))) if list(graph.nodes())[j] in list(map(int, branch_branch_node[k]))])

  bbb_label = []
  for k in range(len(branch_branch_branch_node)):
    bbb_label.append([list(node_labels.values())[j] for j in range(len(list(graph.nodes()))) if list(graph.nodes())[j] in list(map(int, branch_branch_branch_node[k]))])

  main_label = [list(node_labels.values())[j] for j in range(len(list(graph.nodes()))) if list(graph.nodes())[j] in list(map(int, main_node))]

  ## split in monosaccharide & linkages
  main_sugar, main_sugar_modification, main_bond = split_monosaccharide_linkage(main_label)
  branch_sugar, branch_sugar_modification, branch_bond = split_monosaccharide_linkage(branch_label)
  branch_branch_sugar, branch_branch_sugar_modification, branch_branch_bond = split_monosaccharide_linkage(branch_branch_label)
  bbb_sugar, bbb_sugar_modification, bbb_bond = split_monosaccharide_linkage(bbb_label)

  ## process linkages
  main_bond = process_bonds(main_bond)
  branch_bond = process_bonds(branch_bond)
  branch_branch_bond = process_bonds(branch_branch_bond)
  bbb_bond = process_bonds(bbb_bond)

  if show_linkage == False:
    main_bond = ['' for x in main_bond]
    branch_bond = [['' for x in y] for y in branch_bond]
    branch_branch_bond = [['' for x in y] for y in branch_branch_bond]
    bbb_bond = [['' for x in y] for y in bbb_bond]

  ## get connectivity
  branch_connection = []
  for x in branch_node:
    branch_connection = branch_connection + [list(edges)[k][1] for k in range(len(list(edges))) if list(edges)[k][0] == int(x[-1])]
  branch_connection = unwrap(get_indices(main_node[::2][::-1], [str(k) for k in branch_connection]))

  branch_node_old = branch_node
  if '?' not in [k[0] for k in branch_bond]:
    branch_sugar = [branch_sugar[k] for k in list(np.argsort(branch_connection))[::-1]]
    branch_sugar_modification = [branch_sugar_modification[k] for k in list(np.argsort(branch_connection))[::-1]]
    branch_bond = [branch_bond[k] for k in list(np.argsort(branch_connection))[::-1]]
    branch_node = [branch_node[k] for k in list(np.argsort(branch_connection))[::-1]]
    branch_connection = [branch_connection[k] for k in list(np.argsort(branch_connection))[::-1]]

  branch_branch_connection = []
  for x in branch_branch_node:
    branch_branch_connection = branch_branch_connection + [list(edges)[k][1] for k in range(len(list(edges))) if list(edges)[k][0] == int(x[-1])]
  tmp = []
  for k in branch_branch_connection:
    tmp.append([(i, colour.index(str(k))) for i, colour in enumerate([k[::2][::-1] for k in branch_node]) if str(k) in colour])
  branch_branch_connection = unwrap(tmp)

  bbb_connection = []
  for x in branch_branch_branch_node:
    bbb_connection = bbb_connection + [list(edges)[k][1] for k in range(len(list(edges))) if list(edges)[k][0] == int(x[-1])]
  tmp = []
  for k in bbb_connection:
    tmp.append([(i, colour.index(str(k))) for i, colour in enumerate([k[::2][::-1] for k in branch_branch_node]) if str(k) in colour])
  bbb_connection = unwrap(tmp) 

  # order multiple connections on a branch level
  new_order = []
  for k in list(set(branch_connection))[::-1]:
    idx = unwrap(get_indices(branch_connection, [k]))
    if len(idx) == 1:
      new_order = new_order + idx
    else:
      new_order = new_order + [idx[i] for i in np.argsort([k[0][-1] for k in [j for j in [branch_bond[k] for k in idx]]])]

  branch_sugar = [branch_sugar[i] for i in new_order]
  branch_sugar_modification = [branch_sugar_modification[i] for i in new_order]
  branch_bond = [branch_bond[i] for i in new_order]
  branch_node = [branch_node[i] for i in new_order]
  branch_connection = [branch_connection[i] for i in new_order]

  for k in range(len(branch_branch_connection)):
    tmp = get_indices(new_order, [branch_branch_connection[k][0]])
    branch_branch_connection[k] = (tmp[0][0], branch_branch_connection[k][1])

  # order multiple connections on a branch_branch level
  new_order = []
  for k in list(set(branch_branch_connection))[::-1]:
    idx = unwrap(get_indices(branch_branch_connection, [k]))
    if len(idx) == 1:
      new_order = new_order + idx
    else:
      new_order = new_order + [idx[i] for i in np.argsort([k[0][-1] for k in [j for j in [branch_branch_bond[k] for k in idx]]])]

  branch_branch_sugar = [branch_branch_sugar[i] for i in new_order]
  branch_branch_sugar_modification = [branch_branch_sugar_modification[i] for i in new_order]
  branch_branch_bond = [branch_branch_bond[i] for i in new_order]
  branch_branch_node = [branch_branch_node[i] for i in new_order]
  branch_branch_connection = [branch_branch_connection[i] for i in new_order]

  ## main chain x y
  main_sugar_x_pos = [k for k in range(len(main_sugar))]
  main_sugar_y_pos = [0 for k in range(len(main_sugar))]

  if main_sugar[-1] in ['Fuc', 'Xyl'] and branch_sugar != []:
    if len(main_sugar) != 2:
      main_sugar_x_pos[-1] = main_sugar_x_pos[-1]-1
      main_sugar_y_pos[-1] = main_sugar_y_pos[-1]-2
    else:
      main_sugar_x_pos[-1] = main_sugar_x_pos[-1]-1
      main_sugar_y_pos[-1] = main_sugar_y_pos[-1]+2

  ## branch x
  branch_x_pos = []
  tmp = []
  for l in range(len(branch_sugar)):
    for k in range(len(branch_sugar[l])):
      tmp.append(main_sugar_x_pos[branch_connection[l]]+1+k)
    if branch_sugar[l][-1] not in ['Fuc', 'Xyl']: 
      branch_x_pos.append(tmp)
    else:
      tmp[-1] = tmp[-1]-1
      branch_x_pos.append(tmp)
    tmp = []

  ## branch branch x
  branch_branch_x_pos = []
  tmp = []
  for j in range(len(branch_branch_sugar)):
    for k in range(len(branch_branch_sugar[j])):
      tmp.append(branch_x_pos[branch_branch_connection[j][0]][branch_branch_connection[j][1]]+(k+1))
    if branch_branch_sugar[j][-1] not in ['Fuc', 'Xyl']: # is not in list of perpendicular monosaccharides. Add Xyl etc. 
      branch_branch_x_pos.append(tmp)
    else:
      tmp[-1] = tmp[-1]-1
      branch_branch_x_pos.append(tmp)
    tmp = []

  ## branch branch branch x
  bbb_x_pos = []
  tmp = []
  for j in range(len(bbb_sugar)):
    for k in range(len(bbb_sugar[j])):
      tmp.append(branch_branch_x_pos[bbb_connection[j][0]][bbb_connection[j][1]]+(k+1))
    if bbb_sugar[j][-1] not in ['Fuc', 'Xyl']: # is not in list of perpendicular monosaccharides. Add Xyl etc. 
      bbb_x_pos.append(tmp)
    else:
      tmp[-1] = tmp[-1]-1
      bbb_x_pos.append(tmp)
    tmp = []

  ## branch y
  branch_y_pos_zeros = [[0 for x in y] for y in branch_x_pos]
  branch_y_pos = branch_y_pos_zeros

  branch_branch_y_pos_zeros = [[0 for x in y] for y in branch_branch_x_pos]
  branch_branch_y_pos = branch_branch_y_pos_zeros

  bbb_y_pos_zeros = [[0 for x in y] for y in bbb_x_pos]
  bbb_y_pos = bbb_y_pos_zeros

  counter = 0
  for k in range(len(branch_x_pos)):
    # branch terminating in fucose
    if len(branch_sugar[k]) > 1 and branch_sugar[k][-1] in ['Fuc', 'Xyl']:
      tmp = [main_sugar_y_pos[branch_connection[k]]+2+counter for x in branch_x_pos[k]]
      tmp[-1] = tmp[-1]-2
      branch_y_pos[k] = tmp
      counter = counter+2
    # remaining branches longer than one sugar
    elif branch_sugar[k] not in [['Fuc'], ['Xyl']] and len(branch_sugar[k]) > 1:
      if main_sugar[-1] not in ['Fuc', 'Xyl']:
        branch_y_pos[k] = [main_sugar_y_pos[branch_connection[k]]+2+counter for x in branch_x_pos[k]]
        counter = counter+2
      elif len(main_sugar) - (branch_connection[k]+1) == 1:
        branch_y_pos[k] = [main_sugar_y_pos[branch_connection[k]]+counter for x in branch_x_pos[k]]
        # maybe?
        counter = counter+2
      else:
        branch_y_pos[k] = [main_sugar_y_pos[branch_connection[k]]+2+counter for x in branch_x_pos[k]]
        counter = counter+2
    # core fucose
    elif branch_sugar[k] in [['Fuc'], ['Xyl']] and branch_connection[k] == 0:
      if main_sugar_modification[0] != '' or branch_bond[k][0][-1] == '3':
        branch_y_pos[k] = [main_sugar_y_pos[branch_connection[k]]-2 for x in branch_x_pos[k]]
      else:
        branch_y_pos[k] = [main_sugar_y_pos[branch_connection[k]]+2 for x in branch_x_pos[k]]
    # one monosaccharide branches
    elif len(branch_sugar[k]) == 1 and  branch_sugar[k][0] not in ['Fuc', 'Xyl']: # ['Fuc']
      # branch_y_pos[k] = [main_sugar_y_pos[branch_connection[k]] for x in branch_x_pos[k]]
      if main_sugar[-1] not in ['Fuc', 'Xyl']:
        branch_y_pos[k] = [main_sugar_y_pos[branch_connection[k]]+2+counter for x in branch_x_pos[k]]
        counter = counter+2
      elif len(main_sugar) - (branch_connection[k]+1) == 1:
        branch_y_pos[k] = [main_sugar_y_pos[branch_connection[k]]+0+counter for x in branch_x_pos[k]]
        counter = counter+2
      else:
        branch_y_pos[k] = [main_sugar_y_pos[branch_connection[k]]+2+counter for x in branch_x_pos[k]]
        counter = counter+2
    # fucose not on core
    else:
      branch_y_pos[k] = [main_sugar_y_pos[branch_connection[k]]-2 for x in branch_x_pos[k]] 

  ## branch branch y
  counter = 0
  for k in range(len(branch_branch_x_pos)):
    # branch branch terminating in fucose
    if len(branch_branch_sugar[k]) > 1 and branch_branch_sugar[k][-1] in ['Fuc', 'Xyl']:
      tmp = [branch_y_pos[branch_branch_connection[k][0]][branch_branch_connection[k][1]]+2+counter for x in branch_branch_x_pos[k]]
      tmp[-1] = tmp[-1]-2
      branch_branch_y_pos[k] = tmp
    # elif branch_sugar[branch_branch_connection[k][0]][-1] == 'Fuc':#& branch_node[branch_branch_connection[k][0]][::2][::-1][-2] == branch_node[branch_branch_connection[k][0]][branch_branch_connection[k][1]]:
    elif branch_node[branch_branch_connection[k][0]][::2][::-1][-2] == branch_node[branch_branch_connection[k][0]][::2][::-1][branch_branch_connection[k][1]] and branch_sugar[branch_branch_connection[k][0]][-1] == 'Fuc':
      # print('True')
      branch_branch_y_pos[k] = [branch_y_pos[branch_branch_connection[k][0]][branch_branch_connection[k][1]] for x in branch_branch_x_pos[k]]
    #   counter = counter + 0
    elif len(branch_branch_sugar[k]) > 1 and branch_branch_sugar[k][-1] not in ['Fuc', 'Xyl']:
      branch_branch_y_pos[k] = [branch_y_pos[branch_branch_connection[k][0]][branch_branch_connection[k][1]]+2+counter for x in branch_branch_x_pos[k]]
      counter = counter + 2
    elif branch_branch_sugar[k] == ['GlcNAc']:
      branch_branch_y_pos[k] = [branch_y_pos[branch_branch_connection[k][0]][branch_branch_connection[k][1]]+2+counter for x in branch_branch_x_pos[k]]
    elif branch_branch_sugar[k] in [['Fuc'], ['Xyl']]:
      branch_branch_y_pos[k] = [branch_y_pos[branch_branch_connection[k][0]][branch_branch_connection[k][1]]-2+counter for x in branch_branch_x_pos[k]]
    elif min(branch_branch_x_pos[k]) > max(branch_x_pos[branch_branch_connection[k][0]]):
      branch_branch_y_pos[k] = [branch_y_pos[branch_branch_connection[k][0]][branch_branch_connection[k][1]]+0+counter for x in branch_branch_x_pos[k]]
    else:
      branch_branch_y_pos[k] = [branch_y_pos[branch_branch_connection[k][0]][branch_branch_connection[k][1]]+2+counter for x in branch_branch_x_pos[k]]

  ## branch branch branch y
  counter = 0
  for k in range(len(bbb_x_pos)):
    # branch branch terminating in fucose
    if len(bbb_sugar[k]) > 1 and bbb_sugar[k][-1] in ['Fuc', 'Xyl']:
      tmp = [branch_branch_y_pos[bbb_connection[k][0]][bbb_connection[k][1]]+2+counter for x in bbb_x_pos[k]]
      tmp[-1] = tmp[-1]-2
      bbb_y_pos[k] = tmp
    elif len(bbb_sugar[k]) > 1 and bbb_sugar[k][-1] not in ['Fuc', 'Xyl']:
      bbb_y_pos[k] = [branch_branch_y_pos[bbb_connection[k][0]][bbb_connection[k][1]]+2+counter for x in bbb_x_pos[k]]
    elif bbb_sugar[k] == ['GlcNAc']:
      bbb_y_pos[k] = [branch_branch_y_pos[bbb_connection[k][0]][bbb_connection[k][1]]+2+counter for x in bbb_x_pos[k]]
    elif bbb_sugar[k] in [['Fuc'], ['Xyl']]:
      bbb_y_pos[k] = [branch_branch_y_pos[bbb_connection[k][0]][bbb_connection[k][1]]-2+counter for x in bbb_x_pos[k]]
    elif min(bbb_x_pos[k]) > max(branch_branch_x_pos[bbb_connection[k][0]]):
      bbb_y_pos[k] = [branch_branch_y_pos[bbb_connection[k][0]][bbb_connection[k][1]]+0+counter for x in bbb_x_pos[k]]
    else:
      bbb_y_pos[k] = [branch_branch_y_pos[bbb_connection[k][0]][bbb_connection[k][1]]+2+counter for x in bbb_x_pos[k]]

  # y adjust spacing between branches
  splits = [main_node[::2][::-1][k] for k in [branch_connection[k] for k in unwrap(get_indices([k[::2][::-1] for k in branch_node], [k for k in [k[::2][::-1] for k in branch_node]]))]]
  tmp = unwrap(get_indices([k[::2][::-1] for k in branch_branch_node], [k for k in [k[::2][::-1] for k in branch_branch_node]]))
  for k in tmp:
    splits.append([j[::2][::-1] for j in branch_node][branch_branch_connection[k][0]][branch_branch_connection[k][1]])

  splits = unique(splits)
  filter = []

  for k in range(len(branch_sugar)):
    if branch_sugar[k] in [['Fuc'], ['Xyl']]:
    # if branch_sugar[k] == ['Xyl']:
      filter.append(main_node[::2][::-1][branch_connection[k]])
    if branch_sugar[k][-1] == 'Fuc' and len(branch_sugar[k]) > 1:
      filter.append(branch_node[k][::2][::-1][-2])    

  for k in range(len(branch_branch_sugar)):
    if branch_branch_sugar[k] in [['Fuc'], ['Xyl']]:
      filter.append([k[::2][::-1] for k in branch_node][branch_branch_connection[k][0]][branch_branch_connection[k][1]])

  splits = [k for k in splits if k not in filter]

  for n in splits:
    graph = glycan_to_nxGraph(draw_this, libr = draw_lib)
    graph2 = split_node(graph, int(n))

    edges = graph.edges()
    split_node_connections = [e[0] for e in edges if n+'_' in str(e[1])]
    node_crawl = [k for k in [list(nx.node_connected_component(graph2, k)) for k in split_node_connections] if int(main_node[-1]) not in k]
    new_node_crawl = []
    for k in node_crawl:
      new_node_crawl.append([x for x in k if '_' not in str(x)])

    tmp_a = main_node + unwrap(branch_node_old) + unwrap(branch_branch_node) + unwrap(branch_branch_branch_node)
    tmp_b = main_label + unwrap(branch_label) + unwrap(branch_branch_label) + unwrap(bbb_label)

    final_linkage = []
    for k in new_node_crawl:
      final_linkage.append(tmp_b[unwrap(get_indices(tmp_a, [str(k[-1])]))[0]])

    final_linkage = [k[-1] for k in final_linkage]
    new_node_crawl = [new_node_crawl[i] for i in np.argsort(final_linkage)]

    pairwise_node_crawl = list(zip(new_node_crawl, new_node_crawl[1:]))

    for pair in pairwise_node_crawl:

      idx_A = get_indices(main_node[::2][::-1]+unwrap([k[::2][::-1] for k in branch_node_old])+unwrap([k[::2][::-1] for k in branch_branch_node])+unwrap([k[::2][::-1] for k in branch_branch_branch_node]), [str(k) for k in pair[0]])
      idx_A = [k for k in idx_A if k != [None]]
      idx_B = get_indices(main_node[::2][::-1]+unwrap([k[::2][::-1] for k in branch_node_old])+unwrap([k[::2][::-1] for k in branch_branch_node])+unwrap([k[::2][::-1] for k in branch_branch_branch_node]), [str(k) for k in pair[1]])
      idx_B = [k for k in idx_B if k != [None]]

      y_list = main_sugar_y_pos+unwrap(branch_y_pos)+unwrap(branch_branch_y_pos)+unwrap(bbb_y_pos)
      x_list = main_sugar_x_pos+unwrap(branch_x_pos)+unwrap(branch_branch_x_pos)+unwrap(bbb_x_pos)

      if max([y_list[k[0]] for k in idx_A]) > max([y_list[k[0]] for k in idx_B]):
        upper_min = min([y_list[k[0]] for k in idx_A])
        lower_max = max([y_list[k[0]] for k in idx_B])
        # upper_y = [y_list[k[0]] for k in idx_A]
        # upper_x = [x_list[k[0]] for k in idx_A]
        # lower_y = [y_list[k[0]] for k in idx_B]
        # lower_x = [x_list[k[0]] for k in idx_B]
        upper = pair[0]
      else:
        upper_min = min([y_list[k[0]] for k in idx_B])
        lower_max = max([y_list[k[0]] for k in idx_A])
        # upper_y = [y_list[k[0]] for k in idx_B]
        # upper_x = [x_list[k[0]] for k in idx_B]
        # lower_y = [y_list[k[0]] for k in idx_A]
        # lower_x = [x_list[k[0]] for k in idx_A]
        upper = pair[1]

      to_add = 2 - (upper_min-lower_max)
      
      if main_sugar[-1] in ['Fuc', 'Xyl'] and len(main_sugar) == 2:
        pass
      else:
        for k in range(len(branch_y_pos)):
          for j in range(len(branch_y_pos[k])):
            if [k[::2][::-1] for k in branch_node][k][j] in [str(k) for k in upper] and branch_sugar[k] not in [['Fuc'], ['Xyl']]:
              branch_y_pos[k][j] = branch_y_pos[k][j] + to_add
              
      for k in range(len(branch_branch_y_pos)):
        for j in range(len(branch_branch_y_pos[k])):
          if [k[::2][::-1] for k in branch_branch_node][k][j] in [str(k) for k in upper]:
            branch_branch_y_pos[k][j] = branch_branch_y_pos[k][j] + to_add

      for k in range(len(bbb_y_pos)):
        for j in range(len(bbb_y_pos[k])):
          if [k[::2][::-1] for k in branch_branch_branch_node][k][j] in [str(k) for k in upper]:
            bbb_y_pos[k][j] = bbb_y_pos[k][j] + to_add
  
      # y adjust branch_branch connections
  for j in range(len(unique(branch_branch_connection))):
    if branch_branch_sugar[j] not in [['Fuc'], ['Xyl']] and max(branch_x_pos[branch_branch_connection[j][0]]) >= branch_branch_x_pos[j][0]:
      tmp = [branch_branch_y_pos[j][0] for j in unwrap(get_indices(branch_branch_connection, [unique(branch_branch_connection)[j]]))]
      y_adj = (max(tmp)-branch_y_pos[branch_branch_connection[j][0]][branch_branch_connection[j][1]+1])/2
      # for each branch
      for k in range(len(branch_x_pos)):
        # if connected
        if k == branch_branch_connection[j][0]:
          # and if smaller/equal x
          for n in range(len(branch_x_pos[k])):
            if branch_x_pos[k][n] <= branch_x_pos[branch_branch_connection[j][0]][branch_branch_connection[j][1]]:
              branch_y_pos[k][n] = branch_y_pos[k][n] + y_adj
      # for each branch branch
      for k in range(len(branch_branch_x_pos)):
        # if connected
        if branch_branch_connection[k][0] == branch_branch_connection[j][0] and branch_branch_connection[k][1] == branch_branch_connection[j][1]:
          # and if smaller/equal x
          for n in range(len(branch_branch_x_pos[k])):
            if branch_branch_x_pos[k][n] <= branch_x_pos[branch_branch_connection[j][0]][branch_branch_connection[j][1]]:
              branch_branch_y_pos[k][n] = branch_branch_y_pos[k][n] + y_adj

    # y adjust branch connections
  for k in range(len(unique(branch_connection))):
    tmp = [branch_y_pos[j][0] for j in unwrap(get_indices(branch_connection, [unique(branch_connection)[k]]))]
    if ['Fuc'] in [branch_sugar[j] for j in unwrap(get_indices(branch_connection, [unique(branch_connection)[k]]))] and branch_connection.count(unique(branch_connection)[k]) < 2 or ['Fuc'] in [branch_sugar[j] for j in unwrap(get_indices(branch_connection, [unique(branch_connection)[k]]))] and branch_connection.count(0) > 1:# and list(set(unwrap([branch_sugar[k] for k in unwrap(get_indices(unwrap(branch_sugar), ['Fuc']))]))) == ['Fuc']:
      y_adj = 0
    elif ['Xyl'] in [branch_sugar[j] for j in unwrap(get_indices(branch_connection, [unique(branch_connection)[k]]))] and branch_connection.count(unique(branch_connection)[k]) < 2:
      y_adj = 0
    else:
      y_adj = (max(tmp)-main_sugar_y_pos[unique(branch_connection)[k]])/2
    for j in range(len(main_sugar_x_pos)):
      if main_sugar_x_pos[j] <= main_sugar_x_pos[unique(branch_connection)[k]]:
        main_sugar_y_pos[j] = main_sugar_y_pos[j] + y_adj
      else:
        pass
    for j in range(len(branch_x_pos)):
      if branch_connection[j] == unique(branch_connection)[k] or branch_sugar[j] in [['Fuc'], ['Xyl']]:
        for n in range(len(branch_x_pos[j])):
          if branch_x_pos[j][n] <= main_sugar_x_pos[unique(branch_connection)[k]]:
            branch_y_pos[j][n] = branch_y_pos[j][n] + y_adj

  # fix for handling 'wrong' structures with the core fucose in the main chain
  if main_sugar[-1] in ['Fuc', 'Xyl'] and len(main_sugar) == 2 and branch_sugar != []:
    to_add = branch_y_pos[0][0] - main_sugar_y_pos[0]
    main_sugar_y_pos = [k + to_add for k in main_sugar_y_pos]

  ## fix spacing
  splits = [k for k in splits if k not in filter]
  for n in splits:
    graph = glycan_to_nxGraph(draw_this, libr = draw_lib)
    graph2 = split_node(graph, int(n))
    edges = graph.edges()
    split_node_connections = [e[0] for e in edges if n+'_' in str(e[1])]
    node_crawl = [k for k in [list(nx.node_connected_component(graph2, k)) for k in split_node_connections] if int(main_node[-1]) not in k]
    anti_node_crawl = [k for k in [list(nx.node_connected_component(graph2, k)) for k in split_node_connections] if int(main_node[-1]) in k]
    anti_node_crawl = [re.sub(r'_\S*$', '', str(k)) for k in unwrap(anti_node_crawl)]
    new_node_crawl = []
    for k in node_crawl:
      new_node_crawl.append([x for x in k if '_' not in str(x)])
    tmp_a = main_node + unwrap(branch_node_old) + unwrap(branch_branch_node) + unwrap(branch_branch_branch_node)
    tmp_b = main_label + unwrap(branch_label) + unwrap(branch_branch_label) + unwrap(bbb_label)
    final_linkage = []
    for k in new_node_crawl:
      final_linkage.append(tmp_b[unwrap(get_indices(tmp_a, [str(k[-1])]))[0]])
    final_linkage = [k[-1] for k in final_linkage]
    new_node_crawl = [new_node_crawl[i] for i in np.argsort(final_linkage)]
    pairwise_node_crawl = list(zip(new_node_crawl, new_node_crawl[1:]))

    for pair in pairwise_node_crawl:

      idx_A = get_indices(main_node[::2][::-1]+unwrap([k[::2][::-1] for k in branch_node_old])+unwrap([k[::2][::-1] for k in branch_branch_node])+unwrap([k[::2][::-1] for k in branch_branch_branch_node]), [str(k) for k in pair[0]])
      idx_A = [k for k in idx_A if k != [None]]
      idx_B = get_indices(main_node[::2][::-1]+unwrap([k[::2][::-1] for k in branch_node_old])+unwrap([k[::2][::-1] for k in branch_branch_node])+unwrap([k[::2][::-1] for k in branch_branch_branch_node]), [str(k) for k in pair[1]])
      idx_B = [k for k in idx_B if k != [None]]
      
      y_list = main_sugar_y_pos+unwrap(branch_y_pos)+unwrap(branch_branch_y_pos)+unwrap(bbb_y_pos)
      x_list = main_sugar_x_pos+unwrap(branch_x_pos)+unwrap(branch_branch_x_pos)+unwrap(bbb_x_pos)

      if max([y_list[k[0]] for k in idx_A]) > max([y_list[k[0]] for k in idx_B]):
        upper_min = min([y_list[k[0]] for k in idx_A])
        lower_max = max([y_list[k[0]] for k in idx_B])
        upper_y = [y_list[k[0]] for k in idx_A]
        upper_x = [x_list[k[0]] for k in idx_A]
        lower_y = [y_list[k[0]] for k in idx_B]
        lower_x = [x_list[k[0]] for k in idx_B]
        upper = pair[0]
      else:
        upper_min = min([y_list[k[0]] for k in idx_B])
        lower_max = max([y_list[k[0]] for k in idx_A])
        upper_y = [y_list[k[0]] for k in idx_B]
        upper_x = [x_list[k[0]] for k in idx_B]
        lower_y = [y_list[k[0]] for k in idx_A]
        lower_x = [x_list[k[0]] for k in idx_A]
        upper = pair[1]

      diff_to_fix = []
      for x_cor in list(set(upper_x)):
        if x_cor in list(set(lower_x)):
          min_y_upper = min([upper_y[k] for k in unwrap(get_indices(upper_x, [x_cor]))])
          max_y_lower = max([lower_y[k] for k in unwrap(get_indices(lower_x, [x_cor]))])
          diff_to_fix.append(2 - (min_y_upper-max_y_lower))
      
      if diff_to_fix != []:
        to_add = max(diff_to_fix)

      if main_sugar[-1] == 'Fuc':# and len(main_sugar) == 2:
        pass
      else:
        for k in range(len(branch_y_pos)):
          for j in range(len(branch_y_pos[k])):
            if [k[::2][::-1] for k in branch_node][k][j] in [str(k) for k in upper]: #and branch_sugar[k] != ['Fuc']:
              branch_y_pos[k][j] = branch_y_pos[k][j] + to_add
            if branch_x_pos[k][j] == 0:
                branch_y_pos[k][j] = branch_y_pos[k][j] + (to_add/2)

      if main_sugar[-1] == 'Fuc':# and len(main_sugar) == 2:
        pass
      else:
        for k in range(len(main_sugar)):
          if main_sugar_x_pos[k] < min([x for x in unwrap(branch_x_pos) if x > 0]):
            main_sugar_y_pos[k] = main_sugar_y_pos[k] + (to_add/2)
      
      for k in range(len(branch_branch_y_pos)):
        #if branch_branch_sugar[k] != ['Fuc']:
        for j in range(len(branch_branch_y_pos[k])):
          if [k[::2][::-1] for k in branch_branch_node][k][j] in [str(k) for k in upper]:
            branch_branch_y_pos[k][j] = branch_branch_y_pos[k][j] + to_add
            # print(to_add)

      for k in range(len(bbb_y_pos)):
        #if branch_branch_sugar[k] != ['Fuc']:
        for j in range(len(bbb_y_pos[k])):
          if [k[::2][::-1] for k in branch_branch_branch_node][k][j] in [str(k) for k in upper]:
            bbb_y_pos[k][j] = bbb_y_pos[k][j] + to_add

  main_conf = [k.group()[0] if k != None else '' for k in [re.search('^L-|^D-', k) for k in main_sugar_modification]]
  main_sugar_modification = [re.sub('^L-|^D-', '', k) for k in main_sugar_modification]

  b_conf = [[k.group()[0] if k != None else '' for k in j] for j in [[re.search('^L-|^D-', k) for k in j] for j in branch_sugar_modification]]
  branch_sugar_modification = [[re.sub('^L-|^D-', '', k) for k in j] for j in branch_sugar_modification]

  bb_conf = [[k.group()[0] if k != None else '' for k in j] for j in [[re.search('^L-|^D-', k) for k in j] for j in branch_branch_sugar_modification]]
  branch_branch_sugar_modification = [[re.sub('^L-|^D-', '', k) for k in j] for j in branch_branch_sugar_modification]

  bbb_conf = [[k.group()[0] if k != None else '' for k in j] for j in [[re.search('^L-|^D-', k) for k in j] for j in bbb_sugar_modification]]
  bbb_sugar_modification = [[re.sub('^L-|^D-', '', k) for k in j] for j in bbb_sugar_modification]

  data_combined = [
      [main_sugar, main_sugar_x_pos, main_sugar_y_pos, main_sugar_modification, main_bond, main_conf],
      [branch_sugar, branch_x_pos, branch_y_pos, branch_sugar_modification, branch_bond, branch_connection, b_conf],
      [branch_branch_sugar, branch_branch_x_pos, branch_branch_y_pos, branch_branch_sugar_modification, branch_branch_bond, branch_branch_connection, bb_conf],
      [bbb_sugar, bbb_x_pos, bbb_y_pos, bbb_sugar_modification, bbb_bond, bbb_connection, bbb_conf]
  ]

  return data_combined

def draw_bracket(x, y_min_max, direction = 'right', dim = 50):
  # vertial
  p = draw.Path(stroke_width=0.04*dim, stroke = 'black')
  p.M(0-(x*dim), 0+(y_min_max[1]*dim)+0.75*dim)
  p.L(0-(x*dim), 0+(y_min_max[0]*dim)-0.75*dim) 
  d.append(p)
  
  if direction == 'right':
    # upper 
    p = draw.Path(stroke_width=0.04*dim, stroke = 'black')
    p.M(0-(x*dim)-(0.02*dim),          0+(y_min_max[1]*dim)+0.75*dim)
    p.L(0-(x*dim)+0.25*dim, 0+(y_min_max[1]*dim)+0.75*dim) 
    d.append(p)
    #lower
    p = draw.Path(stroke_width=0.04*dim, stroke = 'black')
    p.M(0-(x*dim)-(0.02*dim), 0+(y_min_max[0]*dim)-0.75*dim)
    p.L(0-(x*dim)+0.25*dim, 0+(y_min_max[0]*dim)-0.75*dim) 
    d.append(p)
  elif direction == "left":
    # upper 
    p = draw.Path(stroke_width=0.04*dim, stroke = 'black')
    p.M(0-(x*dim)+(0.02*dim), 0+(y_min_max[1]*dim)+0.75*dim)
    p.L(0-(x*dim)-0.25*dim, 0+(y_min_max[1]*dim)+0.75*dim) 
    d.append(p)
    #lower
    p = draw.Path(stroke_width=0.04*dim, stroke = 'black')
    p.M(0-(x*dim)+(0.02*dim), 0+(y_min_max[0]*dim)-0.75*dim)
    p.L(0-(x*dim)-0.25*dim, 0+(y_min_max[0]*dim)-0.75*dim) 
    d.append(p)

def GlycoDraw(draw_this, compact = False, show_linkage = True, dim = 50, output = None):

  if compact == True:
    show_linkage = False

  # handle floaty bits if present
  floaty_bits = []
  for openpos, closepos, level in matches(draw_this, opendelim='{', closedelim='}'):
      floaty_bits.append(draw_this[openpos:closepos]+'Re')
      draw_this = draw_this[:openpos-1]+ len(draw_this[openpos-1:closepos+1])*'*' + draw_this[closepos+1:]
  draw_this = draw_this.replace('*', '')

  try:
    data = get_coordinates_and_labels(draw_this, show_linkage = show_linkage)
  except:
    try:
      draw_this = motif_list.loc[motif_list.motif_name == draw_this].motif.values.tolist()[0]
      data = get_coordinates_and_labels(draw_this, show_linkage = show_linkage)
    except:
      return print('Error: did you enter a real glycan or motif?')
      sys.exit(1)

  main_sugar, main_sugar_x_pos, main_sugar_y_pos, main_sugar_modification, main_bond, main_conf = data[0]
  branch_sugar, branch_x_pos, branch_y_pos, branch_sugar_modification, branch_bond, branch_connection, b_conf = data[1]
  branch_branch_sugar, branch_branch_x_pos, branch_branch_y_pos, branch_branch_sugar_modification, branch_branch_bond, branch_branch_connection, bb_conf  = data[2]
  bbb_sugar, bbb_x_pos, bbb_y_pos, bbb_sugar_modification, bbb_bond, bbb_connection, bbb_conf  = data[3]

    ## adjust drawing dimentions
  max_y = max(unwrap(bbb_y_pos)+unwrap(branch_branch_y_pos)+unwrap(branch_y_pos)+main_sugar_y_pos)
  min_y = min(unwrap(bbb_y_pos)+unwrap(branch_branch_y_pos)+unwrap(branch_y_pos)+main_sugar_y_pos)
  max_x = max(unwrap(bbb_x_pos)+unwrap(branch_branch_x_pos)+unwrap(branch_x_pos)+main_sugar_x_pos)
  min_x = min(unwrap(bbb_x_pos)+unwrap(branch_branch_x_pos)+unwrap(branch_x_pos)+main_sugar_x_pos)

  ## canvas size
  width = ((((max_x+1)*2)-1)*dim)+dim
  if floaty_bits != []:
    width = width + (max([len(k) for k in min_process_glycans(floaty_bits)], default = 0)+1) * dim

  if len(floaty_bits) > len(list(set(floaty_bits))):
     width = width + dim
  height = ((((max(abs(min_y), max_y)+1)*2)-1)*dim)+10+50
  x_ori = -width+(dim/2)+0.5*dim
  y_ori = (-height/2)+(((max_y-abs(min_y))/2)*dim)

  global d

  ## draw
  d = draw.Drawing(width, height, origin=(x_ori,y_ori), displayInline=True)

  # bond main chain
  [add_bond(main_sugar_x_pos[k+1], main_sugar_x_pos[k], main_sugar_y_pos[k+1], main_sugar_y_pos[k], main_bond[k], dim = dim, compact = compact) for k in range(len(main_sugar)-1)]
  # bond branch
  [add_bond(branch_x_pos[b_idx][s_idx+1], branch_x_pos[b_idx][s_idx], branch_y_pos[b_idx][s_idx+1], branch_y_pos[b_idx][s_idx], branch_bond[b_idx][s_idx+1], dim = dim, compact = compact) for b_idx in range(len(branch_sugar)) for s_idx in range(len(branch_sugar[b_idx])-1) if len(branch_sugar[b_idx]) > 1]
  # bond branch to main chain
  [add_bond(branch_x_pos[k][0], main_sugar_x_pos[branch_connection[k]], branch_y_pos[k][0], main_sugar_y_pos[branch_connection[k]], branch_bond[k][0], dim = dim, compact = compact) for k in range(len(branch_sugar))]
  # bond branch branch
  [add_bond(branch_branch_x_pos[b_idx][s_idx+1], branch_branch_x_pos[b_idx][s_idx], branch_branch_y_pos[b_idx][s_idx+1], branch_branch_y_pos[b_idx][s_idx], branch_branch_bond[b_idx][::-1][s_idx], dim = dim, compact = compact) for b_idx in range(len(branch_branch_sugar)) for s_idx in range(len(branch_branch_sugar[b_idx])-1) if len(branch_branch_sugar[b_idx]) > 1]
  # bond branch branch branch
  [add_bond(branch_branch_x_pos[b_idx][s_idx+1], branch_branch_x_pos[b_idx][s_idx], branch_branch_y_pos[b_idx][s_idx+1], branch_branch_y_pos[b_idx][s_idx], branch_branch_bond[b_idx][::-1][s_idx], dim = dim, compact = compact) for b_idx in range(len(branch_branch_sugar)) for s_idx in range(len(branch_branch_sugar[b_idx])-1) if len(branch_branch_sugar[b_idx]) > 1]
  # bond branch_branch to branch
  [add_bond(branch_branch_x_pos[k][0], branch_x_pos[branch_branch_connection[k][0]][branch_branch_connection[k][1]], branch_branch_y_pos[k][0], branch_y_pos[branch_branch_connection[k][0]][branch_branch_connection[k][1]], branch_branch_bond[k][0], dim = dim, compact = compact) for k in range(len(branch_branch_sugar))]
  # bond branch_branch_branch to branch_branch
  [add_bond(bbb_x_pos[k][0], branch_branch_x_pos[bbb_connection[k][0]][bbb_connection[k][1]], bbb_y_pos[k][0], branch_branch_y_pos[bbb_connection[k][0]][bbb_connection[k][1]], bbb_bond[k][0], dim = dim, compact = compact) for k in range(len(bbb_sugar))]
  # sugar main chain
  [add_sugar(main_sugar[k], main_sugar_x_pos[k], main_sugar_y_pos[k], modification = main_sugar_modification[k], conf = main_conf[k], compact = compact, dim = dim) for k in range(len(main_sugar))]
  # sugar branch
  [add_sugar(branch_sugar[b_idx][s_idx], branch_x_pos[b_idx][s_idx], branch_y_pos[b_idx][s_idx], modification = branch_sugar_modification[b_idx][s_idx], conf = b_conf[b_idx][s_idx], compact = compact, dim = dim) for b_idx in range(len(branch_sugar)) for s_idx in range(len(branch_sugar[b_idx]))]
  # sugar branch_branch 
  [add_sugar(branch_branch_sugar[b_idx][s_idx], branch_branch_x_pos[b_idx][s_idx], branch_branch_y_pos[b_idx][s_idx], modification = branch_branch_sugar_modification[b_idx][s_idx], conf = bb_conf[b_idx][s_idx], compact = compact, dim = dim) for b_idx in range(len(branch_branch_sugar)) for s_idx in range(len(branch_branch_sugar[b_idx]))]
  # sugar branch branch branch
  [add_sugar(bbb_sugar[b_idx][s_idx], bbb_x_pos[b_idx][s_idx], bbb_y_pos[b_idx][s_idx], modification = bbb_sugar_modification[b_idx][s_idx], conf = bbb_conf[b_idx][s_idx], compact = compact, dim = dim) for b_idx in range(len(bbb_sugar)) for s_idx in range(len(bbb_sugar[b_idx]))]

  if floaty_bits != []:
    fb_count = {i:floaty_bits.count(i) for i in floaty_bits}
    floaty_bits = list(set(floaty_bits))
    floaty_data = [get_coordinates_and_labels(floaty_bits[k], show_linkage = show_linkage) for k in range(len(floaty_bits))]
    y_span = max_y-min_y
    n_floats = len(floaty_bits)
    floaty_span = n_floats * 2 - 2
    y_diff = (floaty_span/2) - (y_span/2)

    for j in range(len(floaty_data)):
      floaty_sugar, floaty_sugar_x_pos, floaty_sugar_y_pos, floaty_sugar_modification, floaty_bond, floaty_conf = floaty_data[j][0]
      floaty_sugar_x_pos = [floaty_sugar_x_pos[k] + max_x + 1 for k in floaty_sugar_x_pos]
      floaty_sugar_y_pos = [floaty_sugar_y_pos[k] + 2 * j - y_diff for k in floaty_sugar_y_pos]
      [add_bond(floaty_sugar_x_pos[k+1], floaty_sugar_x_pos[k], floaty_sugar_y_pos[k+1], floaty_sugar_y_pos[k], floaty_bond[k], dim = dim, compact = compact) for k in range(len(floaty_sugar)-1)]
      [add_sugar(floaty_sugar[k], floaty_sugar_x_pos[k], floaty_sugar_y_pos[k], modification = floaty_sugar_modification[k], conf=floaty_conf, compact = compact, dim = dim) for k in range(len(floaty_sugar))]

      if fb_count[floaty_bits[j]] > 1:
        if compact == False:
          add_sugar('Re', max(floaty_sugar_x_pos)+0.5, floaty_sugar_y_pos[-1]-0.75, modification = str(fb_count[floaty_bits[j]]) + 'x', compact=compact, dim=dim )
        else:
          add_sugar('Re', max(floaty_sugar_x_pos)+0.75, floaty_sugar_y_pos[-1]-1.2, modification = str(fb_count[floaty_bits[j]]) + 'x', compact=compact, dim=dim )

    if compact == False:
      draw_bracket(max_x*2+1, (min_y, max_y), direction = 'right')
    elif compact == True:
      draw_bracket(max_x*1.2+1, ((min_y *0.5)* 1.2, (max_y *0.5)* 1.2), direction = 'right')

  fit_to_contents(d, pad=10)
  
  if output is not None:
    data = d.asSvg()
    data = re.sub(r'<text font-size="17.5" ', r'<text font-size="17.5" font-family="century gothic" font-weight="bold" ', data)
    data = re.sub(r'<text font-size="20.0" ', r'<text font-size="20" font-family="century gothic" ', data)
    data = re.sub(r'<text font-size="15.0" ', r'<text font-size="17.5" font-family="century gothic" font-style="italic" ', data)
    
    if 'svg' in output:
      with open(output, 'w') as f:
        f.write(data)
    
    elif 'pdf' in output:
      cairosvg.svg2pdf(bytestring=data, write_to=output)

  return d
