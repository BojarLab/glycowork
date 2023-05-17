from glycowork.glycan_data.loader import lib, unwrap, motif_list, multireplace
from glycowork.motif.graph import glycan_to_nxGraph
from glycowork.motif.tokenization import get_core, get_modification
from glycowork.motif.processing import expand_lib, min_process_glycans

import cairo, cairosvg, cairocffi
import drawsvg as draw
import networkx as nx
import numpy as np
import sys
import re
from math import sin, cos, radians, sqrt, atan, degrees

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
            

# adjusted SNFG color palette
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

# extensions for draw_lib
additions = ['-', 'blank', 'red_end', 'free', 
             '04X', '15A', '02A', '13X',
             '24X', '35X', '04A', '15X',
             '02X', '13A', '24A', '35A',
             '25A', '03A', '14X', '25X',
             '03X', '14A', 'Z', 'Y',
             'B', 'C', 'text', 'non_glycan']

# shape-color mapping
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
  "non_glycan": ['Assigned', 'black', False],

  "blank": ['empty', snfg_white, False],
  "text": ['text', None, None],
  "red_end" : ['red_end', None, None],
  "free" : ['free', None, None],
  "04X" : ['04X', None, None],
  "15A" : ['15A', None, None],
  "02A" : ['02A', None, None],
  "13X" : ['13X', None, None],
  "24X" : ['24X', None, None],
  "35X" : ['35X', None, None],
  "04A" : ['04A', None, None],
  "15X" : ['15X', None, None],
  "02X" : ['02X', None, None],
  "13A" : ['13A', None, None],
  "24A" : ['24A', None, None],
  "35A" : ['35A', None, None],
  "25A" : ['25A', None, None],
  "03A" : ['03A', None, None],
  "14X" : ['14X', None, None],
  "25X" : ['25X', None, None],
  "03X" : ['03X', None, None],
  "14A" : ['14A', None, None],
  "Z" : ['Z', None, None],
  "Y" : ['Y', None, None],
  "B" : ['B', None, None],
  "C" : ['C', None, None]
}   

# build draw_lib with glycoword additions
draw_lib = expand_lib(lib, ['-'] + list(sugar_dict.keys()))

def hex_circumference(x_pos, y_pos, dim):
  """Draw a hexagoncircumference at the specified position and dimensions.\n
  | Arguments:
  | :-
  | x_pos (int): X coordinate of the hexagon's center on the drawing canvas.
  | y_pos (int): Y coordinate of the hexagon's center on the drawing canvas.
  | dim (int): Arbitrary dimension unit used for scaling the hexagon's size.\n
  | Returns:
  | :-
  | None
  """  
  p = draw.Path(stroke_width=0.04*dim, stroke='black')
  p.M((0-x_pos*dim)+0.5*dim, (0+y_pos*dim)+0)
  p.L((0-x_pos*dim)+(0.5*dim)*cos(radians(60)),(0+y_pos*dim)+(0.5*dim)*sin(radians(60)))  
  d.append(p)
  p = draw.Path(stroke_width=0.04*dim, stroke='black')
  p.M((0-x_pos*dim)+(0.5*dim)*cos(radians(60)),(0+y_pos*dim)+(0.5*dim)*sin(radians(60)))  
  p.L((0-x_pos*dim)+(0.5*dim)*cos(radians(120)),(0+y_pos*dim)+(0.5*dim)*sin(radians(120)))  
  d.append(p)
  p = draw.Path(stroke_width=0.04*dim, stroke='black')
  p.M((0-x_pos*dim)+(0.5*dim)*cos(radians(120)),(0+y_pos*dim)+(0.5*dim)*sin(radians(120)))  
  p.L((0-x_pos*dim)-0.5*dim, (0+y_pos*dim)+0)
  d.append(p)
  p = draw.Path(stroke_width=0.04*dim, stroke='black')
  p.M((0-x_pos*dim)-0.5*dim, (0+y_pos*dim)+0)
  p.L((0-x_pos*dim)+(0.5*dim)*cos(radians(240)),(0+y_pos*dim)+(0.5*dim)*sin(radians(240)))  
  d.append(p)
  p = draw.Path(stroke_width=0.04*dim, stroke='black')
  p.M((0-x_pos*dim)+(0.5*dim)*cos(radians(240)),(0+y_pos*dim)+(0.5*dim)*sin(radians(240)))  
  p.L((0-x_pos*dim)+(0.5*dim)*cos(radians(300)),(0+y_pos*dim)+(0.5*dim)*sin(radians(300)))  
  d.append(p)
  p = draw.Path(stroke_width=0.04*dim, stroke='black')
  p.M((0-x_pos*dim)+(0.5*dim)*cos(radians(300)),(0+y_pos*dim)+(0.5*dim)*sin(radians(300)))  
  p.L((0-x_pos*dim)+0.5*dim, (0+y_pos*dim)+0)
  d.append(p)

def hex(x_pos, y_pos, dim, color = 'white'):
  """Draw a hexagon shape at the specified position and dimensions.\n
  | Arguments:
  | :-
  | x_pos (int): X coordinate of the hexagon's center on the drawing canvas.
  | y_pos (int): Y coordinate of the hexagon's center on the drawing canvas.
  | dim (int): Arbitrary dimension unit used for scaling the hexagon's size.
  | color (str): Color of the hexagon. Default is 'white'.\n
  | Returns:
  | :-
  | None
  """  
  d.append(draw.Lines(  (0-x_pos*dim)+0.5*dim,                                          (0+y_pos*dim)+0,
                        (0-x_pos*dim)+(0.5*dim)*cos(radians(300)),                      (0+y_pos*dim)+(0.5*dim)*sin(radians(300)),
                        (0-x_pos*dim)+(0.5*dim)*cos(radians(240)),                      (0+y_pos*dim)+(0.5*dim)*sin(radians(240)),
                        (0-x_pos*dim)-0.5*dim,                                          (0+y_pos*dim)+0,
                        (0-x_pos*dim)+(0.5*dim)*cos(radians(120)),                      (0+y_pos*dim)+(0.5*dim)*sin(radians(120)),
                        (0-x_pos*dim)+(0.5*dim)*cos(radians(60)),                       (0+y_pos*dim)+(0.5*dim)*sin(radians(60)),
                        close=True,
                        fill= color,
                        stroke='black', stroke_width = 0.04*dim))

def draw_shape(shape, color, x_pos, y_pos, modification = '', dim = 50, furanose = False, conf = '', deg = 0, text_anchor = 'middle'): 
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
  inside_hex_dim = ((sqrt(3))/2)*dim

  if shape == 'Hex':
    # xexose - circle
    d.append(draw.Circle(0-x_pos*dim, 0+y_pos*dim, dim/2, fill=color, stroke_width=0.04*dim, stroke='black'))
    # text annotation
    p = draw.Path(stroke_width=0)
    p.M(0-x_pos*dim-dim, 0+y_pos*dim+0.5*dim)
    p.L(0-x_pos*dim+dim, 0+y_pos*dim+0.5*dim)  
    d.append(p)
    d.append(draw.Text(modification, dim*0.35, path=p, text_anchor='middle', line_offset=-3.15))
    if furanose == True:
      p = draw.Path(stroke_width=0)
      p.M(0-x_pos*dim-dim, 0+y_pos*dim)
      p.L(0-x_pos*dim+dim, 0+y_pos*dim)  
      d.append(p)
      d.append(draw.Text('f', dim*0.30, path=p, text_anchor='middle', center=True))
    # ring configuration
    p = draw.Path(stroke_width=0)
    p.M(0-x_pos*dim-dim, 0+y_pos*dim)
    p.L(0-x_pos*dim+dim, 0+y_pos*dim)  
    d.append(p)
    d.append(draw.Text(conf, dim*0.30, path=p, text_anchor='middle', center=True))

  if shape == 'HexNAc':
    # hexnac - square
    d.append(draw.Rectangle((0-x_pos*dim)-(dim/2),(0+y_pos*dim)-(dim/2),dim,dim, fill=color, stroke_width=0.04*dim, stroke = 'black'))
    # text annotation
    p = draw.Path(stroke_width=0)
    p.M(0-x_pos*dim-dim, 0+y_pos*dim+0.5*dim)
    p.L(0-x_pos*dim+dim, 0+y_pos*dim+0.5*dim)  
    d.append(p)
    d.append(draw.Text(modification, dim*0.35, path=p, text_anchor='middle', line_offset=-3.15))
    if furanose == True:
      p = draw.Path(stroke_width=0)
      p.M(0-x_pos*dim-dim, 0+y_pos*dim)
      p.L(0-x_pos*dim+dim, 0+y_pos*dim)  
      d.append(p)
      d.append(draw.Text('f', dim*0.30, path=p, text_anchor='middle', center=True))
    # ring configuration
    p = draw.Path(stroke_width=0)
    p.M(0-x_pos*dim-dim, 0+y_pos*dim)
    p.L(0-x_pos*dim+dim, 0+y_pos*dim)  
    d.append(p)
    d.append(draw.Text(conf, dim*0.30, path=p, text_anchor='middle', center=True))    
  
  if shape == 'HexN':
    # hexosamine - crossed square
    d.append(draw.Rectangle((0-x_pos*dim)-(dim/2),(0+y_pos*dim)-(dim/2),dim,dim, fill='white', stroke_width=0.04*dim, stroke = 'black'))
    d.append(draw.Lines((0-x_pos*dim)-(dim/2), (0+y_pos*dim)-(dim/2),
                        (0-x_pos*dim)+(dim/2), (0+y_pos*dim)-(dim/2),
                        (0-x_pos*dim)+(dim/2), (0+y_pos*dim)+(dim/2),
                        (0-x_pos*dim)-(dim/2), (0+y_pos*dim)-(dim/2),
            close=True,
            fill=color,
            stroke='black', stroke_width = 0))
    p = draw.Path(stroke_width=0.04*dim, stroke='black',)
    p.M((0-x_pos*dim)-(dim/2), (0+y_pos*dim)-(dim/2))
    p.L((0-x_pos*dim)+(dim/2), (0+y_pos*dim)-(dim/2))  
    d.append(p)
    p = draw.Path(stroke_width=0.04*dim, stroke='black',)
    p.M((0-x_pos*dim)+(dim/2), (0+y_pos*dim)-(dim/2))
    p.L((0-x_pos*dim)+(dim/2), (0+y_pos*dim)+(dim/2))  
    d.append(p)
    p = draw.Path(stroke_width=0.04*dim, stroke='black',)
    p.M((0-x_pos*dim)+(dim/2), (0+y_pos*dim)+(dim/2))
    p.L((0-x_pos*dim)-(dim/2), (0+y_pos*dim)-(dim/2))  
    d.append(p)
    # text annotation
    p = draw.Path(stroke_width=0)
    p.M(0-x_pos*dim-dim, 0+y_pos*dim+0.5*dim)
    p.L(0-x_pos*dim+dim, 0+y_pos*dim+0.5*dim)  
    d.append(p)
    d.append(draw.Text(modification, dim*0.35, path=p, text_anchor='middle', line_offset=-3.15))
    # ring configuration
    p = draw.Path(stroke_width=0)
    p.M(0-x_pos*dim-dim, 0+y_pos*dim)
    p.L(0-x_pos*dim+dim, 0+y_pos*dim)  
    d.append(p)
    d.append(draw.Text(conf, dim*0.30, path=p, text_anchor='middle', center=True))    
  
  if shape == 'HexA_2':
    # hexuronate - divided diamond
    # AltA / IdoA
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
    d.append(draw.Text(modification, dim*0.35, path=p, text_anchor='middle', line_offset=-3.15))
    # ring configuration
    p = draw.Path(stroke_width=0)
    p.M(0-x_pos*dim-dim, 0+y_pos*dim)
    p.L(0-x_pos*dim+dim, 0+y_pos*dim)  
    d.append(p)
    d.append(draw.Text(conf, dim*0.30, path=p, text_anchor='middle', center=True))    
  
  if shape == 'HexA':
    # hexuronate - divided diamond (colors flipped)
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
    d.append(draw.Text(modification, dim*0.35, path=p, text_anchor='middle', line_offset=-3.15))
    # ring configuration
    p = draw.Path(stroke_width=0)
    p.M(0-x_pos*dim-dim, 0+y_pos*dim)
    p.L(0-x_pos*dim+dim, 0+y_pos*dim)  
    d.append(p)
    d.append(draw.Text(conf, dim*0.30, path=p, text_anchor='middle', center=True))    
  
  if shape == 'dHex':
    # deoxyhexose - triangle
    d.append(draw.Lines((0-x_pos*dim)-(0.5*dim), (0+y_pos*dim)+(((3**0.5)/2)*dim*0.5), #-(dim*1/3)
                    (0-x_pos*dim)+(dim/2)-(0.5*dim), (0+y_pos*dim)-(((3**0.5)/2)*dim)+(((3**0.5)/2)*dim*0.5),
                    (0-x_pos*dim)+(dim)-(0.5*dim), (0+y_pos*dim)+(((3**0.5)/2)*dim*0.5),
            close=True,
            fill=color,
            stroke='black', stroke_width = 0.04*dim))
    # text annotation
    p = draw.Path(stroke_width=0)
    p.M(0-x_pos*dim-dim, 0+y_pos*dim+0.5*dim)
    p.L(0-x_pos*dim+dim, 0+y_pos*dim+0.5*dim)  
    d.append(p)
    d.append(draw.Text(modification, dim*0.35, path=p, text_anchor='middle', line_offset=-3.15))
    if furanose == True:
      p = draw.Path(stroke_width=0)
      p.M(0-x_pos*dim-dim, 0+y_pos*dim+0.05*dim)
      p.L(0-x_pos*dim+dim, 0+y_pos*dim+0.05*dim)  
      d.append(p)
      d.append(draw.Text('f', dim*0.30, path=p, text_anchor='middle', center=True))
    # ring configuration
    p = draw.Path(stroke_width=0)
    p.M(0-x_pos*dim-dim, 0+y_pos*dim)
    p.L(0-x_pos*dim+dim, 0+y_pos*dim)  
    d.append(p)
    d.append(draw.Text(conf, dim*0.30, path=p, text_anchor='middle', center=True))    
  
  if shape == 'dHexNAc':
    # deoxyhexnac - divided triangle
    d.append(draw.Lines((0-x_pos*dim)-(0.5*dim), (0+y_pos*dim)+(((3**0.5)/2)*dim*0.5), #-(dim*1/3) for center of triangle
                    (0-x_pos*dim)+(dim/2)-(0.5*dim), (0+y_pos*dim)-(((3**0.5)/2)*dim)+(((3**0.5)/2)*dim*0.5), # -(dim*1/3) for bottom alignment
                    (0-x_pos*dim)+(dim)-(0.5*dim), (0+y_pos*dim)+(((3**0.5)/2)*dim*0.5), # -(((3**0.5)/2)*dim*0.5) for half of triangle height
            close=True,
            fill='white',
            stroke='black', stroke_width = 0.04*dim))
    d.append(draw.Lines((0-x_pos*dim), (0+y_pos*dim)+(((3**0.5)/2)*dim*0.5), #-(dim*1/3)
                    (0-x_pos*dim)+(dim/2)-(0.5*dim), (0+y_pos*dim)-(((3**0.5)/2)*dim)+(((3**0.5)/2)*dim*0.5),
                    (0-x_pos*dim)+(dim)-(0.5*dim), (0+y_pos*dim)+(((3**0.5)/2)*dim*0.5),
            close=True,
            fill=color,
            stroke='black', stroke_width = 0))
    p = draw.Path(stroke_width=0.04*dim, stroke='black',)
    p.M((0-x_pos*dim), (0+y_pos*dim)+(((3**0.5)/2)*dim*0.5))
    p.L((0-x_pos*dim)+(dim/2)-(0.5*dim), (0+y_pos*dim)-(((3**0.5)/2)*dim)+(((3**0.5)/2)*dim*0.5))  
    d.append(p)
    p = draw.Path(stroke_width=0.04*dim, stroke='black',)
    p.M((0-x_pos*dim)+(dim/2)-(0.5*dim), (0+y_pos*dim)-(((3**0.5)/2)*dim)+(((3**0.5)/2)*dim*0.5))
    p.L((0-x_pos*dim)+(dim)-(0.5*dim), (0+y_pos*dim)+(((3**0.5)/2)*dim*0.5))  
    d.append(p)
    p = draw.Path(stroke_width=0.04*dim, stroke='black',)
    p.M((0-x_pos*dim)+(dim)-(0.5*dim), (0+y_pos*dim)+(((3**0.5)/2)*dim*0.5))
    p.L((0-x_pos*dim), (0+y_pos*dim)+(((3**0.5)/2)*dim*0.5))  
    d.append(p)
    # text annotation
    p = draw.Path(stroke_width=0)
    p.M(0-x_pos*dim-dim, 0+y_pos*dim+0.5*dim)
    p.L(0-x_pos*dim+dim, 0+y_pos*dim+0.5*dim)  
    d.append(p)
    d.append(draw.Text(modification, dim*0.35, path=p, text_anchor='middle', line_offset=-3.15))
    # ring configuration
    p = draw.Path(stroke_width=0)
    p.M(0-x_pos*dim-dim, 0+y_pos*dim)
    p.L(0-x_pos*dim+dim, 0+y_pos*dim)  
    d.append(p)
    d.append(draw.Text(conf, dim*0.30, path=p, text_anchor='middle', center=True))    
  
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
    d.append(draw.Text(modification, dim*0.35, path=p, text_anchor='middle', line_offset=-3.15))
    # ring configuration
    p = draw.Path(stroke_width=0)
    p.M(0-x_pos*dim-dim, 0+y_pos*dim)
    p.L(0-x_pos*dim+dim, 0+y_pos*dim)  
    d.append(p)
    d.append(draw.Text(conf, dim*0.30, path=p, text_anchor='middle', center=True))    
  
  if shape == 'Pen':
    # pentose - star
    d.append(draw.Lines((0-x_pos*dim)+0 ,         (0+y_pos*dim)-(0.5*dim)/cos(radians(18)),
                    (0-x_pos*dim)+((0.25*dim)/cos(radians(18)))*cos(radians(54)) ,         (0+y_pos*dim)-((0.25*dim)/cos(radians(18)))*sin(radians(54)),
                    (0-x_pos*dim)+((0.5*dim)/cos(radians(18)))*cos(radians(18)) ,         (0+y_pos*dim)-((0.5*dim)/cos(radians(18)))*sin(radians(18)),
                    (0-x_pos*dim)+((0.25*dim)/cos(radians(18)))*cos(radians(18)) ,         (0+y_pos*dim)+((0.25*dim)/cos(radians(18)))*sin(radians(18)),
                    (0-x_pos*dim)+((0.5*dim)/cos(radians(18)))*cos(radians(54)) ,         (0+y_pos*dim)+((0.5*dim)/cos(radians(18)))*sin(radians(54)),
                    (0-x_pos*dim)+0 ,         (0+y_pos*dim)+(0.25*dim)/cos(radians(18)),
                    (0-x_pos*dim)-((0.5*dim)/cos(radians(18)))*cos(radians(54)) ,         (0+y_pos*dim)+((0.5*dim)/cos(radians(18)))*sin(radians(54)),
                    (0-x_pos*dim)-((0.25*dim)/cos(radians(18)))*cos(radians(18)) ,         (0+y_pos*dim)+((0.25*dim)/cos(radians(18)))*sin(radians(18)),
                    (0-x_pos*dim)+-((0.5*dim)/cos(radians(18)))*cos(radians(18)) ,         (0+y_pos*dim)-((0.5*dim)/cos(radians(18)))*sin(radians(18)),
                    (0-x_pos*dim)-((0.25*dim)/cos(radians(18)))*cos(radians(54)) ,         (0+y_pos*dim)-((0.25*dim)/cos(radians(18)))*sin(radians(54)),
                        close=True,
                        fill= color,
                        stroke='black', stroke_width = 0.04*dim))
    # text annotation
    p = draw.Path(stroke_width=0)
    p.M(0-x_pos*dim-dim, 0+y_pos*dim+0.5*dim)
    p.L(0-x_pos*dim+dim, 0+y_pos*dim+0.5*dim)  
    d.append(p)
    d.append(draw.Text(modification, dim*0.35, path=p, text_anchor='middle', line_offset=-3.15))
    if furanose == True:
      p = draw.Path(stroke_width=0)
      p.M(0-x_pos*dim-dim, 0+y_pos*dim)
      p.L(0-x_pos*dim+dim, 0+y_pos*dim)  
      d.append(p)
      d.append(draw.Text('f', dim*0.30, path=p, text_anchor='middle', center=True))
    # ring configuration
    p = draw.Path(stroke_width=0)
    p.M(0-x_pos*dim-dim, 0+y_pos*dim)
    p.L(0-x_pos*dim+dim, 0+y_pos*dim)  
    d.append(p)
    d.append(draw.Text(conf, dim*0.30, path=p, text_anchor='middle', center=True))    
  
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
    d.append(draw.Text(modification, dim*0.35, path=p, text_anchor='middle', line_offset=-3.15))
    # ring configuration
    p = draw.Path(stroke_width=0)
    p.M(0-x_pos*dim-dim, 0+y_pos*dim)
    p.L(0-x_pos*dim+dim, 0+y_pos*dim)  
    d.append(p)
    d.append(draw.Text(conf, dim*0.30, path=p, text_anchor='middle', center=True))    
  
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
    d.append(draw.Text(modification, dim*0.35, path=p, text_anchor='middle', line_offset=-3.15))
    # ring configuration
    p = draw.Path(stroke_width=0)
    p.M(0-x_pos*dim-dim, 0+y_pos*dim)
    p.L(0-x_pos*dim+dim, 0+y_pos*dim)  
    d.append(p)
    d.append(draw.Text(conf, dim*0.30, path=p, text_anchor='middle', center=True))    
   
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
    d.append(draw.Text(modification, dim*0.35, path=p, text_anchor='middle', line_offset=-3.15))
    # ring configuration
    p = draw.Path(stroke_width=0)
    p.M(0-x_pos*dim-dim, 0+y_pos*dim)
    p.L(0-x_pos*dim+dim, 0+y_pos*dim)  
    d.append(p)
    d.append(draw.Text(conf, dim*0.30, path=p, text_anchor='middle', center=True))    
  
  if shape == 'Assigned':
    # assigned - pentagon
    d.append(draw.Lines((0-x_pos*dim)+0 ,         (0+y_pos*dim)-(0.5*dim)/cos(radians(18)),
                        (0-x_pos*dim)+((0.5*dim)/cos(radians(18)))*cos(radians(18)) ,         (0+y_pos*dim)-((0.5*dim)/cos(radians(18)))*sin(radians(18)),
                        (0-x_pos*dim)+((0.5*dim)/cos(radians(18)))*cos(radians(54)) ,         (0+y_pos*dim)+((0.5*dim)/cos(radians(18)))*sin(radians(54)),
                        (0-x_pos*dim)-((0.5*dim)/cos(radians(18)))*cos(radians(54)) ,         (0+y_pos*dim)+((0.5*dim)/cos(radians(18)))*sin(radians(54)),
                        (0-x_pos*dim)+-((0.5*dim)/cos(radians(18)))*cos(radians(18)) ,         (0+y_pos*dim)-((0.5*dim)/cos(radians(18)))*sin(radians(18)),
                        close=True,
                        fill= color,
                        stroke='black', stroke_width = 0.04*dim))
    # text annotation
    p = draw.Path(stroke_width=0)
    p.M(0-x_pos*dim-dim, 0+y_pos*dim+0.5*dim)
    p.L(0-x_pos*dim+dim, 0+y_pos*dim+0.5*dim)  
    d.append(p)
    d.append(draw.Text(modification, dim*0.35, path=p, text_anchor='middle', line_offset=-3.15))
    # ring configuration
    p = draw.Path(stroke_width=0)
    p.M(0-x_pos*dim-dim, 0+y_pos*dim)
    p.L(0-x_pos*dim+dim, 0+y_pos*dim)  
    d.append(p)
    d.append(draw.Text(conf, dim*0.30, path=p, text_anchor='middle', center=True))    
  
  if shape == 'empty':
    d.append(draw.Circle(0-x_pos*dim, 0+y_pos*dim, dim/2, fill='none', stroke_width=0.04*dim, stroke='none'))
    # text annotation
    p = draw.Path(stroke_width=0)
    p.M(0-x_pos*dim-dim, 0+y_pos*dim+0.5*dim)
    p.L(0-x_pos*dim+dim, 0+y_pos*dim+0.5*dim)  
    d.append(p)
    d.append(draw.Text(modification, dim*0.35, path=p, text_anchor=text_anchor, line_offset=-3.15))

  if shape == 'text':
    d.append(draw.Text(modification, dim*0.35, 0-x_pos*dim, 0+y_pos*dim, text_anchor=text_anchor))
  
  if shape == 'red_end':
    p = draw.Path(stroke_width=0.04*dim, stroke='black', fill = 'none')
    p.M(((0-x_pos*dim)+0.1*dim), ((0+y_pos*dim)-0.4*dim))  # Start path at point (-10, 20)
    p.C(((0-x_pos*dim)-0.3*dim), ((0+y_pos*dim)-0.1*dim),
        ((0-x_pos*dim)+0.3*dim), ((0+y_pos*dim)+0.1*dim),
        ((0-x_pos*dim)-0.1*dim), ((0+y_pos*dim)+0.4*dim))
    d.append(p)
    d.append(draw.Circle((0-x_pos*dim), 0+y_pos*dim, 0.15*dim, fill='white', stroke_width=0.04*dim, stroke='black'))

  if shape == 'free':
    p = draw.Path(stroke_width=0.04*dim, stroke='black', fill = 'none')
    p.M(((0-x_pos*dim)+0.1*dim), ((0+y_pos*dim)-0.4*dim))  # Start path at point (-10, 20)
    p.C(((0-x_pos*dim)-0.3*dim), ((0+y_pos*dim)-0.1*dim),
        ((0-x_pos*dim)+0.3*dim), ((0+y_pos*dim)+0.1*dim),
        ((0-x_pos*dim)-0.1*dim), ((0+y_pos*dim)+0.4*dim))
    d.append(p)

  if shape == '04X':
    hex(x_pos, y_pos, dim) 
    d.append(draw.Lines(    (0-x_pos*dim)+0,                                                (0+y_pos*dim)-0,
                            (0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(30)),            (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(30)),
                            (0-x_pos*dim)+(0.5*dim)*cos(radians(60)),                       (0+y_pos*dim)-(0.5*dim)*sin(radians(60)),
                            (0-x_pos*dim)+(0.5*dim)*cos(radians(120)),                      (0+y_pos*dim)-(0.5*dim)*sin(radians(120)),
                            (0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(150)),           (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(150)),
                            close=True,
                            fill= 'grey',
                            stroke='black', stroke_width = 0))
    hex_circumference(x_pos, y_pos, dim)
    p = draw.Path(stroke_width=0.04*dim, stroke='black')
    p.M((0-x_pos*dim)+0,                                                (0+y_pos*dim)-0)
    p.L((0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(30)),            (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(30)))  
    d.append(p)
    p = draw.Path(stroke_width=0.04*dim, stroke='black')
    p.M((0-x_pos*dim)+0,                                                (0+y_pos*dim)-0)
    p.L((0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(150)),           (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(150)))  
    d.append(p)

  if shape == '15A':
    hex(x_pos, y_pos, dim)  
    d.append(draw.Lines(    (0-x_pos*dim)+0,                                                (0+y_pos*dim)-0,
                            (0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(90)),            (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(90)),
                            (0-x_pos*dim)+(0.5*dim)*cos(radians(60)),                       (0+y_pos*dim)-(0.5*dim)*sin(radians(60)),
                            (0-x_pos*dim)+(0.5*dim)*cos(radians(0)),                      (0+y_pos*dim)-(0.5*dim)*sin(radians(0)),
                            (0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(330)),           (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(330)),
                            close=True,
                            fill= 'grey',
                            stroke='black', stroke_width = 0))
    hex_circumference(x_pos, y_pos, dim)
    p = draw.Path(stroke_width=0.04*dim, stroke='black')
    p.M((0-x_pos*dim)+0,                                                (0+y_pos*dim)-0)
    p.L((0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(90)),            (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(90)))  
    d.append(p)
    p = draw.Path(stroke_width=0.04*dim, stroke='black')
    p.M((0-x_pos*dim)+0,                                                (0+y_pos*dim)-0)
    p.L((0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(330)),           (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(330)))  
    d.append(p)

  if shape == '02A':
    hex(x_pos, y_pos, dim) 
    d.append(draw.Lines(    (0-x_pos*dim)+0,                                                (0+y_pos*dim)-0,
                            (0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(30)),            (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(30)),
                            (0-x_pos*dim)+(0.5*dim)*cos(radians(0)),                       (0+y_pos*dim)-(0.5*dim)*sin(radians(0)),
                            (0-x_pos*dim)+(0.5*dim)*cos(radians(300)),                        (0+y_pos*dim)-(0.5*dim)*sin(radians(300)),
                            (0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(270)),           (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(270)),
                            close=True,
                            fill= 'grey',
                            stroke='black', stroke_width = 0))
    hex_circumference(x_pos, y_pos, dim)
    p = draw.Path(stroke_width=0.04*dim, stroke='black')
    p.M((0-x_pos*dim)+0,                                                (0+y_pos*dim)-0)
    p.L((0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(30)),            (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(30)))  
    d.append(p)
    p = draw.Path(stroke_width=0.04*dim, stroke='black')
    p.M((0-x_pos*dim)+0,                                                (0+y_pos*dim)-0)
    p.L((0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(270)),           (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(270)))  
    d.append(p)

  if shape == '13X':
    hex(x_pos, y_pos, dim) 
    d.append(draw.Lines(    (0-x_pos*dim)+0,                                                (0+y_pos*dim)-0,
                            (0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(330)),            (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(330)),
                            (0-x_pos*dim)+(0.5*dim)*cos(radians(300)),                       (0+y_pos*dim)-(0.5*dim)*sin(radians(300)),
                            (0-x_pos*dim)+(0.5*dim)*cos(radians(240)),                        (0+y_pos*dim)-(0.5*dim)*sin(radians(240)),
                            (0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(210)),           (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(210)),
                            close=True,
                            fill= 'grey',
                            stroke='black', stroke_width = 0))
    hex_circumference(x_pos, y_pos, dim)
    p = draw.Path(stroke_width=0.04*dim, stroke='black')
    p.M((0-x_pos*dim)+0,                                                (0+y_pos*dim)-0)
    p.L((0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(330)),            (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(330)))  
    d.append(p)
    p = draw.Path(stroke_width=0.04*dim, stroke='black')
    p.M((0-x_pos*dim)+0,                                                (0+y_pos*dim)-0)
    p.L((0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(210)),           (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(210)))  
    d.append(p) 

  if shape == '24X':
    hex(x_pos, y_pos, dim) 
    d.append(draw.Lines(    (0-x_pos*dim)+0,                                                (0+y_pos*dim)-0,
                            (0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(270)),            (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(270)),
                            (0-x_pos*dim)+(0.5*dim)*cos(radians(240)),                       (0+y_pos*dim)-(0.5*dim)*sin(radians(240)),
                            (0-x_pos*dim)+(0.5*dim)*cos(radians(180)),                        (0+y_pos*dim)-(0.5*dim)*sin(radians(180)),
                            (0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(150)),           (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(150)),
                            close=True,
                            fill= 'grey',
                            stroke='black', stroke_width = 0))
    hex_circumference(x_pos, y_pos, dim)
    p = draw.Path(stroke_width=0.04*dim, stroke='black')
    p.M((0-x_pos*dim)+0,                                                (0+y_pos*dim)-0)
    p.L((0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(270)),            (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(270)))  
    d.append(p)
    p = draw.Path(stroke_width=0.04*dim, stroke='black')
    p.M((0-x_pos*dim)+0,                                                (0+y_pos*dim)-0)
    p.L((0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(150)),           (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(150)))  
    d.append(p)

  if shape == '35X':
    hex(x_pos, y_pos, dim) 
    d.append(draw.Lines(    (0-x_pos*dim)+0,                                                (0+y_pos*dim)-0,
                            (0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(210)),            (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(210)),
                            (0-x_pos*dim)+(0.5*dim)*cos(radians(180)),                       (0+y_pos*dim)-(0.5*dim)*sin(radians(180)),
                            (0-x_pos*dim)+(0.5*dim)*cos(radians(120)),                        (0+y_pos*dim)-(0.5*dim)*sin(radians(120)),
                            (0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(90)),           (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(90)),
                            close=True,
                            fill= 'grey',
                            stroke='black', stroke_width = 0))
    hex_circumference(x_pos, y_pos, dim)
    p = draw.Path(stroke_width=0.04*dim, stroke='black')
    p.M((0-x_pos*dim)+0,                                                (0+y_pos*dim)-0)
    p.L((0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(210)),            (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(210)))  
    d.append(p)
    p = draw.Path(stroke_width=0.04*dim, stroke='black')
    p.M((0-x_pos*dim)+0,                                                (0+y_pos*dim)-0)
    p.L((0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(90)),            (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(90)))  
    d.append(p)

  if shape == '04A':
    hex(x_pos, y_pos, dim, color = 'grey') 
    d.append(draw.Lines(    (0-x_pos*dim)+0,                                                (0+y_pos*dim)-0,
                            (0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(30)),            (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(30)),
                            (0-x_pos*dim)+(0.5*dim)*cos(radians(60)),                       (0+y_pos*dim)-(0.5*dim)*sin(radians(60)),
                            (0-x_pos*dim)+(0.5*dim)*cos(radians(120)),                      (0+y_pos*dim)-(0.5*dim)*sin(radians(120)),
                            (0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(150)),           (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(150)),
                            close=True,
                            fill= 'white',
                            stroke='black', stroke_width = 0))
    hex_circumference(x_pos, y_pos, dim)
    p = draw.Path(stroke_width=0.04*dim, stroke='black')
    p.M((0-x_pos*dim)+0,                                                (0+y_pos*dim)-0)
    p.L((0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(30)),            (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(30)))  
    d.append(p)
    p = draw.Path(stroke_width=0.04*dim, stroke='black')
    p.M((0-x_pos*dim)+0,                                                (0+y_pos*dim)-0)
    p.L((0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(150)),           (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(150)))  
    d.append(p)

  if shape == '15X':
    hex(x_pos, y_pos, dim, color = 'grey')  
    d.append(draw.Lines(    (0-x_pos*dim)+0,                                                (0+y_pos*dim)-0,
                            (0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(90)),            (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(90)),
                            (0-x_pos*dim)+(0.5*dim)*cos(radians(60)),                       (0+y_pos*dim)-(0.5*dim)*sin(radians(60)),
                            (0-x_pos*dim)+(0.5*dim)*cos(radians(0)),                      (0+y_pos*dim)-(0.5*dim)*sin(radians(0)),
                            (0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(330)),           (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(330)),
                            close=True,
                            fill= 'white',
                            stroke='black', stroke_width = 0))
    hex_circumference(x_pos, y_pos, dim)
    p = draw.Path(stroke_width=0.04*dim, stroke='black')
    p.M((0-x_pos*dim)+0,                                                (0+y_pos*dim)-0)
    p.L((0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(90)),            (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(90)))  
    d.append(p)
    p = draw.Path(stroke_width=0.04*dim, stroke='black')
    p.M((0-x_pos*dim)+0,                                                (0+y_pos*dim)-0)
    p.L((0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(330)),           (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(330)))  
    d.append(p)

  if shape == '02X':
    hex(x_pos, y_pos, dim, color = 'grey') 
    d.append(draw.Lines(    (0-x_pos*dim)+0,                                                (0+y_pos*dim)-0,
                            (0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(30)),            (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(30)),
                            (0-x_pos*dim)+(0.5*dim)*cos(radians(0)),                       (0+y_pos*dim)-(0.5*dim)*sin(radians(0)),
                            (0-x_pos*dim)+(0.5*dim)*cos(radians(300)),                        (0+y_pos*dim)-(0.5*dim)*sin(radians(300)),
                            (0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(270)),           (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(270)),
                            close=True,
                            fill= 'white',
                            stroke='black', stroke_width = 0))
    hex_circumference(x_pos, y_pos, dim)
    p = draw.Path(stroke_width=0.04*dim, stroke='black')
    p.M((0-x_pos*dim)+0,                                                (0+y_pos*dim)-0)
    p.L((0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(30)),            (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(30)))  
    d.append(p)
    p = draw.Path(stroke_width=0.04*dim, stroke='black')
    p.M((0-x_pos*dim)+0,                                                (0+y_pos*dim)-0)
    p.L((0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(270)),           (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(270)))  
    d.append(p)

  if shape == '13A':
    hex(x_pos, y_pos, dim, color = 'grey') 
    d.append(draw.Lines(    (0-x_pos*dim)+0,                                                (0+y_pos*dim)-0,
                            (0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(330)),            (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(330)),
                            (0-x_pos*dim)+(0.5*dim)*cos(radians(300)),                       (0+y_pos*dim)-(0.5*dim)*sin(radians(300)),
                            (0-x_pos*dim)+(0.5*dim)*cos(radians(240)),                        (0+y_pos*dim)-(0.5*dim)*sin(radians(240)),
                            (0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(210)),           (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(210)),
                            close=True,
                            fill= 'white',
                            stroke='black', stroke_width = 0))
    hex_circumference(x_pos, y_pos, dim)
    p = draw.Path(stroke_width=0.04*dim, stroke='black')
    p.M((0-x_pos*dim)+0,                                                (0+y_pos*dim)-0)
    p.L((0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(330)),            (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(330)))  
    d.append(p)
    p = draw.Path(stroke_width=0.04*dim, stroke='black')
    p.M((0-x_pos*dim)+0,                                                (0+y_pos*dim)-0)
    p.L((0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(210)),           (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(210)))  
    d.append(p)

  if shape == '24A':
    hex(x_pos, y_pos, dim, color = 'grey') 
    d.append(draw.Lines(    (0-x_pos*dim)+0,                                                (0+y_pos*dim)-0,
                            (0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(270)),            (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(270)),
                            (0-x_pos*dim)+(0.5*dim)*cos(radians(240)),                       (0+y_pos*dim)-(0.5*dim)*sin(radians(240)),
                            (0-x_pos*dim)+(0.5*dim)*cos(radians(180)),                        (0+y_pos*dim)-(0.5*dim)*sin(radians(180)),
                            (0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(150)),           (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(150)),
                            close=True,
                            fill= 'white',
                            stroke='black', stroke_width = 0))
    hex_circumference(x_pos, y_pos, dim)
    p = draw.Path(stroke_width=0.04*dim, stroke='black')
    p.M((0-x_pos*dim)+0,                                                (0+y_pos*dim)-0)
    p.L((0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(270)),            (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(270)))  
    d.append(p)
    p = draw.Path(stroke_width=0.04*dim, stroke='black')
    p.M((0-x_pos*dim)+0,                                                (0+y_pos*dim)-0)
    p.L((0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(150)),           (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(150)))  
    d.append(p)

  if shape == '35A':
    hex(x_pos, y_pos, dim, color = 'grey') 
    d.append(draw.Lines(    (0-x_pos*dim)+0,                                                (0+y_pos*dim)-0,
                            (0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(210)),            (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(210)),
                            (0-x_pos*dim)+(0.5*dim)*cos(radians(180)),                       (0+y_pos*dim)-(0.5*dim)*sin(radians(180)),
                            (0-x_pos*dim)+(0.5*dim)*cos(radians(120)),                        (0+y_pos*dim)-(0.5*dim)*sin(radians(120)),
                            (0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(90)),           (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(90)),
                            close=True,
                            fill= 'white',
                            stroke='black', stroke_width = 0))
    hex_circumference(x_pos, y_pos, dim)
    p = draw.Path(stroke_width=0.04*dim, stroke='black')
    p.M((0-x_pos*dim)+0,                                                (0+y_pos*dim)-0)
    p.L((0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(210)),            (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(210)))  
    d.append(p)
    p = draw.Path(stroke_width=0.04*dim, stroke='black')
    p.M((0-x_pos*dim)+0,                                                (0+y_pos*dim)-0)
    p.L((0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(90)),            (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(90)))  
    d.append(p) 
  
  if shape == '25A':
    hex(x_pos, y_pos, dim) 
    d.append(draw.Lines(    (0-x_pos*dim)+0,                                                (0+y_pos*dim)-0,
                            (0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(90)),            (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(90)),
                            (0-x_pos*dim)+(0.5*dim)*cos(radians(60)),                       (0+y_pos*dim)-(0.5*dim)*sin(radians(60)),
                            (0-x_pos*dim)+(0.5*dim)*cos(radians(0)),                        (0+y_pos*dim)-(0.5*dim)*sin(radians(0)),
                            (0-x_pos*dim)+(0.5*dim)*cos(radians(300)),                      (0+y_pos*dim)-(0.5*dim)*sin(radians(300)),
                            (0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(270)),            (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(270)),
                            close=True,
                            fill= 'grey',
                            stroke='black', stroke_width = 0))
    hex_circumference(x_pos, y_pos, dim)
    p = draw.Path(stroke_width=0.04*dim, stroke='black')
    p.M((0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(90)),            (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(90)))
    p.L((0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(270)),            (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(270)))  
    d.append(p)

  if shape == '03A':
    hex(x_pos, y_pos, dim) 
    d.append(draw.Lines(    (0-x_pos*dim)+0,                                                (0+y_pos*dim)-0,
                            (0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(30)),            (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(30)),
                            (0-x_pos*dim)+(0.5*dim)*cos(radians(0)),                       (0+y_pos*dim)-(0.5*dim)*sin(radians(0)),
                            (0-x_pos*dim)+(0.5*dim)*cos(radians(300)),                        (0+y_pos*dim)-(0.5*dim)*sin(radians(300)),
                            (0-x_pos*dim)+(0.5*dim)*cos(radians(240)),                      (0+y_pos*dim)-(0.5*dim)*sin(radians(240)),
                            (0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(210)),            (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(210)),
                            close=True,
                            fill= 'grey',
                            stroke='black', stroke_width = 0))
    hex_circumference(x_pos, y_pos, dim)
    p = draw.Path(stroke_width=0.04*dim, stroke='black')
    p.M((0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(30)),            (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(30)))
    p.L((0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(210)),            (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(210)))  
    d.append(p)

  if shape == '14X':
    hex(x_pos, y_pos, dim) 
    d.append(draw.Lines(    (0-x_pos*dim)+0,                                                (0+y_pos*dim)-0,
                            (0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(330)),            (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(330)),
                            (0-x_pos*dim)+(0.5*dim)*cos(radians(300)),                       (0+y_pos*dim)-(0.5*dim)*sin(radians(300)),
                            (0-x_pos*dim)+(0.5*dim)*cos(radians(240)),                        (0+y_pos*dim)-(0.5*dim)*sin(radians(240)),
                            (0-x_pos*dim)+(0.5*dim)*cos(radians(180)),                      (0+y_pos*dim)-(0.5*dim)*sin(radians(180)),
                            (0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(150)),            (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(150)),
                            close=True,
                            fill= 'grey',
                            stroke='black', stroke_width = 0))
    hex_circumference(x_pos, y_pos, dim)
    p = draw.Path(stroke_width=0.04*dim, stroke='black')
    p.M((0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(330)),            (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(330)))
    p.L((0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(150)),            (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(150)))  
    d.append(p)

  if shape == '25X':
    hex(x_pos, y_pos, dim, color = 'grey') 
    d.append(draw.Lines(    (0-x_pos*dim)+0,                                                (0+y_pos*dim)-0,
                            (0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(90)),            (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(90)),
                            (0-x_pos*dim)+(0.5*dim)*cos(radians(60)),                       (0+y_pos*dim)-(0.5*dim)*sin(radians(60)),
                            (0-x_pos*dim)+(0.5*dim)*cos(radians(0)),                        (0+y_pos*dim)-(0.5*dim)*sin(radians(0)),
                            (0-x_pos*dim)+(0.5*dim)*cos(radians(300)),                      (0+y_pos*dim)-(0.5*dim)*sin(radians(300)),
                            (0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(270)),            (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(270)),
                            close=True,
                            fill= 'white',
                            stroke='black', stroke_width = 0))
    hex_circumference(x_pos, y_pos, dim)
    p = draw.Path(stroke_width=0.04*dim, stroke='black')
    p.M((0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(90)),            (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(90)))
    p.L((0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(270)),            (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(270)))  
    d.append(p)

  if shape == '03X':
    hex(x_pos, y_pos, dim, color = 'grey') 
    d.append(draw.Lines(    (0-x_pos*dim)+0,                                                (0+y_pos*dim)-0,
                            (0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(30)),            (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(30)),
                            (0-x_pos*dim)+(0.5*dim)*cos(radians(0)),                       (0+y_pos*dim)-(0.5*dim)*sin(radians(0)),
                            (0-x_pos*dim)+(0.5*dim)*cos(radians(300)),                        (0+y_pos*dim)-(0.5*dim)*sin(radians(300)),
                            (0-x_pos*dim)+(0.5*dim)*cos(radians(240)),                      (0+y_pos*dim)-(0.5*dim)*sin(radians(240)),
                            (0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(210)),            (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(210)),
                            close=True,
                            fill= 'white',
                            stroke='black', stroke_width = 0))
    hex_circumference(x_pos, y_pos, dim)
    p = draw.Path(stroke_width=0.04*dim, stroke='black')
    p.M((0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(30)),            (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(30)))
    p.L((0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(210)),            (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(210)))  
    d.append(p)

  if shape == '14A':
    hex(x_pos, y_pos, dim, color = 'grey') 
    d.append(draw.Lines(    (0-x_pos*dim)+0,                                                (0+y_pos*dim)-0,
                            (0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(330)),            (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(330)),
                            (0-x_pos*dim)+(0.5*dim)*cos(radians(300)),                       (0+y_pos*dim)-(0.5*dim)*sin(radians(300)),
                            (0-x_pos*dim)+(0.5*dim)*cos(radians(240)),                        (0+y_pos*dim)-(0.5*dim)*sin(radians(240)),
                            (0-x_pos*dim)+(0.5*dim)*cos(radians(180)),                      (0+y_pos*dim)-(0.5*dim)*sin(radians(180)),
                            (0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(150)),            (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(150)),
                            close=True,
                            fill= 'white',
                            stroke='black', stroke_width = 0))
    hex_circumference(x_pos, y_pos, dim)
    p = draw.Path(stroke_width=0.04*dim, stroke='black')
    p.M((0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(330)),            (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(330)))
    p.L((0-x_pos*dim)+(0.5*inside_hex_dim)*cos(radians(150)),            (0+y_pos*dim)-(0.5*inside_hex_dim)*sin(radians(150)))  
    d.append(p)

  if shape == 'Z':
    
    # deg = 0
    rot = 'rotate(' + str(deg) + ' ' + str(0-x_pos*dim) + ' ' + str(0-y_pos*dim) + ')'
    g = draw.Group(transform=rot)

    p = draw.Path(stroke_width=0.04*dim, stroke='black')
    p.M((0-x_pos*dim),            (0+y_pos*dim)-0.5*dim)
    p.L((0-x_pos*dim),            (0+y_pos*dim)+0.5*dim)  
    g.append(p)
    p = draw.Path(stroke_width=0.04*dim, stroke='black')
    p.M((0-x_pos*dim)-0.02*dim,            (0+y_pos*dim)-0.5*dim)
    p.L((0-x_pos*dim)+0.4*dim,            (0+y_pos*dim)-0.5*dim)  
    g.append(p)
    d.append(g)

  if shape == 'Y':
    
    # deg = 0
    rot = 'rotate(' + str(deg) + ' ' + str(0-x_pos*dim) + ' ' + str(0-y_pos*dim) + ')'
    g = draw.Group(transform=rot)
    
    p = draw.Path(stroke_width=0.04*dim, stroke='black')
    p.M((0-x_pos*dim),            (0+y_pos*dim)-0.5*dim)
    p.L((0-x_pos*dim),            (0+y_pos*dim)+0.5*dim)  
    g.append(p)
    p = draw.Path(stroke_width=0.04*dim, stroke='black')
    p.M((0-x_pos*dim)-0.02*dim,            (0+y_pos*dim)-0.5*dim)
    p.L((0-x_pos*dim)+0.4*dim,            (0+y_pos*dim)-0.5*dim)  
    g.append(p)

    g.append(draw.Circle((0-x_pos*dim)+0.4*dim, 0+y_pos*dim, 0.15*dim, fill='none', stroke_width=0.04*dim, stroke='black'))

    d.append(g)

  if shape == 'B':
    p = draw.Path(stroke_width=0.04*dim, stroke='black')
    p.M((0-x_pos*dim),            (0+y_pos*dim)-0.5*dim)
    p.L((0-x_pos*dim),            (0+y_pos*dim)+0.5*dim)  
    d.append(p)
    p = draw.Path(stroke_width=0.04*dim, stroke='black')
    p.M((0-x_pos*dim)+0.02*dim,            (0+y_pos*dim)+0.5*dim)
    p.L((0-x_pos*dim)-0.4*dim,            (0+y_pos*dim)+0.5*dim)  
    d.append(p)

  if shape == 'C':
    p = draw.Path(stroke_width=0.04*dim, stroke='black')
    p.M((0-x_pos*dim),            (0+y_pos*dim)-0.5*dim)
    p.L((0-x_pos*dim),            (0+y_pos*dim)+0.5*dim)  
    d.append(p)
    p = draw.Path(stroke_width=0.04*dim, stroke='black')
    p.M((0-x_pos*dim)+0.02*dim,            (0+y_pos*dim)+0.5*dim)
    p.L((0-x_pos*dim)-0.4*dim,            (0+y_pos*dim)+0.5*dim)  
    d.append(p)

    d.append(draw.Circle((0-x_pos*dim)-0.4*dim, 0+y_pos*dim, 0.15*dim, fill='none', stroke_width=0.04*dim, stroke='black'))

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
  if label == '-':
    label = ''
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
  d.append(draw.Text(label, dim*0.4, path=p, text_anchor='middle', valign='middle', line_offset=-0.5))

def add_sugar(monosaccharide, x_pos = 0, y_pos = 0, modification = '', dim = 50, compact = False, conf = '', deg = 0, text_anchor = 'middle'):
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
    draw_shape(shape = sugar_dict[monosaccharide][0], color = sugar_dict[monosaccharide][1], x_pos = x_pos, y_pos = y_pos, modification = modification, conf = conf, furanose = sugar_dict[monosaccharide][2], dim = dim, deg = deg, text_anchor = text_anchor)
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
  """Reorder multiple branches by linkage on a given glycan\n
  | Arguments:
  | :-
  | glycan (str): Input glycan string.\n
  | Returns:
  | -------
  | str: Modified glycan string.
  """
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
      if open_close[k+1][0] - open_close[k][1] == 2:
        branch1 = glycan[open_close[k][0]:open_close[k][1]]
        tmp1 = branch1[-2]
        branch2 = glycan[open_close[k+1][0]:open_close[k+1][1]]
        tmp2 = branch2[-2]
        if tmp1 == '?' and tmp2 == '?':
          if '?' in [tmp1, tmp2]:
            if len(branch1) > len(branch2):
              tmp1, tmp2 = [1, 2]
            else:
              tmp1, tmp2 = [2, 1]
        if tmp1 > tmp2:
          glycan = glycan[:open_close[k][0]] + branch2 + '][' + branch1 + glycan[open_close[k+1][1]:]
  return glycan

def multiple_branch_branches(glycan):
  """Reorder nested multiple branches by linkage on a given glycan\n
  | Arguments:
  | :-
  | glycan (str): Input glycan string.\n
  | Returns:
  | -------
  | str: Modified glycan string.
  """
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
        # branch_end = [j[-2] for j in [re.sub(r'\[[^]]+\]', '', group1), re.sub(r'\[[^]]+\]', '', group2)]]
        # branch_end = [k.replace('z', '9') for k in branch_end]
        branch_len = [len(k) for k in min_process_glycans([group1, group2])]

        # print(branch_end[0])
        # print('g1 ' + group1)
        # print(branch_end[1])
        # print('g2 ' + group2)

        # if branch_end[0] in ['?', ')'] and branch_end[1] in ['?', ')']:
        #   branch_end[0] = 1
        #   branch_end[1] = 2
        # if branch_end[0] in ['?', ')']:
        #   branch_end[0] = branch_end[1] + 1
        # if branch_end[1] in ['?', ')']:
        #   branch_end[1] = branch_end[0] + 1

        # print(branch_end[0])
        # print('g1 ' + group1)
        # print(branch_end[1])
        # print('g2 ' + group2)

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
  """order nested branches by linkage, ascending\n
  | Arguments:
  | :-
  | glycan (string): glycan in IUPAC-condensed format
  | Returns:
  | :-
  | Returns re-ordered glycan
  """
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
        if '?' in k[0] and '?' in k[-1]:
          tmp.append('?')
        elif '?' in k[0]:
          tmp.append(' '+k[-1])
        elif '?' in k[-1]:
          if bool(re.compile("^a\d").match(k)):
            tmp.append('\u03B1')
          elif bool(re.compile("^b\d").match(k)):
            tmp.append('\u03B2')
        elif bool(re.compile("^a\d").match(k)):
          tmp.append('\u03B1 '+k[-1])
        elif bool(re.compile("^b\d").match(k)):
          tmp.append('\u03B2 '+k[-1])
        elif bool(re.compile("^\d-\d").match(k)):
          tmp.append(k[0]+' - '+k[-1])
        else:
          tmp.append('-')
      bonds.append(tmp)
      tmp = []
    return bonds
  else:
    for k in linkage_list:
      if '?' in k[0] and '?' in k[-1]:
        bonds.append('?')
      elif '?' in k[0]:
        bonds.append(' '+k[-1])
      elif '?' in k[-1]:
        if bool(re.compile("^a\d").match(k)):
          bonds.append('\u03B1')
        elif bool(re.compile("^b\d").match(k)):
          bonds.append('\u03B2')
      elif bool(re.compile("^a\d").match(k)):
        bonds.append('\u03B1 '+k[-1])
      elif bool(re.compile("^b\d").match(k)):
        bonds.append('\u03B2 '+k[-1])
      elif bool(re.compile("^\d-\d").match(k)):
        bonds.append(k[0]+' - '+k[-1])
      else:
        bonds.append('-')
    return bonds

def split_monosaccharide_linkage(label_list):
  """Split the monosaccharide linkage and extract relevant information from the given label list.\n
  | Arguments:
  | :-
  | label_list (list): List of labels \n
  | Returns:
  | -------
  | Three lists - sugar, sugar_modification, and bond.
  |   - sugar (list): List of sugars 
  |   - sugar_modification (list): List of sugar modifications 
  |   - bond (list): List of bond information 
  """
  if any(isinstance(el, list) for el in label_list):
    sugar = [k[::2][::-1] for k in label_list]
    sugar_modification = [[get_modification(k) if k in lib else '' for k in y] for y in sugar]
    sugar_modification = [[multireplace(k, {'O':'', '-ol':''}) for k in y] for y in sugar_modification]
    sugar = [[get_core(k) if k not in additions else k for k in y] for y in sugar]
    bond = [k[1::2][::-1] for k in label_list]
  else:
    sugar = label_list[::2][::-1]
    sugar_modification = [get_modification(k) if k not in additions else '' for k in sugar]
    sugar_modification = [multireplace(k, {'O':'', '-ol':''}) for k in sugar_modification]
    sugar = [get_core(k) if k not in additions else k for k in sugar]
    bond = label_list[1::2][::-1]

  return sugar, sugar_modification, bond

def glycan_to_skeleton(glycan_string): 
  """convert glycan to skeleton of node labels according to the respective glycan graph object\n
  | Arguments:
  | :-
  | glycan_string (string):
  | Returns:
  | :-
  | node label skeleton of glycan (string)\n
  """
  tmp = multireplace(glycan_string, {'(': ',', ')': ',', '[': ',[,', ']': ',],'})
  tmp = tmp.split(',')
  tmp = [k for k in tmp if k != '']

  tmp2 = []
  idx = 0
  for k in tmp:
    if k in ['[', ']']:
      tmp2.append(k)
    else:
      tmp2.append(str(idx))
      idx = idx + 1

  tmp2 = '-'.join(tmp2)
  tmp2 = multireplace(tmp2, {'[-': '[', '-]': ']'})
  return(tmp2)

def get_coordinates_and_labels(draw_this, show_linkage = True, draw_lib = draw_lib, extend_lib = False):
  """ Extract monosaccharide labels and calculate coordinates for drawing\n
  | Arguments:
  | :-
  | draw_this (str): Glycan structure to be drawn.
  | show_linkage (bool, optional): Flag indicating whether to show linkages. Default is True.
  | draw_lib (dict): lib extended with non-standard glycowords
  | extend_lib (bool): If True, further extend the library with given input. Default is False.\n
  | Returns:
  | :-
  | data_combined (list of lists)
  | contains lists with monosaccharide label, x position, y position, modification, bond, conformation
  """
  if extend_lib == True:
    draw_lib = expand_lib(draw_lib, [draw_this])
  
  if bool(re.search('^\[', draw_this)) == False:
    draw_this = multiple_branch_branches(draw_this)
    draw_this = multiple_branches(draw_this)
    draw_this = reorder_for_drawing(draw_this)
    draw_this = multiple_branches(draw_this)
    draw_this = multiple_branch_branches(draw_this)
  # print(draw_this)

  graph = glycan_to_nxGraph(draw_this, libr = draw_lib)
  node_labels = nx.get_node_attributes(graph, 'string_labels')
  edges = graph.edges()
  branch_points = [e[1] for e in edges if abs(e[0]-e[1]) > 1]
  
  glycan = glycan_to_skeleton(draw_this)

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
  # print(branch_connection)
  for k in range(len(unique(branch_connection))):
    tmp = [branch_y_pos[j][0] for j in unwrap(get_indices(branch_connection, [unique(branch_connection)[k]]))]
    # print(tmp)
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

      tmp_listy = []
      if main_sugar[-1] == 'Fuc':# and len(main_sugar) == 2:
        pass
      else:
        for k in range(len(main_sugar)):
          if main_sugar_x_pos[k] < min([x for x in unwrap(branch_x_pos) if x > 0]):
            tmp_listy.append(main_sugar_x_pos[k])
      for k in range(len(tmp_listy)):
        if max(branch_connection) > max(tmp_listy):
          main_sugar_y_pos[k] = main_sugar_y_pos[k] + (to_add/2)
        else:
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
  """Draw a bracket shape at the specified position and dimensions.\n
  | Arguments:
  | :-
  | x (int): X coordinate of the bracket on the drawing canvas.
  | y_min_max (list): List containing the minimum and maximum Y coordinates for the bracket.
  | direction (str, optional): Direction of the bracket. Possible values are 'right' and 'left'. Default is 'right'.
  | dim (int, optional): Arbitrary dimension unit used for scaling the bracket's size. Default is 50.\n
  | Returns:
  | :-
  | None
  """
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

def GlycoDraw(draw_this, vertical = False, compact = False, show_linkage = True, dim = 50, output = None):
  """Draws a glycan structure based on the provided input.\n
  | Arguments:
  | :-
  | draw_this (str): The glycan structure or motif to be drawn.
  | vertical (bool, optional): Set to True to draw the structure vertically. Defaults to False.
  | compact (bool, optional): Set to True to draw the structure in a compact form. Defaults to False.
  | show_linkage (bool, optional): Set to False to hide the linkage information. Defaults to True.
  | dim (int, optional): The dimension (size) of the individual sugar units in the structure. Defaults to 50.
  | output (str, optional): The path to the output file to save as SVG or PDF. Defaults to None.
  """

  bond_hack = False
  if 'Man(a1-?)' in draw_this: 
    if 'Man(a1-3)' not in draw_this and 'Man(a1-6)' not in draw_this:
      draw_this = 'Man(a1-6)'.join(draw_this.rsplit('Man(a1-?)', 1))
      bond_hack = True
  
  if draw_this[-1] == ')':
    draw_this = draw_this + 'blank'

  if compact == True:
    show_linkage = False

  # handle floaty bits if present
  floaty_bits = []
  for openpos, closepos, level in matches(draw_this, opendelim='{', closedelim='}'):
      floaty_bits.append(draw_this[openpos:closepos]+'blank')
      draw_this = draw_this[:openpos-1]+ len(draw_this[openpos-1:closepos+1])*'*' + draw_this[closepos+1:]
  draw_this = draw_this.replace('*', '')

  try:
    data = get_coordinates_and_labels(draw_this, show_linkage = show_linkage)
  except:
    try:
      draw_this = motif_list.loc[motif_list.motif_name == draw_this].motif.values.tolist()[0]
      data = get_coordinates_and_labels(draw_this, show_linkage = show_linkage)
    except:
        try:
          data = get_coordinates_and_labels(draw_this, show_linkage = show_linkage, extend_lib = True)
        except:
          # return print('Error: did you enter a real glycan or motif?')
          # print(e)
          sys.exit(1)

  main_sugar, main_sugar_x_pos, main_sugar_y_pos, main_sugar_modification, main_bond, main_conf = data[0]
  branch_sugar, branch_x_pos, branch_y_pos, branch_sugar_modification, branch_bond, branch_connection, b_conf = data[1]
  branch_branch_sugar, branch_branch_x_pos, branch_branch_y_pos, branch_branch_sugar_modification, branch_branch_bond, branch_branch_connection, bb_conf  = data[2]
  bbb_sugar, bbb_x_pos, bbb_y_pos, bbb_sugar_modification, bbb_bond, bbb_connection, bbb_conf  = data[3]

  while bond_hack == True:
    for k in range(len(main_bond)):
      if main_sugar[k] + '--' + main_bond[k] == 'Man--α 6':
        main_bond[k] = 'α'
        bond_hack = False

    for branch in range(len(branch_bond)):
      for bond in range(len(branch_bond[branch])):
        if branch_sugar[branch][bond] + '--' + branch_bond[branch][bond] == 'Man--α 6':
          branch_bond[branch][bond] = 'α'
          bond_hack = False
    bond_hack = False

  if show_linkage == False:
    main_bond = ['-' for x in main_bond]
    branch_bond = [['-' for x in y] for y in branch_bond]
    branch_branch_bond = [['-' for x in y] for y in branch_branch_bond]
    bbb_bond = [['-' for x in y] for y in bbb_bond]

  # drawsvg 2.0 y fix
  main_sugar_y_pos = [(k*-1) for k in main_sugar_y_pos]
  branch_y_pos = [[(x*-1) for x in y] for y in branch_y_pos]
  branch_branch_y_pos = [[(x*-1) for x in y] for y in branch_branch_y_pos]
  bbb_y_pos = [[(x*-1) for x in y] for y in bbb_y_pos]

  # calculate angles for main chain Y, Z fragments 
  main_deg = []
  for k in range(len(main_sugar)):
    if main_sugar[k] in ['Z', 'Y']:
      slope = -1*(main_sugar_y_pos[k]-main_sugar_y_pos[k-1])/((main_sugar_x_pos[k]*2)-(main_sugar_x_pos[k-1]*2))
      main_deg.append(degrees(atan(slope)))
    else:
      main_deg.append(0)

  # calculate angles for branch Y, Z fragments 
  branch_deg = []
  for k in range(len(branch_sugar)):
    tmp = []
    for j in range(len(branch_sugar[k])):
      if branch_sugar[k][j] in ['Z', 'Y']:
        if len(branch_sugar[k]) == 1:
          slope = -1*(branch_y_pos[k][j]-main_sugar_y_pos[branch_connection[k]])/((branch_x_pos[k][j]*2)-(main_sugar_x_pos[branch_connection[k]]*2))
          tmp.append(degrees(atan(slope)))
        else:
          slope = -1*(branch_y_pos[k][j]-branch_y_pos[k][j-1])/((branch_x_pos[k][j]*2)-(branch_x_pos[k][j-1]*2))
          tmp.append(degrees(atan(slope)))
      else:
        tmp.append(0)
    branch_deg.append(tmp)

  # calculate angles for branch_branch Y, Z fragments 
  branch_branch_deg = []
  for k in range(len(branch_branch_sugar)):
    tmp = []
    for j in range(len(branch_branch_sugar[k])):
      if branch_branch_sugar[k][j] in ['Z', 'Y']:
        if len(branch_branch_sugar[k]) == 1:
          slope = -1*(branch_branch_y_pos[k][j]-branch_y_pos[branch_branch_connection[k][0]][branch_branch_connection[k][1]])/((branch_branch_x_pos[k][j]*2)-(branch_x_pos[branch_branch_connection[k][0]][branch_branch_connection[k][1]]*2))
          tmp.append(degrees(atan(slope)))
        else:
          slope = -1*(branch_branch_y_pos[k][j]-branch_branch_y_pos[k][j-1])/((branch_branch_x_pos[k][j]*2)-(branch_branch_x_pos[k][j-1]*2))
          tmp.append(degrees(atan(slope)))
      else:
        tmp.append(0)
    branch_branch_deg.append(tmp)

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

  # global d2
  global d

  ## draw
  d2 = draw.Drawing(width, height, origin=(x_ori,y_ori)) #context=draw.Context(invert_y=True)

  if vertical == True:
    deg = 90
  else:
    deg = 0

  rot = 'rotate(' + str(deg) + ' ' + str(width/2) + ' ' + str(height/2) + ')'
  d = draw.Group(transform=rot)

  # bond main chain
  [add_bond(main_sugar_x_pos[k+1], main_sugar_x_pos[k], main_sugar_y_pos[k+1], main_sugar_y_pos[k], main_bond[k], dim = dim, compact = compact) for k in range(len(main_sugar)-1)]
  # bond branch
  [add_bond(branch_x_pos[b_idx][s_idx+1], branch_x_pos[b_idx][s_idx], branch_y_pos[b_idx][s_idx+1], branch_y_pos[b_idx][s_idx], branch_bond[b_idx][s_idx+1], dim = dim, compact = compact) for b_idx in range(len(branch_sugar)) for s_idx in range(len(branch_sugar[b_idx])-1) if len(branch_sugar[b_idx]) > 1]
  # bond branch to main chain
  [add_bond(branch_x_pos[k][0], main_sugar_x_pos[branch_connection[k]], branch_y_pos[k][0], main_sugar_y_pos[branch_connection[k]], branch_bond[k][0], dim = dim, compact = compact) for k in range(len(branch_sugar))]
  # bond branch branch
  [add_bond(branch_branch_x_pos[b_idx][s_idx+1], branch_branch_x_pos[b_idx][s_idx], branch_branch_y_pos[b_idx][s_idx+1], branch_branch_y_pos[b_idx][s_idx], branch_branch_bond[b_idx][s_idx+1], dim = dim, compact = compact) for b_idx in range(len(branch_branch_sugar)) for s_idx in range(len(branch_branch_sugar[b_idx])-1) if len(branch_branch_sugar[b_idx]) > 1]
  # bond branch branch branch
  [add_bond(bbb_x_pos[b_idx][s_idx+1], bbb_x_pos[b_idx][s_idx], bbb_y_pos[b_idx][s_idx+1], bbb_y_pos[b_idx][s_idx], bbb_bond[b_idx][s_idx+1], dim = dim, compact = compact) for b_idx in range(len(bbb_sugar)) for s_idx in range(len(bbb_sugar[b_idx])-1) if len(bbb_sugar[b_idx]) > 1]
  # bond branch_branch to branch
  [add_bond(branch_branch_x_pos[k][0], branch_x_pos[branch_branch_connection[k][0]][branch_branch_connection[k][1]], branch_branch_y_pos[k][0], branch_y_pos[branch_branch_connection[k][0]][branch_branch_connection[k][1]], branch_branch_bond[k][0], dim = dim, compact = compact) for k in range(len(branch_branch_sugar))]
  # bond branch_branch_branch to branch_branch
  [add_bond(bbb_x_pos[k][0], branch_branch_x_pos[bbb_connection[k][0]][bbb_connection[k][1]], bbb_y_pos[k][0], branch_branch_y_pos[bbb_connection[k][0]][bbb_connection[k][1]], bbb_bond[k][0], dim = dim, compact = compact) for k in range(len(bbb_sugar))]
  # sugar main chain
  [add_sugar(main_sugar[k], main_sugar_x_pos[k], main_sugar_y_pos[k], modification = main_sugar_modification[k], conf = main_conf[k], compact = compact, dim = dim, deg = main_deg[k]) for k in range(len(main_sugar))]
  # sugar branch
  [add_sugar(branch_sugar[b_idx][s_idx], branch_x_pos[b_idx][s_idx], branch_y_pos[b_idx][s_idx], modification = branch_sugar_modification[b_idx][s_idx], conf = b_conf[b_idx][s_idx], compact = compact, dim = dim, deg = branch_deg[b_idx][s_idx]) for b_idx in range(len(branch_sugar)) for s_idx in range(len(branch_sugar[b_idx]))]
  # sugar branch_branch 
  [add_sugar(branch_branch_sugar[b_idx][s_idx], branch_branch_x_pos[b_idx][s_idx], branch_branch_y_pos[b_idx][s_idx], modification = branch_branch_sugar_modification[b_idx][s_idx], conf = bb_conf[b_idx][s_idx], compact = compact, dim = dim, deg = branch_branch_deg[b_idx][s_idx]) for b_idx in range(len(branch_branch_sugar)) for s_idx in range(len(branch_branch_sugar[b_idx]))]
  # sugar branch branch branch
  [add_sugar(bbb_sugar[b_idx][s_idx], bbb_x_pos[b_idx][s_idx], bbb_y_pos[b_idx][s_idx], modification = bbb_sugar_modification[b_idx][s_idx], conf = bbb_conf[b_idx][s_idx], compact = compact, dim = dim) for b_idx in range(len(bbb_sugar)) for s_idx in range(len(bbb_sugar[b_idx]))]

  if floaty_bits != []:
    fb_count = {i:floaty_bits.count(i) for i in floaty_bits}
    floaty_bits = list(set(floaty_bits))
    # floaty_data = [get_coordinates_and_labels(floaty_bits[k], show_linkage = show_linkage) for k in range(len(floaty_bits))]
    floaty_data = []
    for k in range(len(floaty_bits)):
      try:
        floaty_data.append(get_coordinates_and_labels(floaty_bits[k], show_linkage = show_linkage))
      except:
        floaty_data.append(get_coordinates_and_labels('blank(-)blank', show_linkage = show_linkage))
    y_span = max_y-min_y
    n_floats = len(floaty_bits)
    floaty_span = n_floats * 2 - 2
    y_diff = (floaty_span/2) - (y_span/2)

    for j in range(len(floaty_data)):
      floaty_sugar, floaty_sugar_x_pos, floaty_sugar_y_pos, floaty_sugar_modification, floaty_bond, floaty_conf = floaty_data[j][0]
      floaty_sugar_x_pos = [floaty_sugar_x_pos[k] + max_x + 1 for k in floaty_sugar_x_pos]
      floaty_sugar_y_pos = [floaty_sugar_y_pos[k] + 2 * j - y_diff for k in floaty_sugar_y_pos]
      # drawsvg 2.0 fix
      floaty_sugar_y_pos = [(k*-1) for k in floaty_sugar_y_pos]
      if floaty_sugar != ['blank', 'blank']:
        [add_bond(floaty_sugar_x_pos[k+1], floaty_sugar_x_pos[k], floaty_sugar_y_pos[k+1], floaty_sugar_y_pos[k], floaty_bond[k], dim = dim, compact = compact) for k in range(len(floaty_sugar)-1)]
        [add_sugar(floaty_sugar[k], floaty_sugar_x_pos[k], floaty_sugar_y_pos[k], modification = floaty_sugar_modification[k], conf=floaty_conf, compact = compact, dim = dim) for k in range(len(floaty_sugar))]
      else:
        add_sugar('text', min(floaty_sugar_x_pos), floaty_sugar_y_pos[-1], modification = floaty_bits[j].replace('blank',''), compact=compact, dim=dim, text_anchor='end')

      if fb_count[floaty_bits[j]] > 1:
        if compact == False:
          add_sugar('blank', max(floaty_sugar_x_pos)+0.5, floaty_sugar_y_pos[-1]-0.75, modification = str(fb_count[floaty_bits[j]]) + 'x', compact=compact, dim=dim )
        else:
          add_sugar('blank', max(floaty_sugar_x_pos)+0.75, floaty_sugar_y_pos[-1]-1.2, modification = str(fb_count[floaty_bits[j]]) + 'x', compact=compact, dim=dim )

    if compact == False:
      draw_bracket(max_x*2+1, (min_y, max_y), direction = 'right', dim = dim)
    elif compact == True:
      draw_bracket(max_x*1.2+1, ((min_y *0.5)* 1.2, (max_y *0.5)* 1.2), direction = 'right', dim = dim)
  
  d2.append(d)

  if output is not None:
      data = d2.as_svg()
      data = re.sub(r'<text font-size="17.5" ', r'<text font-size="17.5" font-family="century gothic" font-weight="bold" ', data)
      data = re.sub(r'<text font-size="20.0" ', r'<text font-size="20" font-family="century gothic" ', data)
      data = re.sub(r'<text font-size="15.0" ', r'<text font-size="17.5" font-family="century gothic" font-style="italic" ', data)
      
      if 'svg' in output:
        with open(output, 'w') as f:
          f.write(data)
      
      elif 'pdf' in output:
        cairosvg.svg2pdf(bytestring=data, write_to=output)
  return d2

def scale_in_range(listy, a, b):
  """Normalize list of numbers in range a to b\n
  | Arguments:
  | :-
  | listy (list): list of numbers as integers/floats
  | a (integer/float): min value in normalized range
  | b (integer/float): max value in normalized range\n
  | Returns:
  | :-
  | Normalized list of numbers
  """  
  tmp = []
  for k in range(len(listy)):
    norm = (b-a) * ((listy[k]-min(listy))/(max(listy)-min(listy))) + a
    tmp.append(norm)
  return tmp

def annotate_figure(svg_input, scale_range = (25, 80), compact = False, glycan_size = 'medium', filepath = '',
                    scale_by_DE_res = None, x_thresh = 1, y_thresh = 0.05, x_metric = 'Log2FC'):
  """Modify matplotlib svg figure to replace text labels with glycan figures\n
  | Arguments:
  | :-
  | svg_input (string): absolute path including full filename for input svg figure
  | scale_range (tuple): tuple of two integers defining min/max glycan dim; default:(25,80)
  | compact (bool): if True, draw compact glycan figures; default:False
  | glycan_size (string): modify glycan size; default:'medium'; options are 'small', 'medium', 'large'
  | scale_by_DE_res (df): result table from motif_analysis.get_differential_expression. Include to scale glycan figure size by -10logp
  | y_thresh (float): corr p threshhold for datapoints included for scaling, set to match get_differential_expression; default:0.05
  | x_thresh (float): absolute x metric threshold for datapoints included for scaling, set to match get_differential_expression; defualt:1.0
  | filepath (string): absolute path including full filename allows for saving the plot\n
  | Returns:
  | :-
  | Modified figure svg code
  """
  glycan_size_dict = {
      'small': 'scale(0.1 0.1)  translate(0, -74)',
      'medium': 'scale(0.2 0.2)  translate(0, -55)',
      'large': 'scale(0.3 0.3)  translate(0, -49)'
      }
  
  glycan_scale = ''
    
  if scale_by_DE_res is not None:
    res_df = scale_by_DE_res.loc[(abs(scale_by_DE_res[x_metric]) > x_thresh) & (scale_by_DE_res['corr p-val'] < y_thresh)]
    y = -np.log10(res_df['corr p-val'].values.tolist())
    l = res_df['Glycan'].values.tolist()
    glycan_scale = [y, l]
    
  # get svg code
  svg_tmp = open(svg_input,"r").read()

  # get all text labels
  matches = re.findall(r"<!--.*-->[\s\S]*?<\/g>", svg_tmp)

  # prepare for appending
  svg_tmp = svg_tmp.replace('</svg>', '')
  element_id = 0

  edit_svg = False

  for match in matches:
    # keep track of current label and position in figure
    current_label = re.findall(r'<!--\s*(.*?)\s*-->', match)[0]
    current_pos = '<g transform' + re.findall(r'<g transform\s*(.*?)\s*">', match)[0] + '">'
    current_pos = current_pos.replace('scale(0.1 -0.1)', glycan_size_dict[glycan_size])
    # check if label is glycan
    try:  
      glycan_to_nxGraph(current_label)
      edit_svg = True
    except:
      pass
    try:  
      glycan_to_nxGraph(motif_list.loc[motif_list.motif_name == current_label].motif.values.tolist()[0])
      edit_svg = True
    except:
      pass
    # delete text label, append glycan figure
    if edit_svg == True:
      svg_tmp = svg_tmp.replace(match, '')
      if glycan_scale == '':
        d = GlycoDraw(current_label, compact = compact)
      else:
        d = GlycoDraw(current_label, compact = compact, dim = scale_in_range(glycan_scale[0], scale_range[0], scale_range[1])[glycan_scale[1].index(current_label)])
      data = d.as_svg()
      data = data.replace('<?xml version="1.0" encoding="UTF-8"?>\n', '')
      id_matches = re.findall(r'd\d+', data)
      # reassign element ids to avoid duplicates
      for id in id_matches:
        data = data.replace(id, 'd' + str(element_id))
        element_id += 1
      svg_tmp = svg_tmp + '\n' + current_pos + '\n' + data + '\n</g>'    
      edit_svg = False
  
  svg_tmp = svg_tmp + '</svg>'
  
  if len(filepath) > 1:
      if filepath.split('.')[-1] == 'pdf':
        cairosvg.svg2pdf(bytestring = svg_tmp, write_to = filepath, dpi = 300)
      elif filepath.split('.')[-1] == 'svg':
        cairosvg.svg2svg(bytestring = svg_tmp, write_to = filepath, dpi = 300)
      elif filepath.split('.')[-1] == 'png':
        cairosvg.svg2png(bytestring = svg_tmp, write_to = filepath, dpi = 300)    
  else:
      return svg_tmp