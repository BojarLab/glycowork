#!/usr/bin/env python
"""Give the docs' glycan drawings a dark-mode rendering as GlycoDraw die-cut stickers, as slate shows them.

Quarto runs this before rendering (project pre-render in _quarto.yml), in nbdev's _proc copy of the notebooks, so
the notebooks themselves keep their outputs. Every code cell whose stored output is a GlycoDraw or
plot_glycans_grid drawing is run again with sticker = True:
* a drawing (image/png) gets the sticker, on a transparent background, as a second output, and the cell gets
  `#| renderings: [light, dark]`, so Quarto shows the stored drawing in light mode and the sticker in dark mode;
* a Markdown text with inline glycans (as_markdown) gets each sticker next to the stored image, as `dark-content`
  next to `light-content`.
A cell that does not run on its own (it needs a variable of an earlier cell) keeps its stored output only. Calls run
with suppress = True, since outside Jupyter GlycoDraw would open a matplotlib window and wait for it. Single-file
renders (nbdev's README.md) skip all this, and a cell that already has its dark rendering is left alone, since nbdev
keeps a _proc notebook whose source did not change.
"""
import ast
import base64
import json
import os
import re
import sys
from pathlib import Path
if os.environ.get('QUARTO_PROJECT_RENDER_ALL') != '1':
    sys.exit()
from IPython.display import Markdown
from glycorender.render import convert_svg_to_png
from glycowork.motif import draw

IMG = re.compile(r'<img src="data:image/svg\+xml;base64,[^"]*"[^>]*>')


def sticker(f):
    return lambda *args, **kwargs: f(*args, **{**kwargs, 'sticker': True, 'suppress': True})


for path in sorted(Path('.').glob('*.ipynb')):
    nb = json.loads(path.read_text(encoding = 'utf-8'))
    changed = 0
    for cell in nb['cells']:
        source = ''.join(cell['source'])
        if cell['cell_type'] != 'code' or 'show_doc' in source or 'renderings:' in source or 'dark-content' in json.dumps(cell.get('outputs')) or not re.search(r'\b(GlycoDraw|plot_glycans_grid)\(', source):
            continue
        out = next((o for o in cell.get('outputs', []) if o['output_type'] == 'execute_result'), None)
        if out is None or not ({'image/png', 'text/markdown'} & set(out['data'])):
            continue
        tree = ast.parse('\n'.join(l for l in source.splitlines() if not l.startswith('#|')))
        last = tree.body.pop()
        scope = {'GlycoDraw': sticker(draw.GlycoDraw), 'plot_glycans_grid': sticker(draw.plot_glycans_grid), 'Markdown': Markdown}
        try:
            exec(compile(tree, path.name, 'exec'), scope)
            result = eval(compile(ast.Expression(last.value), path.name, 'eval'), scope)
        except Exception as e:
            print(f'{path.name}: no dark rendering for {source[:60]!r} ({e})')
            continue
        if 'image/png' in out['data']:
            png = convert_svg_to_png(result.as_svg(), None, scale = 2.0, return_bytes = True, sticker = True)
            cell['outputs'].append({'output_type': 'display_data', 'metadata': {'image/png': {'width': result.drawing_obj.width, 'height': result.drawing_obj.height}},
                                    'data': {'image/png': base64.b64encode(png).decode('ascii'), 'text/plain': ['<sticker>']}})
            cell['source'] = ['#| renderings: [light, dark]\n'] + cell['source']
        else:
            stickers = iter(IMG.findall(result.data))
            md = ''.join(out['data']['text/markdown'])
            out['data']['text/markdown'] = IMG.sub(lambda m: m[0].replace('<img ', '<img class="light-content" ', 1) + next(stickers).replace('<img ', '<img class="dark-content" ', 1), md)
        changed += 1
    if changed:
        path.write_text(json.dumps(nb, indent = 1, ensure_ascii = False), encoding = 'utf-8')
        print(f'{path.name}: {changed} glycan cells with a dark rendering')
