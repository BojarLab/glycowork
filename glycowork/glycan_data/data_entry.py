import re
import ast
import csv
import warnings
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from pathlib import Path
from pandas.io.common import get_handle
from typing import Callable
from functools import cache
from glycowork.glycan_data.loader import GlycoDataFrame, glycomics_data_loader, glycoproteomics_data_loader, lectin_array_data_loader
from glycowork.motif.processing import check_nomenclature, canonicalize_composition, canonicalize_iupac, is_composition, _STRUCGP_CODE, _COMP_ADDUCT
from glycowork.motif.tokenization import glycan_to_composition
from glycowork.motif.graph import glycan_to_nxGraph, compare_glycans


_GP_SIGNATURES = [('fragpipe', ('totalglycancomposition', 'proteinstart')), ('pglyco', ('prosites', 'glycancomposition')), ('byologic', ('xicareasummed', 'mod.summary')),
                  ('byonic', ('proteinname', 'glycansnhfagna')), ('byonic', ('proteinname', 'composition', 'position')), ('glycresoft', ('glycopeptide', 'peptide_start')),
                  ('metamorpheus', ('plausibleglycancomposition', 'fullsequence')), ('decipher', ('glycosite', 'glycancomposition')), ('strucgp', ('glycosite_position', 'glycancomposition')),
                  ('glycanfinder', ('proteinaccession', 'glycantype')), ('decipher_site', ('site', 'glycan'))]  # columns (whitespace removed, lowercased) that identify each glycoproteomics engine
_SUMMARY_HEAD = re.compile(r'Absolute Intensity|Analyte Area(?: - Background Area)?\tCalibrated|(?:Background Subtracted )?Intensities \(Area\)\t')  # an abundance block of a LaCyTools Summary.txt, or of a MassyTools 1 or 2 summary.txt
_LACY_QC = re.compile(r'(Mass Accuracy \[ppm\]|Isotopic Pattern Quality|S/N) \((\d+)\+\)')  # the quality blocks of a LaCyTools Summary.txt, one per charge state
_TAG_MASSES = (140.095, 144.1021, 148.1092, 224.1525, 225.1558, 229.1629, 304.2054,
               304.2071)  # amine tags that also label S and T: mTRAQ, iTRAQ 4-plex, TMT0, TMT2, TMT, iTRAQ 8-plex, TMTpro
_DECOY = re.compile(r'>?\s*(?:rev_|reverse|decoy|#decoy#|xxx_)',
                    flags = re.IGNORECASE)  # decoy protein labels: 'REV_sp|...' (pGlyco, MaxQuant), 'rev_sp|...' (FragPipe), '>Reverse sp|...' (Byonic), 'DECOY_P02763' (MetaMorpheus), '#DECOY#P02763' (PEAKS)


def _read_table(f: str | Path, # an Excel workbook, a .csv separated by commas, semicolons, or tabs, or any other text table separated by tabs
                dtype: type | None = str # passed to pd.read_csv or pd.read_excel
                ) -> list[pd.DataFrame]: # one frame per sheet
    "Reads a table file, including the CSV of European Excel or Skyline (';' between fields, ',' as decimal mark, Windows code page) and published supplementary tables with title, caption, or merged group rows above the header"

    def header(rows: list[list]) -> int | None:  # the first of the top rows with at least half as many filled cells as the widest one; None for a single column, read as is
        filled = [sum(isinstance(x, str) and bool(x.strip()) for x in r) for r in rows]
        return next((i for i, k in enumerate(filled) if k >= max(2, max(filled) / 2)), None)

    if Path(f).suffix.lower() in ('.xlsx', '.xlsm', '.xlsb', '.xls', '.ods'):
        with pd.ExcelFile(f) as xl:
            sheets = [
                xl.parse(s, header = header(xl.parse(s, header = None, nrows = 50, dtype = str).values.tolist()) or 0,
                         dtype = dtype) for s in xl.sheet_names]
        return [d for d in sheets if not d.empty] or sheets[
            :1]  # the empty Sheet2 or Sheet3 of a workbook holds no table
    seps = (',', ';', '\t') if '.csv' in [x.lower() for x in Path(f).suffixes[-2:]] else ('\t',)  # also a compressed '.csv.xz', which pandas decompresses
    with get_handle(f, 'r', encoding = 'utf-8-sig', errors = 'replace', compression = 'infer') as fh:
        lines = [l for _, l in zip(range(50), fh.handle)]
    found = []
    for s in seps:
        try:
            if (h := header(list(csv.reader(lines, delimiter = s)))) is not None:
                found.append((s, h))
        except csv.Error:  # a wide line without this separator is one field beyond the csv module's size limit, e.g., the header of v12_glycan_binding.csv with ';'
            pass
    sep, skip = min(found, key = lambda t: (t[1], -lines[t[1]].count(t[0])),
                    default = (max(seps, key = (lines or [''])[0].count),
                               0))  # the separator that puts the header highest, then splits it most often; ties go to the comma
    marks = set(re.findall(r'(?m)(?:^|;)\s*[-+]?\d+([.,])\d+(?:[eE][-+]?\d+)?\s*(?=;|$)', ''.join(lines[skip + 1:])))
    dec = ',' if sep == ';' and marks != {
        '.'} else '.'  # semicolons mostly come with decimal commas (European Excel, Skyline), but some tools write them with decimal points
    try:
        df = pd.read_csv(f, sep = sep, dtype = dtype, decimal = dec, skiprows = skip)
    except UnicodeDecodeError:  # Excel on Windows writes CSV in its code page, e.g., 'Müller' as cp1252
        df = pd.read_csv(f, sep = sep, dtype = dtype, decimal = dec, skiprows = skip, encoding = 'cp1252',
                         encoding_errors = 'replace')
    return [df.replace(r'^([-+]?\d+),(\d+(?:[eE][-+]?\d+)?)$', r'\1.\2',
                       regex = True) if dec == ',' and dtype is str else df]  # values read as text keep their decimal comma otherwise


def _gp_engine(df: pd.DataFrame # a table as read from file
               ) -> tuple[str | None, dict]: # (engine, or None if none matches; header map with whitespace removed and lowercased)
    "Recognizes the output of a glycoproteomics search engine by its columns"
    c = {re.sub(r'\s+', '', str(k)).lower(): k for k in df.columns}
    ok = {'decipher_site': lambda: df[c['site']].astype(str).str.contains('@').any(), 'byonic': lambda: any(k.startswith('peptide') for k in c)}  # a Glyco-Decipher site is 'P00738@211' and Byonic always reports the peptide, whereas a table of one's own may well have 'Site' and 'Glycan', or 'Protein Name', 'Position', and 'Composition' columns
    return next((s for s, ks in _GP_SIGNATURES if all(k in c for k in ks) and ok.get(s, lambda: True)()), None), c


@cache
def _accession(protein: str # protein label as a search engine writes it, e.g., '>sp|P02763|A1AG1_HUMAN Alpha-1-acid glycoprotein 1' or 'Q9P0K1(pre=R,post=L)'
               ) -> str: # bare accession, '_' turned into '-' so that it cannot be mistaken for a separator of protein_site_composition
    "Extracts the accession from a protein label"
    parts = re.sub(r'^(?:contam_|CON__)', '', str(protein).strip().lstrip('>'), flags = re.IGNORECASE).split()[0].split('(')[0].split('|')  # contaminants: 'contam_sp|P02768|ALBU_HUMAN' (FragPipe), 'CON__P02768' (MaxQuant)
    return (parts[1] if len(parts) > 2 and parts[0] in ('sp', 'tr') else parts[0]).replace('_', '-')


def _glycan_mass(res: str, # modified residue
                 mass: float # modification mass
                 ) -> bool: # whether the modification is a glycan
    "A glycan without a name is a mass of at least 130 Da on N, S, or T that is no isobaric tag; a pentose (132.04) is the lightest glycan"
    return res in 'NST' and mass >= 130 and not any(abs(mass - t) < 0.01 for t in _TAG_MASSES)


def _glycan_mods(peptide: str, # peptide with inline modifications, e.g., 'K.N[+203.0794]GTR.G', 'NC(+57.02)GVN(+1913.68)C', 'YN(N-Glycosylation)GTK', or 'LQT[O-linked glycosylation:H1N1A2 on X]LAL'
                 brackets: str = '[]' # delimiters around each modification
                 ) -> list[tuple[int, str]]: # (1-based peptide position, modification text) of every glycan, N- to C-terminal
    "Locates the glycans in a modified peptide, named as such or as a mass of at least 130 Da on N, S, or T that is no isobaric tag"
    o, c = map(re.escape, brackets)
    pos, res, out = 0, '', []
    for tok in re.findall(rf'{o}[^{c}]*{c}|[A-Za-z]', re.sub(r'^[A-Z-]\.|\.[A-Z-]$', '', peptide)):
        if len(tok) == 1:
            pos, res = pos + 1, tok.upper()
        elif re.search(r'glycan|glycosyl', tok, flags = re.IGNORECASE) or (pos and (m := re.fullmatch(r'\+?(\d+\.?\d*)', tok[1:-1])) and _glycan_mass(res, float(m[1]))):
            out.append((pos, tok[1:-1]))
    return out


def _pair_sites(start: int, # protein position of the peptide's first residue
                positions: list[int], # 1-based peptide positions of the glycans
                comps: list[str] # one composition per glycan, or one total composition
                ) -> list[tuple[str, str]]: # (protein site, composition) per glycan, or one joint 'site1+site2' entry with the total composition if they cannot be told apart
    "Places the glycans of a glycopeptide onto protein sites"
    sites = [str(int(start) + p - 1) for p in positions]
    if len(sites) == len(comps):
        return list(zip(sites, comps))
    return [('+'.join(sites), ','.join(comps))] if sites else []


def _run_names(files: list[str | Path | pd.DataFrame] # input files or DataFrames
               ) -> list[str]: # run name per input
    "Names each input that holds one run by its file name, with its folder in front where files share a name (ctrl/glycans.csv, case/glycans.csv), and a DataFrame as sample<n>"
    stems = [f'sample{fi + 1}' if isinstance(f, pd.DataFrame) else Path(f).stem for fi, f in enumerate(files)]
    return [f'{Path(f).parent.name}_{s}' if stems.count(s) > 1 and not isinstance(f, pd.DataFrame) else s for f, s in zip(files, stems)]


def _abundance_matrix(df: pd.DataFrame, # columns run, ion, label, and value, one row per measurement
                      sample_map: dict[str, str] | Callable[[str], str] | None, # renames runs into samples
                      groups: dict[str, str] | None = None # group per run, kept as the frame's contrasts
                      ) -> GlycoDataFrame: # labels in the first column, samples as columns
    "Keeps one abundance per ion and run, then sums ions and runs into labels and samples"
    rename = sample_map if callable(sample_map) else lambda r: (sample_map or {}).get(r, r)
    df = df.groupby(['run', 'ion', df.columns[2]], sort = False)['value'].max().reset_index()  # an ion has one abundance per run, so its repeated matches must not add up
    df['run'] = df['run'].map(rename)
    out = df.pivot_table(index = df.columns[2], columns = 'run', values = 'value', aggfunc = 'sum', fill_value = 0)[list(dict.fromkeys(df['run']))]
    return GlycoDataFrame(out.rename_axis(columns = None).reset_index(), contrasts = {rename(r): g for r, g in (groups or {}).items()} or None)


def read_glycoproteomics(files: str | Path | pd.DataFrame | list[str | Path | pd.DataFrame], # result file(s) of FragPipe/MSFragger-Glyco or O-Pair (psm.tsv), pGlyco3 or pGlycoQuant, Byonic or Byologic, GlycReSoft, MetaMorpheus O-Pair (.psmtsv), Glyco-Decipher, StrucGP, or PEAKS GlycanFinder
                         max_q: float = 0.01, # highest q-value/FDR a match may have, wherever the engine reports one
                         sample_map: dict[str, str] | Callable[[str], str] | None = None # renames runs (raw files, or FragPipe experiments) into samples; runs sharing a sample, e.g., fractions, are summed
                         ) -> GlycoDataFrame: # glycoforms as 'protein_site_composition' in column 'ID' ('protein_site_sequence' for StrucGP structure codes), samples as columns; ready for get_differential_expression(glycoproteomics = True) or get_glycoshift_per_site
    "Reads the native output of glycoproteomics search engines into a site-specific glycoform x sample table, quantified by intensity where the engine reports one and by spectral counts otherwise"
    recs, skipped = [], 0  # (run, ion, protein, site, composition, abundance); an ion has one abundance per run, so its repeated matches must not add up
    files = [files] if isinstance(files, (str, Path, pd.DataFrame)) else list(files)  # a list, or any iterable such as Path.glob('*/psm.tsv')
    for fi, (f, stem) in enumerate(zip(files, _run_names(files))):
        frames = [f] if isinstance(f, pd.DataFrame) else _read_table(f)
        for df in frames:
            sig, c = _gp_engine(df)
            if sig is not None:
                break
        else:
            raise ValueError(f"'{stem}' is not a recognized search engine output; expected FragPipe/MSFragger-Glyco or O-Pair, pGlyco3 or pGlycoQuant, Byonic or Byologic, GlycReSoft, MetaMorpheus O-Pair, Glyco-Decipher, StrucGP, or PEAKS GlycanFinder, but its columns are {list(frames[0].columns)[:8]}")
        num = lambda k: pd.to_numeric(df[c[k]], errors = 'coerce')
        col = lambda k: df[c[k]] if k in c else pd.Series([None] * len(df), index = df.index, dtype = object)
        keep = pd.Series(True, index = df.index)
        for k in ['glycanq-value', 'qvalue', 'q_value', 'totalfdr', 'glycanfdr'] + ([] if 'totalfdr' in c else ['peptidefdr']):
            if k in c:
                keep &= ~(num(k) > max_q)  # an empty q-value, e.g., without a glycan FDR step, does not reject the match
        for k, ok in (('isdecoy', lambda s: s.str.lower() != 'true'), ('glydecoy', lambda s: ~s.isin(('1', '1.0'))), ('pepdecoy', lambda s: ~s.isin(('1', '1.0'))),
                      ('decoy/contaminant/target', lambda s: ~s.str.contains('D')), ('is_best_match', lambda s: s.str.lower() == 'true'), ('quanusage', lambda s: s.str.lower() != 'false')):  # FragPipe marks PSMs failing its reporter ion filters (purity, minimum intensity) Quan Usage 'false'
            if k in c:
                keep &= ok(df[c[k]].astype(str).str.strip())
        df = df[keep]
        if sig == 'byologic':  # Row# nests all samples (70), one sample (70.1), and single matches (70.1.1); per-sample rows exist unless the peptide was seen once
            rid = col('row#').astype(str)
            depth, lvl1 = rid.str.count(r'\.') + 1, rid.str.split('.').str[0]
            has2 = depth.eq(2).groupby(lvl1).transform('any')
            df = df[(has2 & depth.eq(2)) | (~has2 & depth.eq(1))]
        runs, vals, wide = [stem] * len(df), None, {}
        if sig == 'fragpipe':
            runs = [(re.split(r'[\\/]', str(fl))[-2] if re.search(r'[\\/]', str(fl)) else '', str(s).rsplit('.', 3)[0]) for s, fl in zip(col('spectrum'), col('spectrumfile'))]  # (experiment, raw file)
            prots = col('proteinid') if 'proteinid' in c else col('protein')
            ions = [f'{r}/{p}/{z}/{g}' for (_, r), p, z, g in zip(runs, col('modifiedpeptide'), col('charge'), col('totalglycancomposition'))]  # an experiment's raw files are fractions, so an ion is collapsed per raw file and only then summed into its experiment
            pairs = []
            for g, am, sp, start in zip(col('totalglycancomposition'), col('assignedmodifications'), col('siteprobabilities'), col('proteinstart')):
                if not isinstance(g, str) or g.startswith('Decoy'):
                    pairs.append(None)
                    continue
                comp, _, mass = g.partition('%')
                loc = re.findall(r'\[(\d+),([^,\]]+),', str(sp))  # O-Pair: [peptide position,composition,probability] for each glycan
                mods = re.findall(r'(\d+)([A-Z])\((-?[\d.]+)\)', str(am))  # '3N(2350.8303),4C(57.0215)'
                pos = [int(p) for p, _ in loc] or [int(p) for p, _, m in mods if mass.strip() and abs(float(m) - float(mass)) < 0.05] or [int(p) for p, r, m in mods if _glycan_mass(r, float(m))]
                pairs.append(_pair_sites(int(float(start)), pos, [x for _, x in loc] or [comp.strip()]) if pd.notna(start) else [])
            vals = num('intensity') if 'intensity' in c and num('intensity').gt(0).any() else None
            if 'quanusage' in c:  # TMT or iTRAQ: 'Quan Usage', then one reporter ion column per channel, named by the plex's annotation file or else by the channel ('126', '127N'), which then gets its plex (experiment folder) in front
                plex = runs[0][0] if runs else ''
                wide = {k: f'{plex}_{k}' if plex and re.fullmatch(r'1[1-3]\d[NC]?', str(k)) else str(k) for k in list(df.columns)[list(df.columns).index(c['quanusage']) + 1:] if k not in (
                    'Parent Scan Number', 'Apex Retention Time', 'Apex Scan Number', 'Retention Time Start', 'Retention Time End', 'Retention Time FWHM', 'Traced Scans')}  # IonQuant's per-PSM columns
                ions = list(col('spectrum'))  # reporter ions come from each spectrum, so every PSM of a glycoform adds up
        elif sig == 'pglyco':
            runs, prots = list(col('rawname')) if 'rawname' in c else runs, [p.split(';')[0] if isinstance(p, str) else p for p in col('proteins')]  # an empty protein cell, e.g., of merged cells in a published table, must not become the accession 'nan'
            pairs = [[(re.sub(r'\.0+$', '', str(s).split(';')[0]), g)] if isinstance(g, str) and pd.notna(s) else [] for s, g in zip(col('prosites'), col('glycancomposition'))]
            wide = {k: m[1] for k in df.columns if (m := re.fullmatch(r'ReportIon\s*\((.+)\)', str(k)))}  # pGlycoQuant TMT: one reporter ion column per channel, named by its m/z
            ions = [f'{r}/{s}' for r, s in zip(runs, col('glyspec'))] if wide else [f'{p}/{m}/{z}/{g}' for p, m, z, g in zip(col('peptide'), col('mod'), col('charge'), col('glycancomposition'))]  # reporter ions come from each spectrum, so every PSM adds up, whereas an LFQ intensity belongs to the ion and is repeated on each of its PSMs
            vals = num('monoarea') if 'monoarea' in c and num('monoarea').gt(0).any() else None
        elif sig in ('byonic', 'byologic'):
            pep, comp, start = [c[next(k for k in c if k.startswith(x))] for x in (('sequence', 'glycans', 'startaa') if sig == 'byologic' else ('peptide', 'glycansnhfagna' if 'glycansnhfagna' in c else 'composition', 'startingposition' if 'startingposition' in c else 'position'))]
            prots = col('proteinname')
            pos = [[int(p) for p, t in re.findall(r'[A-Z](\d+)\(([^)]*)\)', str(ms)) if 'glycan' in t.lower()] for ms in col('mod.summary')] if sig == 'byologic' else [
                [p for p, _ in _glycan_mods(str(s))] for s in df[pep]]
            pairs = [(_pair_sites(int(float(st)), p, [x.strip() for x in g.split(',')]) if pd.notna(st) else []) if isinstance(g, str) else None for p, g, st in zip(pos, df[comp], df[start])]
            ions = [f'{r}/{s}/{z}/{g}' for r, s, z, g in zip(col('row#'), df[pep], col('z'), df[comp])]
            if sig == 'byologic':
                runs, vals = [str(a).split(';')[0].strip() for a in col('msaliasname')] if 'msaliasname' in c else runs, num('xicareasummed')
            elif 'comment' in c:
                runs = [str(m).split('.ScanId')[0] if '.ScanId' in str(m) else stem for m in col('comment')]
        elif sig == 'glycresoft':
            runs = list(col('analysis')) if 'analysis' in c else runs
            prots, pairs = col('protein_name'), []
            for gp, start in zip(col('glycopeptide'), col('peptide_start')):
                seq, _, g = str(gp).partition('{')
                pairs.append((_pair_sites(int(float(start)) + 1, [p for p, _ in _glycan_mods(seq, brackets = '()')], ['{' + g]) if pd.notna(start) else []) if g else None)  # peptide_start is 0-based
            ions = [f'{g}/{z}' for g, z in zip(col('glycopeptide'), col('charge'))]
            vals = next((num(k) for k in ('total_signal', 'precursor_abundance') if k in c), None)
        elif sig == 'metamorpheus':
            runs, prots = [re.split(r'[\\/]', str(fl))[-1].rsplit('.', 1)[0] for fl in col('filename')] if 'filename' in c else runs, col('proteinaccession')
            mods = [_glycan_mods(str(s)) for s in col('fullsequence')]  # 'LQT[O-linked glycosylation:H1N1A2 on X]LAL'
            pairs = [_pair_sites(int(re.search(r'\d+', str(se))[0]), [p for p, _ in m], [t.split(':', 1)[-1].rsplit(' on ', 1)[0] for _, t in m]) for m, se in zip(mods, col('startandendresiduesinprotein'))]
            ions = list(col('fullsequence'))
        elif sig in ('decipher', 'strucgp'):
            runs = list(col(k)) if (k := 'file' if sig == 'decipher' else 'filename') in c else runs  # published tables often drop the file column, so a DataFrame or file without it is one sample
            prots = [re.split(r'[;,]', p)[0] if isinstance(p, str) else p for p in col('protein' if sig == 'decipher' else 'proteinid')]
            pairs = [[(re.sub(r'\.0+$', '', re.split(r'[;,]', str(s))[0]), (sc.strip() if isinstance(sc, str) and _STRUCGP_CODE.fullmatch(sc.strip()) else re.sub(r'S(?=\d)', 'A', g.split('+')[0])) if sig == 'strucgp' else g)] if isinstance(g, str) and pd.notna(s) else [] for s, g, sc in
                     zip(col('glycosite' if sig == 'decipher' else 'glycosite_position'), col('glycancomposition'), col('structure_coding'))]  # StrucGP's structure code becomes the full sequence wherever it is given; otherwise its composition, in which StrucGP writes NeuAc as S and appends adducts like '+Ammonium(+17)'
            ions = list(col('peptide'))
        elif sig == 'glycanfinder':
            prots, ends = col('proteinaccession'), list(df.columns).index(c['sampleprofile(ratio)']) if 'sampleprofile(ratio)' in c else len(df.columns)
            pairs = [_pair_sites(int(float(st)), [p for p, _ in _glycan_mods(str(s), brackets = '()')], str(g).split(';')) if pd.notna(st) else [] for s, g, st in zip(col('peptide'), col('glycan'), col('start'))]
            ions, wide = [f'{s}/{g}' for s, g in zip(col('peptide'), col('glycan'))], {k: k[5:] for k in df.columns[:ends] if str(k).startswith('Area ')}  # the per-group 'Area' columns follow the sample profile
        else:  # Glyco-Decipher site table: 'P00738@211;P00739@153' maps to the first protein, 'O43866@226;O43866@229' to the ambiguous site '226/229'
            sites = [str(s).split(';') for s in col('site')]
            prots = [s[0].split('@')[0] if '@' in s[0] else None for s in sites]
            pairs = [[('/'.join(x.split('@')[-1] for x in s if x.split('@')[0] == p), g)] if isinstance(g, str) else None for s, p, g in zip(sites, prots, col('glycan'))]
            ions, wide = list(range(len(df))), {k: k for k in df.columns[2:]}
        wide = wide or {k: m[1] for k in df.columns if (m := re.fullmatch(r'Intensity\((.+)\)', str(k)))}  # pGlycoQuant appends one column per run to pGlyco3 or Byonic results
        abund = df[list(wide)].apply(pd.to_numeric, errors = 'coerce').fillna(0).to_numpy() if wide else None
        vals = None if vals is None else vals.to_numpy()
        for i, (prot, pr, ion) in enumerate(zip(prots, pairs, ions)):
            if pr is None or (isinstance(prot, str) and _DECOY.match(prot)):  # no glycan, or a decoy
                continue
            if not pr or pd.isna(prot):
                skipped += 1
                continue
            obs = zip(wide.values(), abund[i]) if wide else [(runs[i], 1.0 if vals is None else vals[i])]
            recs.extend((run, ion if wide or vals is not None else (fi, i), _accession(prot), site, comp, v) for run, v in obs if v > 0 for site, comp in pr)
    if skipped:
        warnings.warn(f"{skipped} matches had no protein or glycosylation site to place them on and were skipped.", stacklevel = 2)
    if not recs:
        raise ValueError("No glycopeptide passed the filters; check that the files hold glycopeptide matches and that max_q is not too strict.")
    df = pd.DataFrame(recs, columns = ['run', 'ion', 'protein', 'site', 'comp', 'value'])
    exps = {r[0] for r in df['run'] if isinstance(r, tuple)}
    df['run'] = [(r[0] if len(exps) > 1 else r[1]) if isinstance(r, tuple) else r for r in df['run']]  # a FragPipe experiment is a sample, unless the search defined no experiments
    canon = {}
    for g in df['comp'].unique():
        try:
            canon[g] = canonicalize_iupac(g) if _STRUCGP_CODE.fullmatch(g) else None if re.search(r'[+\[]', _COMP_ADDUCT.sub('', g)) else canonicalize_composition(g, as_string = True) or None  # a residual mass delta such as '+225.06' has no composition, an adduct such as '+Na' or 'Na(1)' is dropped from it
        except ValueError:
            canon[g] = None
    if bad := [g for g, v in canon.items() if v is None]:
        warnings.warn(f"Skipped glycoforms with {len(bad)} unparseable compositions, e.g., {bad[:3]}.", stacklevel = 2)
    df = df.assign(ID = df['protein'] + '_' + df['site'] + '_' + df['comp'].map(canon)).dropna(subset = ['ID'])
    return _abundance_matrix(df[['run', 'ion', 'ID', 'value']], sample_map)  # the protein_site_composition labels are what switches the analysis functions to the per-glycosite analysis


# glycosphingolipid headgroups as lipidomics software names them (LIPID MAPS shorthand, MS-DIAL, LipidSearch, Lipidyzer), with human sialic acid; Hex3Cer's third hexose is left open (Gb3, iGb3)
_GSL_GLYCANS = {'HEXCER': 'Hex1Cer', 'HEX1CER': 'Hex1Cer', 'HCER': 'Hex1Cer', 'GLCCER': 'Glc1Cer', 'GALCER': 'Gal1Cer', 'HEX2CER': 'Gal(b1-4)Glc1Cer', 'LACCER': 'Gal(b1-4)Glc1Cer', 'LCER': 'Gal(b1-4)Glc1Cer',
                'HEX3CER': 'Hex(?1-?)Gal(b1-4)Glc1Cer', 'SHEXCER': 'Gal3S1Cer', 'SULFATIDE': 'Gal3S1Cer', 'GB3': 'Gal(a1-4)Gal(b1-4)Glc1Cer', 'GB4': 'GalNAc(b1-3)Gal(a1-4)Gal(b1-4)Glc1Cer',
                'GM3': 'Neu5Ac(a2-3)Gal(b1-4)Glc1Cer', 'GM2': 'GalNAc(b1-4)[Neu5Ac(a2-3)]Gal(b1-4)Glc1Cer', 'GM1': 'Gal(b1-3)GalNAc(b1-4)[Neu5Ac(a2-3)]Gal(b1-4)Glc1Cer',
                'GM1A': 'Gal(b1-3)GalNAc(b1-4)[Neu5Ac(a2-3)]Gal(b1-4)Glc1Cer', 'GM1B': 'Neu5Ac(a2-3)Gal(b1-3)GalNAc(b1-4)Gal(b1-4)Glc1Cer', 'GD3': 'Neu5Ac(a2-8)Neu5Ac(a2-3)Gal(b1-4)Glc1Cer',
                'GD2': 'GalNAc(b1-4)[Neu5Ac(a2-8)Neu5Ac(a2-3)]Gal(b1-4)Glc1Cer', 'GD1A': 'Neu5Ac(a2-3)Gal(b1-3)GalNAc(b1-4)[Neu5Ac(a2-3)]Gal(b1-4)Glc1Cer',
                'GD1B': 'Gal(b1-3)GalNAc(b1-4)[Neu5Ac(a2-8)Neu5Ac(a2-3)]Gal(b1-4)Glc1Cer', 'GT1B': 'Neu5Ac(a2-3)Gal(b1-3)GalNAc(b1-4)[Neu5Ac(a2-8)Neu5Ac(a2-3)]Gal(b1-4)Glc1Cer',
                'GQ1B': 'Neu5Ac(a2-8)Neu5Ac(a2-3)Gal(b1-3)GalNAc(b1-4)[Neu5Ac(a2-8)Neu5Ac(a2-3)]Gal(b1-4)Glc1Cer'}
# a headgroup, an optional MS-DIAL subclass (_NS, _NDS), then the ceramide (d18:1/16:0, 34:1;O2, (d36:1), Lipidyzer's (24:1)) and whatever adduct follows
_GSL = re.compile(r'\s*(Hex[1-3]?Cer|GlcCer|GalCer|LacCer|SHexCer|Sulfatide|HCER|LCER|GM1[ab]?|GM[23]|GD1[ab]|GD[23]|GT1b|GQ1b|Gb[34])(?:_[A-Z]+)?[\s(]*[dtm]?\d{1,2}:\d', re.IGNORECASE)


def _glycomics_layout(df: pd.DataFrame # a glycomics table as read from file
                      ) -> tuple: # (tool, or None for a plain glycans x samples table; lowercased header map; molecule, replicate, and area column; {pivoted area column: replicate})
    "Recognizes the export of a glycomics tool by its columns"
    c = {re.sub(r'\s+', '', str(k)).lower(): k for k in df.columns}
    mol = next((c[k] for k in ('moleculename', 'molecule', 'peptide', 'peptidesequence') if k in c), None)
    rep = next((c[k] for k in ('replicatename', 'replicate') if k in c), None)
    area = next((c[k] for k in ('totalareams1', 'totalarea', 'area') if k in c), None)
    wide = {k: m[1] for k in df.columns if (m := re.fullmatch(r'(.+) Total Area MS1', str(k)))} or {k: m[1] for k in df.columns if (m := re.fullmatch(r'(.+) Total Area', str(k)))}
    tool = 'glycresoft' if all(k in c for k in ('composition', 'total_signal', 'neutral_mass')) else 'candycrunch' if (
                                                                                                                                  'num_spectra' in c or 'n_files_ms2' in c) and 'composition' in c else 'compound_discoverer' if 'name' in c and any(
        str(k).startswith('Area: ') for k in df.columns) else 'skyline' if mol is not None and (
                (rep is not None and area is not None) or wide) else 'glycogenius' if len(df) > 0 and str(
        df.columns[0]) == 'Sample' and str(df.iloc[0, 0]) == 'Group' else None
    return tool, c, mol, rep, area, wide


def read_glycomics(files: str | Path | pd.DataFrame | list[str | Path | pd.DataFrame], # export(s) of Skyline (report, long or pivoted by replicate), LaCyTools or MassyTools (Summary.txt), GlycoWorkbench (.gwp workspace or .gwa annotated peak list), Thermo Compound Discoverer, GlycoGenius, GlyHunter, GlycReSoft, or CandyCrunch, or any table with glycans in its first column and samples in the others; glycosphingolipids named as lipidomics software names them (GM3 d18:1/16:0, HexCer_NS d18:1/24:1, Hex2Cer 34:1;O2) are read as their glycans, summed over ceramides
                   sample_map: dict[str, str] | Callable[[str], str] | None = None,  # renames runs (raw files, Skyline replicates, or sample columns) into samples; runs sharing a sample, e.g., fractions, are summed
                   qc: dict[str, float] = {'ppm': 20, 'ipq': 0.2, 'sn': 9}, # GlycoDash's limits for a LaCyTools charge state to pass in a run: absolute mass error (ppm), isotopic pattern quality (at most), S/N (at least); {} keeps every analyte
                   min_pass: float = 0.25 # fraction of runs a LaCyTools charge state has to pass qc in to be kept in every run, as LaCyTools reports each listed analyte in each run, noise included; GlycoDash's analyte curation uses 0.8 per biological group, after removing failed spectra
                   ) -> GlycoDataFrame: # glycans (IUPAC-condensed, canonical composition, or the label as written if it is neither) in column 'glycan', samples as columns; ready for get_differential_expression and the rest of glycowork
    "Reads the native output of glycomics software into a glycan x sample table, quantified by the tool's abundance where it reports one and by feature counts otherwise"
    recs, groups, checks, units = [], {}, {}, {}  # (run, ion, label, abundance, S is NeuAc); Skyline, LaCyTools, MassyTools, and GlycoGenius write NeuAc as S, which glycowork reads as sulfate
    files = [files] if isinstance(files, (str, Path, pd.DataFrame)) else list(files)  # a list, or any iterable such as Path.glob('*.csv')
    for fi, (f, stem) in enumerate(zip(files, _run_names(files))):
        ext = '' if isinstance(f, pd.DataFrame) else Path(f).suffix.lower()
        if ext in ('.gwp', '.gwa'):  # GlycoWorkbench workspace, one top-level Scan per sample, or one annotated peak list; MS/MS scans are skipped
            root = ET.parse(f).getroot()
            scans = [root] if root.tag == 'AnnotatedPeakList' else [s for s in root.findall('Scan') if s.get('is_msms') != 'true']
            for si, scan in enumerate(scans):
                for pa in scan.findall(('' if scan is root else 'AnnotatedPeakList/') + 'Annotations/PeakAnnotationCollection/PeakAnnotation'):
                    peak, frag = pa.find('Peak'), pa.find('Annotation/FragmentEntry')
                    g = '' if frag is None else frag.get('fragment', '')
                    if g and not re.search(r'/#[a-km-z]cleavage', g) and float(peak.get('intensity', 0)) > 0:  # the structure on this peak, possibly with a labile loss (/#lcleavage), e.g., of a sulfate; any other cleavage makes it an MS/MS fragment
                        recs.append((scan.get('name') or (stem if len(scans) == 1 else f'{stem}_{si + 1}'), (fi, peak.get('mz_ratio')), g.replace('/#lcleavage', ''), float(peak.get('intensity')), False))
            continue
        lines = Path(f).read_text(encoding = 'utf-8-sig', errors = 'replace').splitlines() if ext == '.txt' else []
        heads = [i for i, l in enumerate(lines) if re.match(r'Absolute Intensity \(Background Subtracted|Analyte Area - Background Area\tCalibrated|Background Subtracted Intensities \(Area\)\t', l)] or [
            i for i, l in enumerate(lines) if _SUMMARY_HEAD.match(l)]
        blocks, qcs = {}, {(m[1], m[2]): i for i, l in enumerate(lines) if (m := _LACY_QC.match(l))}
        for h in heads + list(qcs.values()):  # LaCyTools: one block per charge state, analytes in the header and samples as rows below 'Fraction' and the mass row; MassyTools 1: 'Calibrated' and the mass row in front; MassyTools 2: each analyte and charge as 'H5N4F1E1 [1z]', then an 'Exact m/z' row
            head = [re.sub(r'\s*\[\d+z\]$', '', k) for k in lines[h].split('\t')]
            off = 2 if head[1:2] == ['Calibrated'] else 1
            labs, rows = [lab.strip() for lab in head[off:]], {}
            for l in lines[h + 1:]:
                row = l.split('\t')
                if not l.strip():
                    break
                if row[0] != '' and not row[0].startswith(('Exact mass', 'Exact m/z', 'Monoisotopic mass')):
                    rows[row[0]] = pd.to_numeric(pd.Series(row[off:off + len(labs)]), errors = 'coerce').reindex(range(len(labs))).to_numpy()
            blocks[h] = (labs, rows.pop('Fraction', [1] * len(labs)), rows)  # the share of its isotope envelope LaCyTools integrated differs between analytes, so GlycoDash divides by it
        for h in heads:
            labs, frac, rows = blocks[h]
            z = re.search(r'(\d+)\+\)', lines[h])
            q = {kind: np.array([blocks[i][2].get(r, [np.nan] * len(labs)) for r in rows]) for (kind, zq), i in qcs.items() if z and zq == z[1]}
            if q and qc and rows:
                ok = np.ones((len(rows), len(labs)), dtype = bool)
                for kind, key in (('Mass Accuracy [ppm]', 'ppm'), ('Isotopic Pattern Quality', 'ipq'), ('S/N', 'sn')):
                    if kind in q and key in qc:
                        ok &= (q[kind] >= qc[key]) if key == 'sn' else (np.abs(q[kind]) <= qc[key])  # a missing value fails, as in GlycoDash
                for j, lab in enumerate(labs):
                    checks.setdefault((lab, z[1]), []).extend(ok[:, j])
                    units[(fi, h, j)] = (lab, z[1])
            recs.extend((r, (fi, h, j), lab, v / (frac[j] if frac[j] > 0 else 1), True) for r, vals in rows.items() for j, (lab, v) in enumerate(zip(labs, vals)) if lab and v > 0)
        if heads:
            continue
        frames = [f] if isinstance(f, pd.DataFrame) else _read_table(f)
        df = next((d for d in frames if _glycomics_layout(d)[0] is not None), frames[0])  # a tool's export may sit on any sheet of a workbook, a plain table of glycans x samples is its first sheet
        tool, c, mol, rep, area, wide = _glycomics_layout(df)
        num = lambda k: pd.to_numeric(df[k].astype(str).str.lstrip('*'), errors = 'coerce')  # Skyline writes some values as '*2.4246E+7'
        if tool == 'glycresoft':  # one analysis per file; unidentified chromatograms have the composition 'None'
            recs.extend((stem, (fi, i), g, v, False) for i, (g, v) in enumerate(zip(df[c['composition']], num(c['total_signal']))) if isinstance(g, str) and g != 'None' and v > 0)
        elif tool == 'candycrunch':  # one table per raw file, or a wrap_inference_batch feature table with one abundance column per raw file (each with its evidence_<file> column); peaks without a predicted structure keep their composition
            tops = df[c['top1_pred']] if 'top1_pred' in c else pd.Series(None, index = df.index, dtype = object)
            runs = [(k, num(k)) for k in df.columns if f'evidence_{k}' in df.columns] or [
                (stem, num(c['rel_abundance']) if 'rel_abundance' in c else pd.Series(1.0, index = df.index))]
            recs.extend((run, (fi, i), t if isinstance(t, str) and t.strip() else canonicalize_composition(
                ast.literal_eval(g) if isinstance(g, str) else g, as_string = True), v, False)
                        for run, vals in runs for i, (t, g, v) in enumerate(zip(tops, df[c['composition']], vals)) if
                        v > 0 and (isinstance(t, str) or isinstance(g, (str, dict))))
        elif tool == 'compound_discoverer':  # Thermo Compound Discoverer compounds table: one 'Area: <file>.raw (F<n>)' column per raw file, unnamed features skipped
            for k in df.columns:
                if m := re.fullmatch(r'Area: (.+?)(?:\.raw)?(?: \(F\d+\))?', str(k)):
                    recs.extend((m[1], (fi, i), g, v, False) for i, (g, v) in enumerate(zip(df[c['name']], num(k))) if isinstance(g, str) and g.strip() and v > 0)
        elif tool == 'skyline':  # one row per precursor (or transition) and replicate, or replicates pivoted into '<replicate> Total Area' columns
            ions = list(df[[c[k] for k in ('proteinname', 'protein', 'moleculelistname', 'moleculenote', 'precursorcharge', 'precursoradduct', 'precursormz', 'isotopelabeltype') if k in c] + [mol]].fillna('').astype(str).agg('/'.join, axis = 1))
            ions = [(fi, i) for i in range(len(df))] if not wide and area == c.get('area') else ions  # transition areas add up, a precursor's Total Area repeats on each of its transitions
            study = ~df[c['sampletype']].astype(str).str.strip().str.lower().isin(('blank', 'double blank', 'solvent', 'quality control', 'standard')) if 'sampletype' in c and not wide else pd.Series(True, index = df.index)  # Skyline's Sample Type marks the replicates that are not study samples
            if not study.all():
                warnings.warn(f"{df.loc[~study, rep].nunique()} Skyline replicates marked as blank, solvent, quality control, or standard were left out, e.g., {sorted(df.loc[~study, rep].astype(str).unique())[:3]}.", stacklevel = 2)
            for k, run in (wide.items() if wide else [(area, None)]):
                recs.extend((r if run is None else run, ion, g, v, True) for r, ion, g, v, ok in zip(df[rep] if run is None else [run] * len(df), ions, df[mol], num(k), study) if ok and isinstance(g, str) and v > 0)
        else:  # GlyHunter, GlycoGenius, a CandyCrunch batch, or glycowork's own layout: glycans in the first column, one column per sample
            gg = tool == 'glycogenius'  # a 'Group' row under the sample names, and '_<RT>' after each glycan of the peak-separated table
            if gg:
                groups |= {k: g for k, g in zip(df.columns[1:], df.iloc[0, 1:]) if isinstance(g, str) and g != 'Ungrouped'}
                df = df.iloc[1:]
            labs = [re.sub(r'_\d+(\.\d+)?$', '', g) if gg and isinstance(g, str) else g for g in df.iloc[:, 0]]
            for k in df.columns[1:]:
                recs.extend((k, (fi, i), g, v, gg) for i, (g, v) in enumerate(zip(labs, num(k))) if isinstance(g, str) and v > 0)
    if failed := {u for u, v in checks.items() if np.mean(v) < min_pass}:  # over all runs of all files, so the plates of one study keep the same analytes
        recs = [r for r in recs if units.get(r[1]) not in failed]
        warnings.warn(f"{len(failed)} LaCyTools charge states passed qc in fewer than {min_pass:.0%} of runs and were removed, e.g., {sorted(failed)[:3]}.", stacklevel = 2)
    if not recs:
        raise ValueError("No abundances found; expected a Skyline report, a LaCyTools or MassyTools Summary.txt, a GlycoWorkbench workspace, or a Compound Discoverer, GlycoGenius, GlyHunter, GlycReSoft, or CandyCrunch table with glycans and positive abundances, or any table with glycans in its first column (not the index; use df.reset_index()) and samples in the others.")
    df = pd.DataFrame(recs, columns = ['run', 'ion', 'label', 'value', 'sia'])
    canon = {}
    for g, sia in df[['label', 'sia']].drop_duplicates().itertuples(index = False):
        gsl = _GSL.match(g)  # a glycosphingolipid of a lipidomics export: its glycan, so that its ceramide species are summed below
        lab = _GSL_GLYCANS[gsl[1].upper()] if gsl else g.strip()
        is_comp = is_composition(lab)
        try:
            iupac = canonicalize_iupac(lab)
            glycan_to_composition(iupac)  # raises on what canonicalize_iupac hands back unread, e.g., 'H5N4' or 'Internal Standard'; Kdo, heptoses, or QuiNAc are glycans without a composition ({})
            iupac = None if is_comp and '(' not in iupac else iupac
        except Exception:
            iupac = None
        try:
            comp = canonicalize_composition(re.sub(r'S(?=\d)', 'A', g.strip()) if sia else g.strip(), as_string = True, strict = True) if iupac is None and is_comp else ''  # strict: a label like 'IgGI1H5N4F1' parses into a nonsense residue
        except ValueError:
            comp = ''
        if not (iupac or comp) and (pep := re.fullmatch(r'([A-Za-z]+?)1((?:[A-Z][a-z]?\d+)+)', g.strip())):  # a LaCyTools glycopeptide, one peptide block and the glycan, e.g., IgGI1H5N4F1S1 (IgGI is EEQYNSTYR of IgG1); the block becomes the protein and its count the site, so each cluster is closed and traited on its own, as in GlycoDash
            try:
                comp = f'{pep[1]}_1_' + canonicalize_composition(re.sub(r'S(?=\d)', 'A', pep[2]) if sia else pep[2], as_string = True, strict = True)
            except ValueError:
                pass
        canon[(g, sia)] = iupac or comp or None
    if raw := [g for (g, _), v in canon.items() if v is None]:
        warnings.warn(f"{len(raw)} labels are neither a glycan sequence nor a composition glycowork can read and were kept as written, e.g., {raw[:3]}.", stacklevel = 2)
    df['glycan'] = [canon[(g, s)] or g.strip() for g, s in zip(df['label'], df['sia'])]
    if merged := {k: v for k, v in df.groupby('glycan')['label'].unique().items() if len(v) > 1}:
        warnings.warn(f"{len(merged)} glycans were written in several ways and summed, e.g., {list(merged.items())[0][1].tolist()} into '{list(merged)[0]}'; adducts (Na, NH4, [M+Na]+) are dropped, and of sialic acid derivatives only the linkage they tell is kept (E, ethyl ester or dimethylamide, as a2-6; L, lactone or amide, as a2-3).", stacklevel = 2)
    return _abundance_matrix(df[['run', 'ion', 'glycan', 'value']], sample_map, groups = groups)


def read_abundances(file: str | Path # glycowork table (.csv with commas or semicolons, .tsv, .xlsx) with glycans in the first column and samples in the others, the export of a tool that read_glycomics or read_glycoproteomics supports, or the name of a dataset shipped with glycowork (e.g., 'human_serum_bacteremia_N_PMID33535571')
                    ) -> pd.DataFrame: # the table as written, the read_glycomics or read_glycoproteomics output for a recognized tool export, or a copy of the shipped dataset with its contrasts
    "Reads the input of glycowork's analysis functions: shipped dataset names load with their contrasts, tool exports recognized by their columns go through read_glycomics or read_glycoproteomics, every other table is read exactly as written"
    if not Path(file).exists():
        for loader in (glycomics_data_loader, glycoproteomics_data_loader, lectin_array_data_loader):
            if (name := Path(file).stem.removeprefix(loader.prefix)) in dir(loader):
                return getattr(loader, name).copy()
    ext = Path(file).suffix.lower()
    if ext in ('.gwp', '.gwa') or (ext == '.txt' and any(_SUMMARY_HEAD.match(l) for l in Path(file).read_text(encoding = 'utf-8-sig', errors = 'replace').splitlines())):
        return read_glycomics(file)
    sheets = _read_table(file, dtype = None)
    for df in sheets:
        if _gp_engine(df)[0] is not None:
            return read_glycoproteomics(file)
        if _glycomics_layout(df)[0] is not None:
            return read_glycomics(file)
    return sheets[0]


def check_presence(glycan: str, # IUPAC-condensed glycan sequence
                   df: pd.DataFrame, # glycan dataframe where glycans are under colname
                   colname: str = 'glycan', # column name containing glycans
                   name: str | None = None, # name of species of interest
                   rank: str = 'Species', # column name for filtering
                   fast: bool = False # True uses precomputed glycan graphs
                   ) -> None:
    "checks whether glycan (of that species) is already present in dataset"
    if not isinstance(glycan, str) or any(p in glycan for p in ['RES', '=']):
        check_nomenclature(glycan)
        return
    lookup_col = 'graph' if fast else colname
    if lookup_col not in df.columns:
        raise KeyError(f"'{lookup_col}' is not a column of df; available columns are {df.columns.tolist()}")
    if name is not None:
        if rank not in df.columns:
            raise KeyError(f"'{rank}' is not a column of df; available columns are {df.columns.tolist()}")
        name = name.replace(" ", "_")
        df = df[df[rank] == name]
        if len(df) == 0:
            print(f"This is the best: {name} is not in dataset")
    ggraph = glycan_to_nxGraph(glycan) if fast else glycan
    check_all = [compare_glycans(ggraph, k) for k in df[lookup_col]]
    if any(check_all):
        print("Glycan already in dataset.")
    else:
        print("It's your lucky day, this glycan is new!")
