#!/usr/bin/env python
"""Regenerate the SKELETONS, ALDITOLS and ENANTIOMER tables of glycowork/motif/smiles.py.

This is the only place RDKit (and GlyLES, for its curated base structures) is needed: everything the
runtime would otherwise have to redo per call - ring perception, carbon numbering, rooted output -
happens here, once, and is baked into template strings. Run it after adding a monosaccharide, paste
the printed blocks into glycowork/motif/smiles.py, and run tests/test_smiles.py.

    pip install rdkit 'glyles~=1.2.3a0'
    python bin/build_smiles_templates.py > /tmp/tables.py

The base stereochemistry of the ring skeletons is read out of GlyLES' curated monosaccharide tables
(MIT, (c) Roman Joeres) rather than typed in by hand, and then checked: the aldose family is
re-derived independently from Fischer configurations off alpha-D-glucopyranose and must agree, the
alditols must reproduce the meso and numbering-reversal identities of their sugars, and inositol is
numbered off 1D-myo-inositol 1,4,5-trisphosphate.

Every skeleton is emitted as a SMILES rooted at the anomeric oxygen, with '{a}' for the anomeric tag,
'{r}' for the ring-closure digit and '{pN}' for the substituent at carbon N. The alpha and beta forms
are asserted to differ in nothing but that one tag, which is itself a check on the input structures.
"""
import re
import sys
from rdkit import Chem, RDLogger
from glyles.glycans.factory.factory_p import PyranoseFactory
from glyles.glycans.factory.factory_f import FuranoseFactory

RDLogger.DisableLog('rdApp.*')
PYRANOSE = PyranoseFactory._PyranoseFactory__monomers
FURANOSE = FuranoseFactory._FuranoseFactory__monomers
SPEC = {  # skeleton: (source table and key, edits applied to it)
    'Glc': ('P:GLC', {}), 'Gal': ('P:GAL', {}), 'Man': ('P:MAN', {}), 'Alt': ('P:ALT', {}), 'Gul': ('P:GUL', {}),
    'Ido': ('P:IDO', {}), 'Tal': ('P:TAL', {}), 'Qui': ('P:QUI', {}), 'Rha': ('P:RHA', {}), 'Fuc': ('P:FUC', {}),
    'Xyl': ('P:XYL', {}), 'Ara': ('P:ARA', {}), 'Rib': ('P:RIB', {}), 'Lyx': ('P:LYX', {}), 'Api': ('P:API', {}),
    'Fru': ('P:FRU', {}), 'Tag': ('P:TAG', {}), 'Sor': ('P:SOR', {}), 'Psi': ('P:PSI', {}), 'Neu': ('P:NEU', {}),
    'Kdn': ('P:KDN', {}), 'Kdo': ('P:KDO', {}), 'Ko': ('P:KO', {}), 'Dha': ('P:DHA', {}), 'Leg': ('P:LEG', {}),
    'Pse': ('P:PSE', {}), 'Aci': ('P:ACI', {}), 'Fus': ('P:FUS', {}), 'Bac': ('P:BAC', {}), 'Mur': ('P:MUR', {}),
    'Tyv': ('P:TYV', {}), 'Abe': ('P:ABE', {}), 'Par': ('P:PAR', {}), 'Col': ('P:COL', {}), 'Asc': ('P:ASC', {}),
    'Dig': ('P:DIG', {}), 'Oli': ('P:OLI', {}), 'Pau': ('P:PAU', {}), 'Yer': ('P:YER', {}), 'Aco': ('P:ACO', {}),
    'Erwiniose': ('P:ERWINIOSE', {}), 'Per': ('P:PER', {}), 'Vio': ('P:VIO', {}), 'Sed': ('P:SED', {}),
    'Glcf': ('F:GLC', {}), 'Galf': ('F:GAL', {}), 'Manf': ('F:MAN', {}), 'Fucf': ('F:FUC', {}), 'Araf': ('F:ARA', {}),
    'Ribf': ('F:RIB', {}), 'Xylf': ('F:XYL', {}), 'Lyxf': ('F:LYX', {}), 'Fruf': ('F:FRU', {}), 'Apif': ('F:API', {}),
    'Kdof': ('F:KDO', {}), 'Sedf': ('F:SED', {}), 'Xluf': ('F:XLU', {}), 'Rulf': ('F:RUL', {}), 'Altf': ('F:ALT', {}),
    'Idof': ('F:IDO', {}), 'Gulf': ('F:GUL', {}), 'Talf': ('F:TAL', {}), 'Quif': ('F:QUI', {}), 'Rhaf': ('F:RHA', {}),
    'Eryf': ('F:ERY', {}), 'Thref': ('F:THRE', {}), 'Acef': ('F:ACE', {}), 'Abef': ('F:ABE', {}),
    'All': ('P:ALL', {'mirrored': True}),  # glycowork's COMMON_ENANTIOMER treats a bare All as L-All, GlyLES as D-All
    'Allf': ('F:ALL', {'mirrored': True}),
    '6dTal': ('P:TAL', {'deoxy': (6,)}), '6dAlt': ('P:ALT', {'deoxy': (6,)}), '6dGul': ('P:GUL', {'deoxy': (6,)}),
    '6dAltf': ('F:ALT', {'deoxy': (6,)}), '4eLeg': ('P:LEG', {'invert': (4,)}),
}
HEPTOSES = {  # heptose: (hexopyranose ring it extends, configuration of the exocyclic C6)
    'LDManHep': ('Man', 'L'), 'DDManHep': ('Man', 'D'), 'DDGlcHep': ('Glc', 'D'), 'LDGlcHep': ('Glc', 'L'),
    'DDAltHep': ('Alt', 'D'), 'LDAltHep': ('Alt', 'L'), 'ManHep': ('Man', None), 'GalHep': ('Gal', None),
    'IdoHep': ('Ido', None), 'GlcHep': ('Glc', None), 'DLGlcHep': ('Glc', 'D'), 'LLGlcHep': ('Glc', 'L'),
}
MIRRORED_RING = {'DLGlcHep', 'LLGlcHep'}  # second letter L: the ring sugar itself is the L enantiomer
ALDITOLS = ['Glc', 'Gal', 'Man', 'All', 'Alt', 'Gul', 'Ido', 'Tal', 'Xyl', 'Ara', 'Rib', 'Lyx', 'Fuc', 'Rha', 'Qui',
            '6dTal', 'Fru', 'Api', 'Glcf', 'Galf', 'Eryf', 'Thref']
INOSITOL = 'O[C@@H]{r}[C@H]({p2})[C@H]({p3})[C@@H]({p4})[C@H]({p5})[C@H]{r}{p6}'  # 1D-myo, numbered off 1D-myo-inositol 1,4,5-trisphosphate
WILDCARD = {  # a Hex/dHex/Pen in a sequence states a ring size and nothing about stereochemistry, so the template states nothing either
    'Hex': ('OC{r}OC(C{p6})C({p4})C({p3})C{r}{p2}', {2: 'O', 3: 'O', 4: 'O', 6: 'O'}),
    'dHex': ('OC{r}OC(C)C({p4})C({p3})C{r}{p2}', {2: 'O', 3: 'O', 4: 'O'}),
    'Pen': ('OC{r}OCC({p4})C({p3})C{r}{p2}', {2: 'O', 3: 'O', 4: 'O'}),
}


def number_atoms(mol):
    """Map every atom of the main ring and its tails onto a carbon number, starting at the anomeric carbon."""
    rings = [r for r in mol.GetRingInfo().AtomRings() if sum(mol.GetAtomWithIdx(a).GetAtomicNum() == 8 for a in r) == 1]
    if not rings:
        raise ValueError('no ring with a single oxygen')
    ring = max(rings, key = len)
    ring_oxygen = next(a for a in ring if mol.GetAtomWithIdx(a).GetAtomicNum() == 8)
    candidates = [c for c in (n.GetIdx() for n in mol.GetAtomWithIdx(ring_oxygen).GetNeighbors())
                  if [n for n in mol.GetAtomWithIdx(c).GetNeighbors() if n.GetAtomicNum() == 8 and n.GetIdx() not in ring]]
    if len(candidates) != 1:
        raise ValueError(f'anomeric carbon is ambiguous: {candidates}')
    anomeric = candidates[0]
    head = [n.GetIdx() for n in mol.GetAtomWithIdx(anomeric).GetNeighbors() if n.GetAtomicNum() == 6 and n.GetIdx() not in ring]
    if len(head) > 1:
        raise ValueError('branched anomeric head')
    position, index = ({head[0]: 1}, 2) if head else ({}, 1)  # a ketose or ulosonic acid carries C1 outside the ring
    position[anomeric], previous, current = index, ring_oxygen, anomeric
    while True:
        following = [n.GetIdx() for n in mol.GetAtomWithIdx(current).GetNeighbors()
                     if n.GetIdx() in ring and n.GetIdx() not in (previous, ring_oxygen)]
        if not following:
            break
        index += 1
        previous, current = current, following[0]
        position[current] = index
    while True:
        following = [n.GetIdx() for n in mol.GetAtomWithIdx(current).GetNeighbors()
                     if n.GetAtomicNum() == 6 and n.GetIdx() not in position and n.GetIdx() not in ring]
        if not following:
            break
        if len(following) > 1:
            raise ValueError('branched tail')
        index += 1
        previous, current = current, following[0]
        position[current] = index
    return position, ring, ring_oxygen, anomeric


def dummies(mol, position, ring, anomeric_oxygen):
    """Replace every substitutable hydroxyl or amine with an isotope-labelled dummy atom."""
    editable, hetero = Chem.RWMol(mol), {}
    for carbon, number in sorted(position.items(), key = lambda item: item[1]):
        atom = mol.GetAtomWithIdx(carbon)
        if any(bond.GetBondTypeAsDouble() == 2 for bond in atom.GetBonds()):
            continue
        exocyclic = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() in (7, 8) and n.GetIdx() not in ring
                     and n.GetIdx() != anomeric_oxygen and not any(b.GetBondTypeAsDouble() == 2 for b in n.GetBonds())]
        if len(exocyclic) == 1:
            hetero[number] = 'N' if mol.GetAtomWithIdx(exocyclic[0]).GetAtomicNum() == 7 else 'O'
            editable.GetAtomWithIdx(exocyclic[0]).SetAtomicNum(0)
            editable.GetAtomWithIdx(exocyclic[0]).SetIsotope(number)
    return editable, hetero


def slotted(smiles):
    """Turn one monosaccharide SMILES into a template rooted at its anomeric oxygen."""
    mol = Chem.MolFromSmiles(smiles)
    position, ring, ring_oxygen, anomeric = number_atoms(mol)
    anomeric_oxygen = next(n.GetIdx() for n in mol.GetAtomWithIdx(anomeric).GetNeighbors()
                           if n.GetAtomicNum() == 8 and n.GetIdx() not in ring)
    editable, hetero = dummies(mol, position, ring, anomeric_oxygen)
    out = Chem.MolToSmiles(editable.GetMol(), rootedAtAtom = anomeric_oxygen, canonical = True)
    out = re.sub(r'\[[^\]]*\]|\d', lambda m: m.group() if m.group()[0] == '[' else '{r}', out)
    for number in hetero:
        out = out.replace('[%d*]' % number, '{p%d}' % number)
    return out, hetero


def open_chain(smiles):
    """Template for the reduced sugar, numbered exactly like the ring form it comes from."""
    mol = Chem.MolFromSmiles(smiles)
    position, ring, ring_oxygen, anomeric = number_atoms(mol)
    anomeric_oxygen = next(n.GetIdx() for n in mol.GetAtomWithIdx(anomeric).GetNeighbors()
                           if n.GetAtomicNum() == 8 and n.GetIdx() not in ring)
    editable, hetero = dummies(mol, position, ring, anomeric_oxygen)
    editable.RemoveBond(anomeric, ring_oxygen)
    editable.GetAtomWithIdx(anomeric).SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    editable.GetAtomWithIdx(anomeric).SetNumExplicitHs(0)
    editable.GetAtomWithIdx(anomeric).SetNoImplicit(False)
    last = max(number for carbon, number in position.items() if carbon in ring)
    editable.GetAtomWithIdx(ring_oxygen).SetAtomicNum(0)  # the ring oxygen becomes the hydroxyl of the last ring carbon
    editable.GetAtomWithIdx(ring_oxygen).SetIsotope(last)
    hetero[last] = 'O'
    out = Chem.MolToSmiles(editable.GetMol(), rootedAtAtom = anomeric_oxygen, canonical = True)
    for number in hetero:
        out = out.replace('[%d*]' % number, '{p%d}' % number)
    return out, dict(sorted(hetero.items()))


def mirror(smiles):
    "The enantiomer: every stereocentre inverted at once"
    return Chem.CanonSmiles(smiles.replace('@@', '\x00').replace('@', '@@').replace('\x00', '@'))


def edit(smiles, invert = (), deoxy = (), amino = ()):
    """Invert, deoxygenate or aminate the given carbon positions of a monosaccharide."""
    mol = Chem.MolFromSmiles(smiles)
    position, ring, ring_oxygen, anomeric = number_atoms(mol)
    index = {number: carbon for carbon, number in position.items()}
    editable = Chem.RWMol(mol)
    oxygen = lambda p: next(n.GetIdx() for n in mol.GetAtomWithIdx(index[p]).GetNeighbors()
                            if n.GetAtomicNum() == 8 and n.GetIdx() not in ring)
    for p in invert:
        editable.GetAtomWithIdx(index[p]).InvertChirality()
    for p in amino:
        editable.GetAtomWithIdx(oxygen(p)).SetAtomicNum(7)
    for atom in sorted((oxygen(p) for p in deoxy), reverse = True):
        editable.RemoveAtom(atom)
    return Chem.MolToSmiles(editable.GetMol())


def anomers(source, invert = (), deoxy = (), mirrored = False):
    """The alpha and beta base structures for one entry of the SPEC table."""
    table = PYRANOSE if source[0] == 'P' else FURANOSE
    alpha, beta = (table['A_' + source[2:]]['smiles'], table['B_' + source[2:]]['smiles'])
    if mirrored:
        alpha, beta = mirror(alpha), mirror(beta)
    if invert or deoxy:
        alpha, beta = edit(alpha, invert = invert, deoxy = deoxy), edit(beta, invert = invert, deoxy = deoxy)
    return Chem.CanonSmiles(alpha), Chem.CanonSmiles(beta)


def compact(name, alpha_smiles, beta_smiles):
    """Fold the alpha and beta templates into one string with an '{a}' placeholder."""
    alpha, hetero = slotted(alpha_smiles)
    beta, beta_hetero = slotted(beta_smiles)
    assert hetero == beta_hetero, (name, hetero, beta_hetero)
    assert alpha.replace('@@', '@') == beta.replace('@@', '@'), (name, alpha, beta)
    match = re.match(r'^O\[C@(@?)[H\]]', alpha)
    assert match, (name, alpha)
    tag = match.group(1)
    template = alpha[:4] + '{a}' + alpha[4 + len(tag):]
    assert template.replace('{a}', tag) == alpha, (name, template, alpha)
    assert template.replace('{a}', '@' if tag == '' else '') == beta, (name, template, beta)
    return template, hetero, tag


def heptose(template, hetero, tag, glycero):
    """Grow C7 onto a hexopyranose; D-glycero means C6 comes out R, L-glycero S."""
    assert 'C{p6}' in template, template
    if glycero is None:
        return template.replace('C{p6}', 'C({p6})C{p7}'), {**hetero, 7: 'O'}
    hits = []
    for candidate in ('', '@'):
        grown = template.replace('C{p6}', '[C@%sH]({p6})C{p7}' % candidate)
        extended = {**hetero, 7: 'O'}
        smiles = grown.replace('{a}', tag).replace('{r}', '1')
        for number, atom in extended.items():
            smiles = smiles.replace('{p%d}' % number, atom)
        mol = Chem.MolFromSmiles(smiles)
        Chem.AssignStereochemistry(mol, cleanIt = True, force = True)
        position, ring, ring_oxygen, anomeric = number_atoms(mol)
        carbon = next(c for c, number in position.items() if number == 6)
        if mol.GetAtomWithIdx(carbon).GetPropsAsDict().get('_CIPCode') == {'D': 'R', 'L': 'S'}[glycero]:
            hits.append((grown, extended))
    assert len(hits) == 1, (glycero, len(hits))
    return hits[0]


def build():
    """The whole table: ring skeletons, heptoses, inositol and the open-chain alditols."""
    skeletons, alditols, enantiomer = {}, {}, {}
    for name, (source, kwargs) in SPEC.items():
        alpha, beta = anomers(source, **kwargs)
        skeletons[name] = compact(name, alpha, beta)
        table = PYRANOSE if source[0] == 'P' else FURANOSE
        default = str(table['A_' + source[2:]]['isomer']).split('.')[-1][0]
        if default in ('D', 'L'):
            enantiomer[name] = {'D': 'L', 'L': 'D'}[default] if kwargs.get('mirrored') else default
    for name, (parent, glycero) in HEPTOSES.items():
        template, hetero, tag = skeletons[parent]
        if name in MIRRORED_RING:
            source, kwargs = SPEC[parent]
            template, hetero, tag = compact(name, *anomers(source, mirrored = True, **kwargs))
        grown, extended = heptose(template, hetero, tag, glycero)
        skeletons[name] = (grown, extended, tag)
        enantiomer[name] = {'D': 'L', 'L': 'D'}[enantiomer[parent]] if name in MIRRORED_RING else enantiomer[parent]
    skeletons['Ins'] = (INOSITOL, {p: 'O' for p in range(2, 7)}, '')
    for name, (template, hetero) in WILDCARD.items():
        skeletons[name] = (template, hetero, '')
    for name in ALDITOLS:
        source, kwargs = SPEC[name]
        alditols[name.rstrip('f') if name in ('Eryf', 'Thref') else name] = open_chain(anomers(source, **kwargs)[1])
    return skeletons, alditols, enantiomer


if __name__ == '__main__':
    skeletons, alditols, enantiomer = build()
    print('SKELETONS = {  # monosaccharide: (template, alpha tag, {position: heteroatom})')
    for name in sorted(skeletons):
        template, hetero, tag = skeletons[name]
        print(f'    {name!r}: ({template!r}, {tag!r}, {dict(sorted(hetero.items()))!r}),')
    print('}\n')
    print("ALDITOLS = {  # open-chain form of a reduced sugar ('-ol'), numbered like the parent: (template, {position: heteroatom})")
    for name in sorted(alditols):
        print(f'    {name!r}: {alditols[name]!r},')
    print('}\n')
    print('ENANTIOMER = {' + ', '.join(f'{k!r}: {v!r}' for k, v in sorted(enantiomer.items())) + '}')
    print(f'\n# {len(skeletons)} skeletons, {len(alditols)} alditols', file = sys.stderr)
