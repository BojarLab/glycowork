from glycowork.helper.func import *
from glycowork.motif.processing import *

df_glysum = load_file("df_glyco_substitution_iso2.csv")
df_glycan = load_file("v3_sugarbase.csv")

lib = get_lib(df_glycan.glycan.values.tolist())

def pairwiseAlign(query, corpus, n = 5, database = df_glycan, vocab = lib,
                  submat = df_glysum, mismatch = -10, gap = -5, self_contain = True, query_in_corpus = True):
  """aligns glycan sequence from database against rest of the database and returns the best n alignments"""
  if n == 0:
    n = len(corpus)
  seqs = database.glycan.values.tolist()
  if query_in_corpus:
    a_in = seqs[query].split('*')
  else:
    a_in = small_motif_find(query).split('*')
  a = Sequence(a_in)
  v = Vocabulary()
  voc = v.encodeSequence(Sequence(vocab))
  a_enc = v.encodeSequence(a)
  scoring = SubstitutionScoring(submat, mismatch)
  aligner = GlobalSequenceAligner(scoring, gap)
  specs = database.species.values.tolist()
  track = []

  for k in range(len(corpus)):
    b = Sequence(seqs[corpus[k]].split('*'))
    b_enc = v.encodeSequence(b)
    score, encodeds = aligner.align(a_enc, b_enc, backtrace = True)
    track.append((score, encodeds, corpus[k], specs[corpus[k]], len(b)))

  track.sort(key = operator.itemgetter(0), reverse = True)
  if self_contain:
    for k in track[1:n+1]:
      score,encodeds,idx,species,length = k
      for encoded in encodeds:
        alignment = v.decodeSequenceAlignment(encoded)
        print(str(a_in.index(alignment[0][0])+1),
            len(alignment)*' '*5,
            str((len(a)+1)-(a_in[::-1].index(alignment[-1][0])+1)))
        print(alignment)
        print('Alignment Score:', alignment.score)
        print('Percent Identity:', alignment.percentIdentity())
        print('Percent Coverage:', min([(len(alignment)/len(a))*100, 100.0]))
        print('Sequence Index:', idx)
        print('Species:', species)
        print()
  else:
    for k in track[:n]:
      score,encodeds,idx,species,length = k
      for encoded in encodeds:
        alignment = v.decodeSequenceAlignment(encoded)
        print(str(a_in.index(alignment[0][0])+1),
            len(alignment)*' '*5,
            str((len(a)+1)-(a_in[::-1].index(alignment[-1][0])+1)))
        print(alignment)
        print('Alignment Score:', alignment.score)
        print('Percent Identity:', alignment.percentIdentity())
        print('Percent Coverage:', min([(len(alignment)/len(a))*100, 100.0]))
        print('Sequence Index:', idx)
        print('Species:', species)
        print()
