from glycowork.glycan_data.loader import lib, df_glysum, df_glycan
from glycowork.motif.processing import small_motif_find
try:
    import numpypy as np
except ImportError:
    import numpy as np
import operator
from abc import ABCMeta
from abc import abstractmethod

GAP_ELEMENT = '-'
GAP_CODE = 0

def pairwiseAlign(query, corpus = None, n = 5, vocab = None,
                  submat = None, mismatch = -10, gap = -5,
                  col = 'glycan'):
  """aligns glycan sequence from database against rest of the database and returns the best n alignments\n
  query -- glycan string in IUPACcondensed notation\n
  corpus -- database to align query against; default is SugarBase\n
  n -- how many alignments to show; default shows top 5\n
  vocab -- list of glycowords used for mapping to tokens\n
  submat -- GLYSUM substitution matrix\n
  mismatch -- mismatch penalty; default: -10\n
  gap -- gap penalty; default: -5\n
  col -- column name where glycan sequences are; default: glycan\n

  returns the n best alignments of query against corpus in text form with scores etc
  """
  if corpus is None:
      corpus = df_glycan
  if vocab is None:
      vocab = lib
  if submat is None:
      submat = df_glysum
  if n == 0:
    n = len(corpus)
  seqs = corpus[col].values.tolist()
  a_in = small_motif_find(query).split('*')
  a = Sequence(a_in)
  v = Vocabulary()
  voc = v.encodeSequence(Sequence(vocab))
  a_enc = v.encodeSequence(a)
  scoring = SubstitutionScoring(submat, mismatch)
  aligner = GlobalSequenceAligner(scoring, gap)
  specs = corpus.Species.values.tolist()
  corpus = [small_motif_find(k) for k in corpus[col].values.tolist()]
  track = []

  for k in range(len(corpus)):
    b = Sequence(corpus[k].split('*'))
    b_enc = v.encodeSequence(b)
    try:
        score, encodeds = aligner.align(a_enc, b_enc, backtrace = True)
    except:
        score = 0
        encodeds = None
    track.append((score, encodeds, corpus[k], specs[k], len(b)))

  track.sort(key = operator.itemgetter(0), reverse = True)
  for k in track[:n]:
      score, encodeds, idx, species, length = k
      if score > 0:
          for encoded in encodeds:
              alignment = v.decodeSequenceAlignment(encoded)
              print(str(a_in.index(alignment[0][0]) + 1),
                    len(alignment)*' '*5,
                    str((len(a) + 1) - (a_in[::-1].index(alignment[-1][0]) + 1)))
              print(alignment)
              print('Alignment Score:', alignment.score)
              print('Percent Identity:', alignment.percentIdentity())
              print('Percent Coverage:', min([(len(alignment)/len(a))*100, 100.0]))
              print('Sequence Index:', idx)
              print('Species:', species)
              print()

class BaseSequence(object):

    def __init__(self, elements, id = None):
        self.elements = elements
        self.id = id

    def key(self):
        return tuple(self.elements)

    def reversed(self):
        return type(self)(self.elements[::-1], id = self.id)

    def __eq__(self, other):
        if self.id is None or other.id is None:
            return self.elements == other.elements
        else:
            return self.id == other.id

    def __hash__(self):
        if self.id is None:
            return hash(self.key())
        else:
            return hash(self.id)

    def __len__(self):
        return len(self.elements)

    def __getitem__(self, item):
        return self.elements[item]

    def __setitem__(self, key, value):
        self.elements[key] = value

    def __iter__(self):
        return iter(self.elements)

    def __repr__(self):
        return repr(self.elements)

    def __str__(self):
        if self.id is None:
            result = ''
        else:
            result = '> %s\n' % self.id
        result += ' '.join(str(e) for e in self.elements)
        return result

    def __unicode__(self):
        if self.id is None:
            result = u''
        else:
            result = u'> %s\n' % self.id
        result += u' '.join(text_type(e) for e in self.elements)
        return result

# Sequence Types ---------------------------------

class Sequence(BaseSequence):

    def __init__(self, elements = None, id = None):
        if elements is None:
            super(Sequence, self).__init__(list(), id)
        else:
            super(Sequence, self).__init__(list(elements), id)

    def push(self, element):
        self.elements.append(element)

    def pop(self):
        return self.elements.pop()

class EncodedSequence(BaseSequence):

    def __init__(self, argument, id=None):
        if isinstance(argument, int):
            super(EncodedSequence, self).__init__(
                np.zeros(argument, int), id)
            self.position = 0
        else:
            if isinstance(argument, np.ndarray) \
                    and argument.dtype.name.startswith('int'):
                super(EncodedSequence, self).__init__(
                    np.array(argument), id)
            else:
                super(EncodedSequence, self).__init__(
                    np.array(list(argument), int), id)
            self.position = len(self.elements)

    def push(self, element):
        self.elements[self.position] = element
        self.position += 1

    def pop(self):
        self.position -= 1
        return int(self.elements[self.position])

    def key(self):
        return tuple(int(e) for e in self.elements[:self.position])

    def reversed(self):
        return EncodedSequence(
            self.elements[self.position - len(self.elements) - 1::-1],
            id=self.id,
        )

    def __len__(self):
        return self.position

    def __iter__(self):
        return (int(e) for e in self.elements)

# Vocabulary ----------------------------------

class Vocabulary(object):
    def __init__(self):
        self.__elementToCode = {GAP_ELEMENT: GAP_CODE}
        self.__codeToElement = {GAP_CODE: GAP_ELEMENT}

    def has(self, element):
        return element in self.__elementToCode

    def hasCode(self, code):
        return code in self.__codeToElement

    def encode(self, element):
        code = self.__elementToCode.get(element)
        if code is None:
            code = len(self.__elementToCode)
            self.__elementToCode[element] = code
            self.__codeToElement[code] = element
        return code

    def decode(self, code):
        try:
            return self.__codeToElement[code]
        except KeyError:
            raise KeyError(
                'there is no elements in the vocabulary encoded as %r'
                % code)

    def encodeSequence(self, sequence):
        encoded = EncodedSequence(len(sequence), id=sequence.id)
        for element in sequence:
            encoded.push(self.encode(element))
        return encoded

    def decodeSequence(self, sequence):
        decoded = Sequence(id=sequence.id)
        for code in sequence:
            decoded.push(self.decode(code))
        return decoded

    def decodeSequenceAlignment(self, alignment):
        first = self.decodeSequence(alignment.first)
        second = self.decodeSequence(alignment.second)
        return SequenceAlignment(first, second, self.decode(alignment.gap),
                                 alignment)

    def decodeSoft(self, softCode):
        weights = dict()
        for code, weight in softCode.pairs():
            weights[self.__codeToElement[code]] = weight
        return SoftElement(weights)

    def decodeProfile(self, profile):
        decoded = Profile()
        for softCode in profile:
            decoded.push(self.decodeSoft(softCode))
        return decoded

    def decodeProfileAlignment(self, alignment):
        first = self.decodeProfile(alignment.first)
        second = self.decodeProfile(alignment.second)
        return ProfileAlignment(first, second,
                                self.decodeSoft(alignment.gap),
                                alignment)

    def elements(self):
        return [self.decode(c) for c in sorted(self.__codeToElement)]

    def __len__(self):
        return len(self.__elementToCode)

    def __iter__(self):
        return iter(self.__elementToCode)

    def __repr__(self):
        return repr(self.elements())

# Scoring -------------------------------

class Scoring(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __call__(self, firstElement, secondElement):
        return 0

class SimpleScoring(Scoring):

    def __init__(self, matchScore, mismatchScore):
        self.matchScore = matchScore
        self.mismatchScore = mismatchScore

    def __call__(self, firstElement, secondElement):
        if firstElement == secondElement:
            return self.matchScore
        else:
            return self.mismatchScore
        
class SubstitutionScoring(Scoring):
    
    def __init__(self, subMatrix, mismatchScore):
        self.matchScore = subMatrix
        self.mismatchScore = mismatchScore

    def __call__(self, firstElement, secondElement):
        if firstElement == 0:
            return self.mismatchScore
        elif secondElement == 0:
            return self.mismatchScore
        else:
            try:
              temp = self.matchScore.iloc[firstElement-1, secondElement-1]
            except:
              temp = self.mismatchScore
            return temp

# Alignment ---------------------------------

class SequenceAlignment(object):

    def __init__(self, first, second, gap=GAP_CODE, other=None):
        self.first = first
        self.second = second
        self.gap = gap
        if other is None:
            self.scores = [0] * len(first)
            self.score = 0
            self.identicalCount = 0
            self.similarCount = 0
            self.gapCount = 0
        else:
            self.scores = list(other.scores)
            self.score = other.score
            self.identicalCount = other.identicalCount
            self.similarCount = other.similarCount
            self.gapCount = other.gapCount

    def push(self, firstElement, secondElement, score=0):
        self.first.push(firstElement)
        self.second.push(secondElement)
        self.scores.append(score)
        self.score += score
        if firstElement == secondElement:
            self.identicalCount += 1
        if score > 0:
            self.similarCount += 1
        if firstElement == self.gap or secondElement == self.gap:
            self.gapCount += 1
        pass

    def pop(self):
        firstElement = self.first.pop()
        secondElement = self.second.pop()
        score = self.scores.pop()
        self.score -= score
        if firstElement == secondElement:
            self.identicalCount -= 1
        if score > 0:
            self.similarCount -= 1
        if firstElement == self.gap or secondElement == self.gap:
            self.gapCount -= 1
        return firstElement, secondElement

    def key(self):
        return self.first.key(), self.second.key()

    def reversed(self):
        first = self.first.reversed()
        second = self.second.reversed()
        return type(self)(first, second, self.gap, self)

    def percentIdentity(self):
        try:
            return float(self.identicalCount) / len(self) * 100.0
        except ZeroDivisionError:
            return 0.0

    def percentSimilarity(self):
        try:
            return float(self.similarCount) / len(self) * 100.0
        except ZeroDivisionError:
            return 0.0

    def percentGap(self):
        try:
            return float(self.gapCount) / len(self) * 100.0
        except ZeroDivisionError:
            return 0.0

    def quality(self):
        return self.score, \
            self.percentIdentity(), \
            self.percentSimilarity(), \
            -self.percentGap()

    def __len__(self):
        assert len(self.first) == len(self.second)
        return len(self.first)

    def __getitem__(self, item):
        return self.first[item], self.second[item]

    def __repr__(self):
        return repr((self.first, self.second))

    def __str__(self):
        first = [str(e) for e in self.first.elements]
        second = [str(e) for e in self.second.elements]
        for i in range(len(first)):
            n = max(len(first[i]), len(second[i]))
            format = '%-' + str(n) + 's'
            first[i] = format % first[i]
            second[i] = format % second[i]
        return '%s\n%s' % (' '.join(first), ' '.join(second))

    def __unicode__(self):
        first = [text_type(e) for e in self.first.elements]
        second = [text_type(e) for e in self.second.elements]
        for i in range(len(first)):
            n = max(len(first[i]), len(second[i]))
            format = u'%-' + text_type(n) + u's'
            first[i] = format % first[i]
            second[i] = format % second[i]
        return u'%s\n%s' % (u' '.join(first), u' '.join(second))

# Aligner ---------------------------------------------------------------------

class SequenceAligner(object):
    __metaclass__ = ABCMeta

    def __init__(self, scoring, gapScore):
        self.scoring = scoring
        self.gapScore = gapScore

    def align(self, first, second, backtrace=False):
        f = self.computeAlignmentMatrix(first, second)
        score = self.bestScore(f)
        if backtrace:
            alignments = self.backtrace(first, second, f)
            return score, alignments
        else:
            return score

    def emptyAlignment(self, first, second):
        # Pre-allocate sequences.
        return SequenceAlignment(
            EncodedSequence(len(first) + len(second), id=first.id),
            EncodedSequence(len(first) + len(second), id=second.id),
        )

    @abstractmethod
    def computeAlignmentMatrix(self, first, second):
        return np.zeros(0, int)

    @abstractmethod
    def bestScore(self, f):
        return 0

    @abstractmethod
    def backtrace(self, first, second, f):
        return list()


class GlobalSequenceAligner(SequenceAligner):

    def __init__(self, scoring, gapScore):
        super(GlobalSequenceAligner, self).__init__(scoring, gapScore)

    def computeAlignmentMatrix(self, first, second):
        m = len(first) + 1
        n = len(second) + 1
        f = np.zeros((m, n), int)
        for i in range(1, m):
            for j in range(1, n):
                # Match elements.
                ab = f[i - 1, j - 1] \
                    + self.scoring(first[i - 1], second[j - 1])

                # Gap on first sequence.
                if i == m - 1:
                    ga = f[i, j - 1]
                else:
                    ga = f[i, j - 1] + self.gapScore

                # Gap on second sequence.
                if j == n - 1:
                    gb = f[i - 1, j]
                else:
                    gb = f[i - 1, j] + self.gapScore

                f[i, j] = max(ab, max(ga, gb))
        return f

    def bestScore(self, f):
        return f[-1, -1]

    def backtrace(self, first, second, f):
        m, n = f.shape
        alignments = list()
        alignment = self.emptyAlignment(first, second)
        self.backtraceFrom(first, second, f, m - 1, n - 1,
                           alignments, alignment)
        return alignments

    def backtraceFrom(self, first, second, f, i, j, alignments, alignment):
        if i == 0 or j == 0:
            alignments.append(alignment.reversed())
        else:
            m, n = f.shape
            c = f[i, j]
            p = f[i - 1, j - 1]
            x = f[i - 1, j]
            y = f[i, j - 1]
            a = first[i - 1]
            b = second[j - 1]
            if c == p + self.scoring(a, b):
                alignment.push(a, b, c - p)
                self.backtraceFrom(first, second, f, i - 1, j - 1,
                                   alignments, alignment)
                alignment.pop()
            else:
                if i == m - 1:
                    if c == y:
                        self.backtraceFrom(first, second, f, i, j - 1,
                                           alignments, alignment)
                elif c == y + self.gapScore:
                    alignment.push(alignment.gap, b, c - y)
                    self.backtraceFrom(first, second, f, i, j - 1,
                                       alignments, alignment)
                    alignment.pop()
                if j == n - 1:
                    if c == x:
                        self.backtraceFrom(first, second, f, i - 1, j,
                                           alignments, alignment)
                elif c == x + self.gapScore:
                    alignment.push(a, alignment.gap, c - x)
                    self.backtraceFrom(first, second, f, i - 1, j,
                                       alignments, alignment)
                    alignment.pop()

class LocalSequenceAligner(SequenceAligner):

    def __init__(self, scoring, gapScore, minScore=None):
        super(LocalSequenceAligner, self).__init__(scoring, gapScore)
        self.minScore = minScore

    def computeAlignmentMatrix(self, first, second):
        m = len(first) + 1
        n = len(second) + 1
        f = numpy.zeros((m, n), int)
        for i in range(1, m):
            for j in range(1, n):
                # Match elements.
                ab = f[i - 1, j - 1] \
                    + self.scoring(first[i - 1], second[j - 1])

                # Gap on sequenceA.
                ga = f[i, j - 1] + self.gapScore

                # Gap on sequenceB.
                gb = f[i - 1, j] + self.gapScore

                f[i, j] = max(0, max(ab, max(ga, gb)))
        return f

    def bestScore(self, f):
        return f.max()

    def backtrace(self, first, second, f):
        m, n = f.shape
        alignments = list()
        alignment = self.emptyAlignment(first, second)
        if self.minScore is None:
            minScore = self.bestScore(f)
        else:
            minScore = self.minScore
        for i in range(m):
            for j in range(n):
                if f[i, j] >= minScore:
                    self.backtraceFrom(first, second, f, i, j,
                                       alignments, alignment)
        return alignments

    def backtraceFrom(self, first, second, f, i, j, alignments, alignment):
        if f[i, j] == 0:
            alignments.append(alignment.reversed())
        else:
            c = f[i, j]
            p = f[i - 1, j - 1]
            x = f[i - 1, j]
            y = f[i, j - 1]
            a = first[i - 1]
            b = second[j - 1]
            if c == p + self.scoring(a, b):
                alignment.push(a, b, c - p)
                self.backtraceFrom(first, second, f, i - 1, j - 1,
                                   alignments, alignment)
                alignment.pop()
            else:
                if c == y + self.gapScore:
                    alignment.push(alignment.gap, b, c - y)
                    self.backtraceFrom(first, second, f, i, j - 1,
                                       alignments, alignment)
                    alignment.pop()
                if c == x + self.gapScore:
                    alignment.push(a, alignment.gap, c - x)
                    self.backtraceFrom(first, second, f, i - 1, j,
                                       alignments, alignment)
                    alignment.pop()
