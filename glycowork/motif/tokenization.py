

def character_to_label(character, libr):
  """tokenizes character by indexing passed library
  character -- character to index
  libr -- list of library items

  returns index of character in library
  """
  character_label = libr.index(character)
  return character_label

def string_to_labels(character_string, libr):
  """tokenizes word by indexing characters in passed library
  character_string -- string of characters to index
  libr -- list of library items

  returns indexes of characters in library
  """
  return list(map(lambda character: character_to_label(character, libr), character_string))

def pad_sequence(seq, max_length, pad_label = len(['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T',
     'V','W','Y','X'])):
  """brings all sequences to same length by adding padding token
  seq -- sequence to pad
  max_length -- sequence length to pad to
  pad_label -- which padding label to use

  returns padded sequence
  """
  seq += [pad_label for i in range(max_length-len(seq))]
  return seq
