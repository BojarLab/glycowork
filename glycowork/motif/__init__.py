def __getattr__(name):
  if name == "graph":
    import glycowork.motif.graph as graph
    globals()[name] = graph
    return graph
  elif name == "annotate":
    import glycowork.motif.annotate as annotate
    globals()[name] = annotate
    return annotate
  elif name == "draw":
    import glycowork.motif.draw as draw
    globals()[name] = draw
    return draw
  elif name == "analysis":
    import glycowork.motif.analysis as analysis
    globals()[name] = analysis
    return analysis
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
