from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("glycowork")
except PackageNotFoundError:  # a source checkout on sys.path, or a frozen app, has no installed metadata
    __version__ = "unknown"
from .motif.draw import GlycoDraw

__all__ = ['GlycoDraw']

def __getattr__(name):
  if name == "glycan_data":
    import glycowork.glycan_data as glycan_data
    globals()[name] = glycan_data
    return glycan_data
  elif name == "motif":
    import glycowork.motif as motif
    globals()[name] = motif
    return motif
  elif name == "network":
    import glycowork.network as network
    globals()[name] = network
    return network
  elif name == "ml":
    import glycowork.ml as ml
    globals()[name] = ml
    return ml
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
