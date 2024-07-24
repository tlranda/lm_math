import importlib
import itertools
import pathlib

# To implement another tokenizer, make a directory with its importable name at
# this level, then in that directory make the same name as a .py file
# This file should implement all attributes in the ATTRIBUTES list
IMPLEMENTERS = [_.stem for _ in pathlib.Path(__file__).parents[0].iterdir() \
                if _.is_dir() and (_ / f"{_.stem}.py").exists()]
ATTRIBUTES = ['tk', # Tokenizer object that implements at minimum the function
                    # tokenize(str) -> List[str], such that each returned string
                    # is a single token
              'number_vocab', # numpy array of numeric strings that are
                              # represented as single tokens
              'number_vocab_int', # Same as above, but unique numbers as integers
              ]
# Importable names for this module are mapped to their local import path and
# fetchable attribute
modules = dict(("_".join([model,attr]),
                (f"lm_tokenizers.{model}.{model}",attr)) \
                        for (model,attr) in itertools.product(
                            IMPLEMENTERS,
                            ATTRIBUTES))
# Dictionary of saved lazy imports
loaded_modules = dict()

def getattr_fn(name):
    if name not in modules:
        raise AttributeError

    module, attr = modules[name]
    # Lazy load of the relative submodule
    if module not in loaded_modules:
        loaded_modules[module] = importlib.import_module(module)
    return getattr(loaded_modules[module], attr)

__getattr__ = getattr_fn

