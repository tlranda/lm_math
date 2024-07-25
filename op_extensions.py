import importlib
import pathlib
import sys
import warnings

op_extensions_path = pathlib.Path(__file__).parents[0] / 'op_extensions'
if str(op_extensions_path) not in sys.path:
    sys.path.append(str(op_extensions_path))

"""
    HOW TO ADD A CUSTOM OPERATOR TO LM_MATH:

    Create a Python script under the op_extensions_path directory above
    (usually ./op_extensions relative to this file's location)

    In that file, implement a function with the SAME NAME as the file, ie:
    `op_extensions/myNewFunc.py` should implement a function named myNewFunc().

    Functions are expected to operate on TWO integer operands and return a
    value that can be `==` equality compared with the LLM's output.
    Currently, only INTEGER return values are fully suppored for post-processing,
    but in the future we expect to support more diverse types for the LLM to
    produce via custom regexes. Note that your custom types are NOT imported,
    so python-primitives are the best choices of return values. Numpy
    primitives are also supported, but the code will NOT call np.arrayequal()
    for you.

    Next, declare a global-scope variable in your file named 'sign', and set
    its value to a STRING that represents a text-based symbol for invoking your
    new operation, ie:
    `op_extensions/myNewFunc.py` may represent the myNewFunc() operation using
    the `@` character by adding the following in their code:
        sign = '@'

    Finally, declare a description string for your attribute that explains to
    the LLM approximately how this operator is expected to function. Your
    description can be as leading as you like, it will be provided with every
    LLM prompt unless using the no-prompt setting.
    Your description will be prefixed by the following string:
        The \'?\' symbol 
    Where ? is automatically replaced by the symbol used in LLM prompts (which
    may override your default symbol)
"""

names = list()
signs = list()
calls = list()
descriptions = list()

# These names would clash with default operators or mess up imports
banned_names = ['add','mod','mul','pow','sub'] # Operators from operator library
# Automatically detect operator extensions in this directory
for file in op_extensions_path.iterdir():
    if not file.is_file() or file.suffix != '.py' or file.stem.startswith('__'):
        continue
    if file.stem in banned_names:
        warnings.warn(f"Operator Extension '{file}' not imported due to name clash")
        continue
    # Attempt import
    try:
        auto_import = importlib.import_module(file.stem)
        func = getattr(auto_import, file.stem)
        if not callable(func):
            raise ValueError("Operator is not callable")
        sign = getattr(auto_import, 'sign')
        description = getattr(auto_import, 'description')
        names.append(file.stem)
        signs.append(sign)
        calls.append(func)
        descriptions.append(description)
    except Exception as e:
        warnings.warn(f"Operator extension in file '{file}' failed to load: {e}", UserWarning)

