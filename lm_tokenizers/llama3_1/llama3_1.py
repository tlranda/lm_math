# Builtin
import functools
import pathlib
import re
import sys
from typing import AbstractSet, Collection, List, Literal, Optional, Union
# Dependent
import numpy as np
import tiktoken
from tiktoken.load import load_tiktoken_bpe
# Local

# Clone https://github.com/meta-llama/llama-models (1b58927) and set a symlink
# from: <prior_path_here>/llama_models/models/llama3_1/api/
# to: ./llama_models_api
from .llama_models_api.tokenizer import Tokenizer
# If you install the llama-models repository via its provided setuptools in the
# top-level directory of the repository, you may instead import this as:
#from llama_models.llama3_1.api.tokenizer import Tokenizer
# However, you'll need to update this TOKENIZER_MODEL_PATH below to properly
# source the token vocabulary

TOKENIZER_MODEL_PATH = pathlib.Path(__file__).parents[0] /\
                       'llama_models_api/tokenizer.model'

__all__ = ['tk','number_vocab']

def get_number_vocab(pth):
    mergeable_ranks = load_tiktoken_bpe(str(pth))
    integer_pattern = re.compile(r"-?\d+")
    all_vocab_decoded = [t.decode('latin-1') for t in mergeable_ranks.keys()]
    return np.array(sorted([t for t in all_vocab_decoded \
                            if integer_pattern.match(t)],
                    key=lambda x: int(x)))

number_vocab = get_number_vocab(TOKENIZER_MODEL_PATH)
number_vocab_int = np.array(sorted(set(number_vocab.astype(int))))
tk = Tokenizer(str(TOKENIZER_MODEL_PATH))

# Extension to support a .tokenize() call
def tokenize(self,
    s: str,
    bos: bool = False,
    eos: bool = False,
    allowed_special: Optional[Union[Literal["all"], AbstractSet[str]]] = None,
    disallowed_special: Union[Literal["all"], Collection[str]] = (),
    count: bool = False) -> List[str]:
    """
    Splits a string into its tokenized components and returns them as
    their token components.
    """
    input_tokens = self.encode(s, bos=bos, eos=eos,
                               allowed_special=allowed_special,
                               disallowed_special=disallowed_special)
    if count:
        return len(input_tokens)
    dec_strings = []
    for token in input_tokens:
        dec_strings.append(self.decode([token]))
    return dec_strings

tk.tokenize = functools.partial(tokenize, tk)

