# Builtin
import re
# Dependent
import numpy as np
from transformers import GPT2TokenizerFast

__all__ = ['tk','number_vocab']

integer_pattern = re.compile(r"-?\d+")
tk = GPT2TokenizerFast.from_pretrained('gpt2')
number_vocab = np.array(sorted([t for t in tk.vocab.keys() \
                                if integer_pattern.match(t)],
                               key=lambda x: int(x)))
number_vocab_int = np.array(sorted(set(number_vocab.astype(int))))

