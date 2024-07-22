import re
import numpy as np
from transformers import GPT2TokenizerFast
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

possible_tokens = list(tokenizer.vocab.keys())
is_num = re.compile(r"-?\d+")
numbers = np.array(sorted(set([int(_) for _ in possible_tokens if is_num.match(_)])))
MAX_FINITE_COMPARE = 1000
all_numbers = range(0,min(max(numbers), MAX_FINITE_COMPARE))
over_compare_limit = numbers[np.where(numbers >= MAX_FINITE_COMPARE)[0]]

not_included = set(list(all_numbers)).difference(set(numbers))
print(f"{len(not_included)} missing numbers <{MAX_FINITE_COMPARE}:", not_included)
print(f"{len(over_compare_limit)} present numbers >={MAX_FINITE_COMPARE}:", over_compare_limit)
