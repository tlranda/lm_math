import re
import numpy as np
from transformers import GPT2TokenizerFast
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

possible_tokens = list(tokenizer.vocab.keys())
is_num = re.compile(r"-?\d+")
numbers = np.array(sorted(set([int(_) for _ in possible_tokens if is_num.match(_)])))
# Printing all numbers may be a bit crazy, use reduced format
front_breaks = [0] + (np.where(np.diff(numbers)>1)[0]+1).tolist()
back_breaks = (np.array(front_breaks[1:])-1).tolist() + [len(numbers)-1]
print(", ".join([f"[{numbers[fb]}-{numbers[bb]}]" if fb != bb else str(numbers[fb]) for (fb,bb) in zip(front_breaks,back_breaks)]))

