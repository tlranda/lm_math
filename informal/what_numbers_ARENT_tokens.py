import re
import time
import numpy as np
from transformers import GPT2TokenizerFast

# Step 1: get tokenizer and find all numbers in its vocabulary
toke_start = time.time()
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
possible_tokens = list(tokenizer.vocab.keys())
is_num = re.compile(r"-?\d+")
numbers = np.array(sorted(set([int(_) for _ in possible_tokens if is_num.match(_)])))
toke_stop = time.time()
print(f"Vocab identification time: {toke_stop-toke_start:.6f}s")

# Step 2: bucket by powers of 10
bucket_start = time.time()
limit = 1
digit_dict = dict()
used_idx = 0
while used_idx < len(numbers):
    sub_numbers = numbers[used_idx:]
    present = sub_numbers[np.where(sub_numbers < limit)[0]]
    digit_dict[limit] = present 
    used_idx += len(present)
    print(limit, digit_dict[limit])
    limit *= 10
bucket_stop = time.time()
print(f"Bucketizing time: {bucket_stop-bucket_start:.6f}s")

# Step 3: Access based on buckets
N_SAMPLE = 10
access_start = time.time()
for limit, bucket in digit_dict.items():
    n_possible_items = max(1,limit*9//10)
    print(f"Bucket {limit} can hold {n_possible_items} items, has {len(bucket)} items")
    # Special case, full buckets have to be skipped
    if n_possible_items == len(bucket):
        print(f"All possible values are in the bucket")
    elif len(bucket)/n_possible_items >= 0.2:
        print(f"Sample against items NOT in the bucket")
        absent = sorted(set(range(limit//10, limit)).difference(set(bucket)))
        selections = np.random.choice(absent,size=N_SAMPLE)
        print(f"Picked: {selections}")
    else:
        print(f"Sample against items in range, resample any items in bucket")
        selections = []
        n_rounds = 0
        while len(selections) < N_SAMPLE:
            picked = [_ for _ in np.random.randint(limit//10, limit, N_SAMPLE) if _ not in bucket]
            selections.extend(picked)
            n_rounds += 1
        selections = selections[:N_SAMPLE]
        print(f"Picked in {n_rounds} rounds: {selections}")
"""
MAX_FINITE_COMPARE = 1000
all_numbers = range(0,min(max(numbers), MAX_FINITE_COMPARE))
over_compare_limit = numbers[np.where(numbers >= MAX_FINITE_COMPARE)[0]]

not_included = set(list(all_numbers)).difference(set(numbers))
print(f"{len(not_included)} missing numbers <{MAX_FINITE_COMPARE}:", not_included)
print(f"{len(over_compare_limit)} present numbers >={MAX_FINITE_COMPARE}:", over_compare_limit)
"""
access_stop = time.time()
print(f"Access time: {access_stop-access_start:.6f}s")

