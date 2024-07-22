import pathlib
import re
pound_extract = re.compile(r".*?(\d+)")
equation_extract = re.compile(r".*?(\d+ \+ \d+)")
truth = dict()
entry = 0
#icl_file = 'icl_examples.txt'
icl_file = 'icl_2.txt'
with open(icl_file,'r') as f:
    for line in f.read().splitlines():
        if len(line) > 0:
            equation, result = line.split('=',1)
            equation = equation_extract.match(equation).groups()[0]
            result = int(pound_extract.match(result).groups()[0])
            truth[equation] = (entry, result)
            entry += 1
print(f"ICL provides {len(truth)} plagiarisable answers")
print(truth)

#seedfiles = [f"seed_{_}.txt" for _ in range(10)]
seedfiles = ["seed.txt"]
for seedfile in seedfiles:
    if not pathlib.Path(seedfile).exists():
        continue
    print(f"Working on seed file {seedfile}")
    with open(seedfile,'r') as f:
        parsing = True
        agree = []
        disagree = []
        new = []
        for lidx, line in enumerate(f.read().splitlines()):
            if len(line) < 1:
                continue
            if line.startswith('Response'):
                parsing = True
                answer = line.split('=',1)[1]
                pe = pound_extract.match(answer)
                if pe is not None and len(pe) > 0:
                    print("Maybe missed one")
            elif parsing:
                try:
                    eq, ans = line.split('=',1)
                except:
                    print(f"Skipping line '{line}'")
                    continue
                try:
                    eq = equation_extract.match(eq).groups()[0]
                    ans = int(pound_extract.match(ans).groups()[0])
                except:
                    print(f"Failed to parse line {lidx} '{line}'")
                if eq in truth:
                    #print(f"Line {lidx} '{line}' is ICL {truth[eq][0]}")
                    if truth[eq][1] == ans:
                        agree += [truth[eq][0]]
                        #print(f"LLM answer agrees with ICL")
                    else:
                        disagree += [truth[eq][0]]
                        #print(f"LLM answer '{ans}' disagrees with ICL answer '{truth[eq][1]}'")
                else:
                    new += [lidx]
                    #if ans == 272:
                    if ans == 472:
                        print("LLM answers new prompt correctly")
        print(f"LLM agrees {agree}")
        print(f"LLM disagrees {disagree}")
        print(f"LLM new {new}")

