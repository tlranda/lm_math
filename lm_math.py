#!/usr/env/bin python3
"""
    FOR UP-TO-DATE USAGE: python3 lm_math.py --help

    Generate arithmetic expressions under various circumstances to tease out
    the likely memorized behaviors of LLMs and the capabilities LLMs can
    display (memorized or not).
"""
###############################################################################
#                                                                             #
#          Set up the operators and their string representations here         #
#                                                                             #
###############################################################################
# Builtin python imports
import operator
# Dependent python imports
# none
# Local file python imports
from special_operation import special_operation

# Friendly name the LLM may be allowed to read, needs to be an operator or
# callable name imported above into the global namespace
op_names = ['special_operation','add','mod','mul','pow','sub']
# Accurate sign corresponding to the friendly name
op_signs = ['†','+','%','*','^','-']
# Possibly inaccurate name shown to the LLM; any discrepancy will be explained
# to the LLM in the prompt.
op_use_signs = [_ for _ in op_signs]
###############################################################################
#                                                                             #
#           Below here are automated setups based on the above                #
#               just let the program set these up for you                     #
#                                                                             #
###############################################################################
op_calls = []
for op in op_names:
    try:
        func = getattr(operator,op)
    except AttributeError:
        try:
            func = globals()[op]
        except KeyError:
            raise NotImplemented(f"No known callable for operator '{op}'")
    op_calls.append(func)

# Builtin python imports (continued)
import argparse
import itertools
import json
import pathlib
import re
from typing import Dict, List, Optional, Tuple, Union
# Dependent python imports
import numpy as np
import tqdm
import ollama
from transformers import GPT2TokenizerFast
# Local file python imports
# none

LLM_MODEL = 'llama3'
LLM_SEEDS = [1,2024,104987552,404,1337,987654321,777,13,4898,10648]
OLLAMA_CONFIGS = []
# At some point, this tokenizer needs to become pickable - but for now we'll
# always assume that a GPT2 tokenizer is appropriate
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
# Make a vocab of numbers one time
number_vocab = np.array(sorted(set([int(_) for _ in tokenizer.vocab.keys()\
                                    if re.match(r"-?\d+", _)])))

def bitshift_list(i_value, max_bits=None):
    if max_bits is None:
        max_bits = len(OLLAMA_CONFIGS)
    n_values = 0
    blist = []
    while n_values < max_bits:
        blist.append((i_value>>n_values) % 2)
        n_values += 1
    return blist

def tokenize(str_: Union[str,List[str],dict,List[dict]],
             count: bool = False) -> Union[List[str], List[List[str]], int]:
    """
        Wrapper to the tokenizer that gives access to the actual substrings
        that form individual tokens from an input string.

        If count==True, only return the count (or sum count of list)
    """
    if isinstance(str_, str):
        # Some special characters can be encoded strangely, removing spaces
        # from the input string usually resolves the issue but is less accurate
        # to what the LLM perceives
        decoded = tokenizer.tokenize(str_)
        #encoded_ids = tokenizer.encode_plus(str_)['input_ids']
        #decoded = tuple([tokenizer.decode(_) for _ in encoded_ids])
        if count:
            return len(decoded)
        else:
            return decoded
    # Permit operating directly on a message dictionary
    if isinstance(str_, dict):
        return tokenize(str_['content'], count=count)
    # Permit recursion on lists in case I bulk-tokenize
    if isinstance(str_, list):
        all_strs = []
        for s_ in str_:
            all_strs.append(tokenize(s_, count=count))
        if count:
            return sum(all_strs)
        else:
            return all_strs
    raise NotImplemented

def natural_language_math(operands_list: List[int],
                          op: callable,
                          with_answer: bool = True,
                          aggregate: bool = True,
                          feature_expression: bool = False,
                          feature_names: List[str] = None)\
    -> List[Dict[str,str]]:
    """
        Take a list of operands (every 2 operands form an expression) and make
        the natural language expression of each expression based on the
        provided callable.

        `with_answer`: Include the answer by calling the operation to compute
                       the actual value and display it after the equals sign.
        `aggregate`: Combine examples using newlines as delimiters instead of
                     multiple separate messages.
        `feature_expression`: Convert from operand-operator-operand to a more
                              natural language format.
        `feature_names`: Replace 'operand_X' with a special name that may have
                         significance the LLM can meaningfully use to improve
                         its odds at generating valuable answers.
    """
    messages = []
    msg = ""
    for (op_left, op_right) in zip(operands_list[::2], operands_list[1::2]):
        # Always provide the expression
        if feature_expression:
            msg += f"operator is {op_use_signs[op_calls.index(op)]}, "
            msg += 'operand_0' if feature_names is None else feature_names[0]
            msg += f" is {op_left}, "
            msg += 'operand_1' if feature_names is None else feature_names[1]
            msg += f" is {op_right} | Answer = "
        else:
            msg += f"{op_left} {op_use_signs[op_calls.index(op)]} {op_right} = "
        # Optionally provide the answer
        if with_answer:
            msg += f"## {op(op_left,op_right)} ##"
        # Aggregation across loop bounds
        if aggregate:
            msg += "\n"
        else:
            messages.append({'role': 'user', 'content': msg})
            msg = ""
    # Cleanup for final loop / multiline aggregation
    if msg != "":
        messages.append({'role': 'user', 'content': msg})
    return messages

def show_last_results(results: List[int]) -> None:
    """
        I'm dumb and using bits to compress these results, so this helps you
        read the bits as a string in a debugging fashion
    """
    print(" | ".join([f"{k}: {bitshift_list(results[k][-1])}" for k in results]))

def generate_operands(n_to_create: int,
                      max_digits: int,
                      strict: bool,
                      include_negative: bool,
                      in_vocab: Optional[bool] = None) -> np.ndarray:
    """
        Create a number of operands with specified digit range and
        characteristics, including respect to token vocabulary

        `n_to_create`: Number of operands to create and return
        `max_digits`: Operand values will not exceed this number of base-10
                      digit lengths
        `strict`: When true, ALL operands will be maximum length; when false,
                  operands will be strictly lte maximum length
        `include_negative`: When true, operands may be negative-valued; when
                            false operands will be strictly non-negative
        `in_vocab`: When non-None, guarantee that operands are(n't) in the token
                    vocabulary based on the truthiness of this value
    """
    min_value = 0 if not strict else 10**(max_digits-1)
    max_value = 10**max_digits
    n_possible = (max_value - min_value) * (2 if include_negative else 1)
    if in_vocab is None:
        # Straightforward to select valid values via randint
        operands = np.random.randint(min_value, max_value, n_to_create)
        if include_negative:
            operands[np.where(np.random.rand(n_to_create) < 0.5)[0]] *= -1
    else:
        # Start with non-negative set
        gte_min = np.where(number_vocab >= min_value)[0]
        lt_max = np.where(number_vocab < max_value)[0]
        tokens_in_range = number_vocab[np.intersect1d(gte_min,lt_max,
                                                      assume_unique=True)]
        if include_negative:
            # Generally I expect negative values to not be individual tokens,
            # but check anyways
            gte_min = np.where(number_vocab <= -1*min_value)[0]
            lt_max = np.where(number_vocab > -1*max_value)[0]
            negative_range = number_vocab[np.intersect1d(gte_min,lt_max,
                                                         assume_unique=True)]
            # If empty it will upcast to float, which messes up hstack's type
            # as well
            negative_range = negative_range.astype(int)
            tokens_in_range = np.hstack((negative_range,tokens_in_range))
        if in_vocab:
            operands = np.random.choice(token_in_range, size=n_to_create)
        else:
            # Determine power and representations for best sampling strategy
            tokened = len(tokens_in_range) / n_possible
            if tokened == 1.0:
                # It's not possible to sample out-of-vocabulary for this setup
                raise ValueError(f"All {max_digit}-digit integers are in-token "
                                 f"vocabulary")
            elif tokened >= 0.2:
                # Rejection sampling would possibly be bad, and this list
                # should end up being short enough to not worry too much
                possible = set(range(min_value,max_value))
                if include_negative:
                    possible.add(set(range(-max_value,-min_value)))
                possible = sorted(possible.difference(set(tokens_in_range)))
                operands = np.random.choice(possible, size=n_to_create)
            else:
                # Rejection sampling should need a minimal number of trials
                selections = []
                while len(selections) < n_to_create:
                    picked = np.random.randint(min_value,max_value, n_to_create)
                    if include_negative:
                        picked[np.where(np.random.rand(n_to_create) < 0.5)[0]] *= -1
                    picked = [_ for _ in picked if _ not in tokens_in_range]
                    selections.extend(picked)
                operands = selections[:n_to_create]
    return operands

def llm_examples(op: callable,
                 tqdm_updater: tqdm.tqdm = None,
                 n_examples: Union[int,float] = 10,
                 token_limit: Optional[int] = None,
                 n_evals: int = 10,
                 max_digits: int = 2,
                 strict_digits: bool = False,
                 include_negative: bool = False,
                 no_prompt: bool = False,
                 with_context: bool = True,
                 use_both: bool = False,
                 feature_expression: bool = False,
                 substitution_operand: Union[None,Tuple[str,str]] = None,
                 encourage_rewrites: bool = False,
                 icl_in_vocab: Optional[bool] = None,
                 eval_in_vocab: Optional[bool] = None,
                 ) -> Dict[str, Dict[str, int]]:
    """
        Main driver function of the test battery this script intends to provide.
        The return dictionary is one of metric names mapping to distribution
        information. For now, that is:
            {Metric_Name: {
                    'Observed': Number_of_Metric_Successes,
                    'Out_of': Number_of_Metric_Trials,
                }
            }

        `op`: The callable function to utilize for these tests
        `tqdm_updater`: tqdm.tqdm object to update once per LLM generation-and-
                        parse
        `n_examples`: The number of examples to provide for ICL to the LLM,
                      if this value is a floating-point type, then tokens are
                      generated until an example would exceed this proprotion
                      of the model's context length
        `token_limit`: The number of tokens that fit the model's context
        `n_evals`: The number of different expressions given to the LLM for
                   evaluating across different inputs. NOTE that the LLM will
                   use each configuration PER eval, so the total number of LLM
                   calls produced by this function are actually:
                        len(OLLAMA_CONFIGS) * n_evals
                   Whether offline or online, plan your usage accordingly.
        `max_digits`: The maximum base-10 integers to generate in this
                      procedure are:
                        |10 ^ (max_digits)|
        `strict_digits`: Guarantee maximum digits are used
        `include_negative`: Permits negative operand values
        `no_prompt`: Do not supply any prompts explaining the task to perform
        `with_context`: Generate ICL using natural_language_math(), to be
                        presented to the LLM prior to each task as ground-truth
                        examples. This REPLACES the system prompt as
                        instructions, so ICL is all that the LLM will have for
                        context of its task to complete.
        `use_both`: Generate ICL using natural_language_math() and ADD this as
                    additional instructions for the LLM to support proper task
                    fulfillment.
        `feature_expression`: Passed directly to uses of natural_language_math()
        `substitution_operand`: When provided, an exchanged pair of strings to
                                explain to the LLM that notation generated by
                                natural_language_math() may use unfamiliar
                                operators (typical_string, actual_string). This
                                gives the LLM additional prompting to explain
                                this substitution and encourages it to take
                                measures that increase its likelihood of
                                overcoming the substitution and answering
                                correctly.
        `encourage_rewrites`: When a substitution is made, encourage the LLM
                              to rewrite the expression to its ground-truth
                              prior to submitting its answer. This should
                              benefit the LLM by permitting its memorization
                              to work more effectively than a completely
                              context-based symbol substitution.
        `icl_in_vocab`: Prefer operands to be (in/out of) vocabulary single
                        tokens (unless None, wherein no preference) for ICL
        `eval_in_vocab`: Prefer operands to be (in/out of) vocabulary single
                         tokens (unless None, wherein no preference) for evaluations
    """
    # ENSURE RANDOM GENERATIONS ARE CONSISTENT ACROSS REPEATED PROGRAM
    # INVOKATIONS
    np.random.seed(1)
    token_expenditure = 0
    count_tokens = token_limit is not None and isinstance(n_examples, float)

    ###########################################################################
    #                                                                         #
    #                        Create the system prompt                         #
    #                                                                         #
    ###########################################################################

    # System message (conditionally included in LLM prompt)
    system_messages = [{'role': 'system',
                        'content': 'You are a helpful AI assistant. When '
                                   'presented with a mathematical expression '
                                   'or equation, respond with the numeric '
                                   'value that correctly completes the input '
                                   'surrounded by "##" characters, ie: '
                                   '## 1 ##.'}]
    # The special operation should be impossible for the LLM to observe in its
    # training data; ergo it is only fair to explain to the LLM what this
    # operation is generally anticipated to mean. However, we're using this as
    # a check against memorizing ICL data, so an exact representation is
    # unnecessary
    op_explanation = ""
    if op == special_operation:
        op_explanation = f'The \'{op_use_signs[op_calls.index(op)]}\' symbol '
        op_explanation += 'represents a special nonlinear operation that always '
        'returns integer values. Do not attempt to explain '
        'what it does, only provide your best guess of what '
        'the correct value for this operation is given the '
        'provided inputs.'
        system_messages[0]['content'] += " "+op_explanation
    elif substitution_operand is not None and \
         substitution_operand[0] != substitution_operand[1]:
        # For non-special operations, if there's an indication that a
        # nontraditional symbol/name for the operator will be used by
        # natural_language_math(), we can explain to the LLM what substitution
        # occurs.
        # This allows us to see if text manipulation can overcome a harder
        # version of memorizing, where special contextual rules override
        # the commonly memorized patterns.
        op_explanation = f"As an added twist, the {substitution_operand[1]} "+\
                        "symbol will be used in place of "+\
                        f"{substitution_operand[0]}. Please consider every "+\
                        f"use of {substitution_operand[1]} to be "+\
                        f"{substitution_operand[0]}"
        if encourage_rewrites:
            op_explanation += " and rewrite the operation replacing uses of "+\
                        f"{substitution_operand[1]} with "+\
                        f"{substitution_operand[0]} prior to indicating your "+\
                        "answer."
        system_messages[0]['content'] += " "+op_explanation
    # Only contributes to count if we utilize the system messages
    if count_tokens and use_both or not with_context:
        token_expenditure += tokenize(system_messages, count=True)
        print(f"Token count for system messages: {token_expenditure}")

    ###########################################################################
    #                                                                         #
    #                    Create the evaluation prompts                        #
    #                                                                         #
    ###########################################################################

    # Challenges for the LLM to solve are generated and converted into natural
    # language prompts based on desired settings
    prompt_operands = generate_operands(n_evals*2, max_digits, strict_digits,
                                        include_negative, in_vocab=eval_in_vocab)
    prompt_queue = natural_language_math(prompt_operands,
                        op,
                        with_answer=False, # Challenges never include answer
                        aggregate=False, # Only ask LLM one question at a time
                        feature_expression=feature_expression)
    # Ground-truth answers to check LLM against
    prompt_answers = [op(pops_l, pops_r) for (pops_l, pops_r) \
                      in zip(prompt_operands[::2], prompt_operands[1::2])]
    # Ensure every prompt fits within context, but only one at a time is
    # necessary so only add the longest prompt to the token count
    if count_tokens:
        prompt_lengths = [tokenize(prompt, count=True) for prompt in prompt_queue]
        longest_prompt = np.argmax(prompt_lengths)
        token_expenditure += prompt_lengths[longest_prompt]
        print(f"Longest evaluation prompt adds {prompt_lengths[longest_prompt]}"
              f" tokens, running total: {token_expenditure}")

    ###########################################################################
    #                                                                         #
    #                         Create the ICL prompts                          #
    #                                                                         #
    ###########################################################################

    icl_queue = []
    # This logic combination DOESN'T have the system message, so ensure that
    # the special instructions are passed along via the ICL mechanism
    if op_explanation != "" and with_context and not use_both:
        icl_queue = [{'role': 'system', 'content': op_explanation}]
        if count_tokens:
            op_explanation_tokens = tokenize(op_explanation, count=True)
            token_expenditure += op_explanation_tokens
            print(f"Operation requires explanation, adding {op_explanation_tokens} "
                  f"tokens, running total: {token_expenditure}")
    while (with_context or use_both) and \
          (count_tokens or (len(icl_queue) < n_examples)):
        icl_operands = generate_operands(2, max_digits, strict_digits,
                                         include_negative, in_vocab=icl_in_vocab)
        icl = natural_language_math(icl_operands,
                                    op,
                                    with_answer=True, # ICL always includes answer
                                    feature_expression=feature_expression)
        if count_tokens:
            new_prompt_len = tokenize(icl, count=True)
            if token_expenditure + new_prompt_len >= token_limit:
                break
            token_expenditure += new_prompt_len
            print(f"ICL prompt {len(icl_queue)+1} adds {new_prompt_len} tokens, "
                  f"running total: {token_expenditure}")
        icl_queue.extend(icl)

    ###########################################################################
    #                                                                         #
    #                       Template for evaluations                          #
    #                                                                         #
    ###########################################################################

    if no_prompt:
        initial_mstate = []
    elif use_both:
        initial_mstate = system_messages + icl_queue
    elif with_context:
        initial_mstate = icl_queue
    else:
        initial_mstate = system_messages
    print("Basic prompt without query:")
    print("".join([_['content'] for _ in initial_mstate]))
    if count_tokens:
        print(f"Counted tokens: {token_expenditure} / {token_limit}")
        post_hoc = tokenize(initial_mstate+[prompt_queue[longest_prompt]],
                            count=True)
        print(f"Post-hoc count: {post_hoc}")

    ###########################################################################
    #                                                                         #
    #                        Prepare for evaluations                          #
    #                                                                         #
    ###########################################################################

    # Prepare results to track through testing
    result_metrics = ['follows_regex', # The LLM responds as requested with
                                       # ONLY: '## <number> ##'
                      'parseable', # Extended parsing rules are able to recover
                                   # a value from the LLM response that can be
                                   # evaluated
                      'correct', # The parsed value from the LLM response is
                                 # exactly the same as the ground-truth value.
                                 # For an LLM to have a mathematically-useful
                                 # capability to perform an operation via
                                 # natural language, 'close' does not count,
                                 # only EXACT matches are useful.
                      ]
    if use_both or with_context:
        # These metrics only make sense if ICL is provided
        result_metrics += [
                      'memorized_verbatim', # Indicates that the LLM may have
                                            # identified an example from ICL
                                            # in its challenge set and
                                            # successfully reproduced the ICL
                                            # answer.
                      'memorized_copycat', # Indicates that the LLM appears to
                                           # have copied an ICL answer, though
                                           # the challenge is not a direct
                                           # copy of an ICL example. Note that
                                           # even though some operations are
                                           # commutative, the VERBATIM/COPYCAT
                                           # checks are NOT commutative.
                      ]
    results = dict((k,list()) for k in result_metrics)
    # Set up the information for memorization checks when ICL is provided
    known_values = dict()
    if use_both or with_context:
        print("Parsing for each contextual example")
        for example_msg in icl_queue:
            examples = example_msg['content'].split('\n')
            for example in examples:
                try:
                    trial = tuple(tokenize(example[:example.index('=')+2]))
                    answer = tuple(tokenize(example[example.index('=')+2:]))
                except ValueError:
                    continue
                print(trial,'-->',answer)
                if answer in known_values:
                    known_values[answer].append(trial)
                else:
                    known_values[answer] = [trial]

    ###########################################################################
    #                                                                         #
    #                          Perform evaluations                            #
    #                                                                         #
    ###########################################################################

    for (trial, answer) in zip(prompt_queue, prompt_answers):
        # Update for another trial
        for k in results:
            results[k].append(0)
        # Add this trial's prompt to the queue of messages for the LLM
        mstate = initial_mstate + [trial]
        # Reset trial tokenized
        llm_trial_tokenized = None
        print(f"TRUTH: {trial['content']}## {answer} ##")
        for (attempt,CONFIG) in enumerate(OLLAMA_CONFIGS):
            # The LLM produces its answer
            print(f"LLM Attempt {attempt+1}/{len(OLLAMA_CONFIGS)}")
            response = ollama.chat(model=LLM_MODEL,
                                   messages=mstate,
                                   options=CONFIG)
            print(f"Response {attempt+1}/{len(OLLAMA_CONFIGS)}: "
                  f"{trial['content']}{response['message']['content']}")
            # We check if the LLM memorized this answer
            llm_answer_tokenized = tokenize(response['message'])
            print("\t",f"Tokenized: {llm_answer_tokenized}")
            if len(known_values) > 0 and \
                tuple(llm_answer_tokenized) in known_values:
                if llm_trial_tokenized is None:
                    llm_trial_tokenized = tuple(tokenize(trial))
                if llm_trial_tokenized in known_values[llm_answer_tokenized]:
                    print("This is a verbatim memorized answer")
                    results['memorized_verbatim'][-1] += (1 << attempt)
                else:
                    print("This is a previously-seen memorized answer "
                          "(copycat), but it is answering a different trial "
                          "input")
                    results['memorized_copycat'][-1] += (1 << attempt)
            # We check if the LLM followed our requested output format and
            # attempt to parse a comparable value to verify correctness
            regex_portion = re.findall(r"## ?(-?[0-9]+) ?##",
                                       response['message']['content'])
            if regex_portion:
                results['follows_regex'][-1] += (1 << attempt)
                # No try/except here, following the regex SHOULD make this
                # parse, and if not I want the program to crash so I can see
                # the bug. We choose the last match in case the LLM repeated
                # some ICL examples.
                llm_answer = int(regex_portion[-1])
                print("LLM followed requested output format and gave answer:",
                      llm_answer)
            else:
                # Sometimes the LLM responds with JUST a number.
                # Sometimes it likes to yap a little bit and then give its
                # answer.
                # Sometimes we explicitly instruct the LLM to rephrase things
                # before giving its answer.
                # In all-of-the-above cases, we reasonably expect the FINAL
                # number produced by the LLM to be its actual answer, even if
                # it talks further than the final number.
                #
                # NOTE: If you limit the number of tokens in the LLM response,
                # this might successfully parse a number that is clearly not
                # the LLM's answer because its response was truncated. We
                # do not defensively program around this case, but YOU should
                # be aware of it, especially on tight token limits or if your
                # LLM is producing mountains of text for some reason.
                try:
                    final_match = re.findall(r"\D*?(-?\d+)",
                                             response['message']['content'])[-1]
                    llm_answer = int(final_match)
                    print("LLM DID NOT follow requested output format, but we "
                          "recovered this value as its possible answer:",
                          llm_answer)
                except:
                    print("LLM DID NOT follow requested output format, and we "
                          "were UNABLE to recover any value as its possible "
                          "answer")
                    if tqdm_updater is not None:
                        tqdm_updater.update(n=1)
                    continue
            results['parseable'][-1] += (1 << attempt)
            print("LLM's answer is ", end='')
            if llm_answer == answer:
                print("correct")
                results['correct'][-1] += (1 << attempt)
            else:
                print("INCORRECT")
            if tqdm_updater is not None:
                tqdm_updater.update(n=1)
        show_last_results(results)

    ###########################################################################
    #                                                                         #
    #                  Summarize results for logging and return               #
    #                                                                         #
    ###########################################################################

    # Prepare results for return/logging
    final_results = dict((k,dict()) for k in results.keys())
    for (k,v) in results.items():
        # Each trial has 1-bits to indicate per-LLM-config pass/fail.
        # Show the sum of trial passes and compare to the total number of
        # possible passes.
        final_results[k]['observe'] = sum([sum(bitshift_list(vv)) for vv in v])
        final_results[k]['out_of'] = len(OLLAMA_CONFIGS)*len(v)
    return final_results

def result_wrapper(results: dict,
                   file_target: str,
                   call: callable,
                   *args, **kwargs) -> None:
    """
        Make a call with args/kwargs, then update a cumulative dictionary with
        the results of that call based on the given arguments.
        Log these results to a given filename in case the program is
        interrupted.

        We do not write to temporary->move after successfully writing, which
        would be far more fault-tolerant. We expect the call to dominate runtime
        and for interruptions to generally land there, ergo the odds of partial
        file write corruption are pretty minimal.

        If it bothers you, by all means patch it in.
    """
    new_val = call(*args, **kwargs)
    # JSON cannot cope with many args/kwargs in a normal fashion, so make sure
    # the key is represented as a string that reasonably parses for humans
    new_key = str((*[a if not callable(a) else a.__name__ for a in args],
                   *[f"{k if not callable(v) else k.__name__}:{v}" for (k,v) \
                        in kwargs.items() if not isinstance(v, tqdm.tqdm)]))
    results[new_key] = new_val
    # Log this data to disk
    with open(file_target,'w') as f:
        json.dump(results,f,indent=1)

if __name__ == '__main__':
    """
        Arguments for execution
    """
    prs = argparse.ArgumentParser()
    prs.add_argument('file',
                     help="Path to log results to (in JSON format)")
    llmconf = prs.add_argument_group("Problem High-Level Settings")
    llmconf.add_argument("--n-examples", type=str, default=10,
                     help="Number of ICL examples for the LLM (prepend with "
                          "'f' for proportion of context length, which must "
                          "also be given via --context-length) (default: "
                          "%(default)s)")
    llmconf.add_argument("--n-evals", type=int, default=10,
                     help="Number of examples LLM is prompted with per setting "
                          "(default: %(default)s)")
    llmconf.add_argument("--model", default=LLM_MODEL,
                     help="Ollama LLM to use for inference (default: "
                          "%(default)s)")
    llmconf.add_argument("--context-length", default=None, type=int,
                     help="Context length for the model (must be specified to "
                          "use floating-point --n-examples)")
    llmconf.add_argument("--seeds", type=int, default=None, nargs="*",
                     help="LLM seeds to use--each seed is used on EVERY "
                          f"evaluation (default: {LLM_SEEDS})")
    llmconf.add_argument("--select-seeds", type=int, default=None, nargs="*",
                     help="Select subset of default LLM seeds via 0-indexing. "
                          "Alternative to --seeds (default: %(default)s)")
    opconf = prs.add_argument_group("Operator Settings")
    opconf.add_argument("--operators", choices=op_names, default=None, nargs='+',
                     required=True, help="Operators to test the LLM against")
    opconf.add_argument("--symbols", type=str, default=None, nargs='*',
                     help="Symbols to use for each operator. Use 'DEFAULT' to "
                          "utilize the default symbol while overriding others "
                          "(default: "
                          f"{[_ if _ != '%' else '%%' for _ in op_signs]})")
    exconf = prs.add_argument_group("Example Configurations (repeated per "
                                    "evaluation)")
    exconf.add_argument('--n-digits', type=int, nargs='+', required=True,
                     help="Maximum number of digits to use in operands")
    exconf.add_argument('--strict', choices=['True','False'], nargs='+',
                     required=True,
                     help="Force maximum number of digits to be used")
    exconf.add_argument('--negatives', choices=['True','False'], nargs='+',
                     required=True,
                     help="Permit negative values")
    exconf.add_argument('--icl-in-vocab', choices=['None','True','False'],
                     nargs='*', default=None, help="Force in-context learning "
                     "values to be (None=any | True=in | False=out-of) "
                     "vocabulary")
    exconf.add_argument('--eval-in-vocab', choices=['None','True','False'],
                     nargs='*', default=None, help="Force evaluation prompt "
                     "values to be (None=any | True=in | False=out-of) "
                     "vocabulary")
    exconf.add_argument('--features', choices=['True','False'], nargs='+',
                     required=True,
                     help="Express prompt/ICL using features instead of purer "
                     "math notations")
    exconf.add_argument('--encourage-rewrites', choices=['True','False'],
                     nargs='+', required=True,
                     help="Include encouragement for LLM to rewrite expressions "
                     "to use normal operators if substitutions exist")
    exconf.add_argument("--prompts", choices=['icl','system','icl+system','none'],
                     default=None, nargs='*',
                     help="Prompt styles to use (default: ALL)")
    args = prs.parse_args()
    if pathlib.Path(args.file).exists():
        raise ValueError(f"File '{args.file}' exists and would be overwritten!")
    LLM_MODEL = args.model
    if args.seeds is not None:
        LLM_SEEDS = args.seeds
    elif args.select_seeds is not None:
        LLM_SEEDS = [LLM_SEEDS[_] for _ in args.select_seeds]
    if args.prompts is None:
        args.prompts = ['icl','system','icl+system']
    # Special case: n-examples is integer unless prepended with 'f'
    if args.n_examples.startswith('f'):
        if args.context_length is None:
            raise ValueError("--context-length MUST be specified to use "
                             "proportion of context length for examples!")
        args.n_examples = float(args.n_examples[1:])
        args.token_limit = int(args.context_length * args.n_examples)
    else:
        args.n_examples = int(args.n_examples)
    if args.icl_in_vocab is None:
        args.icl_in_vocab = ['None']
    if args.eval_in_vocab is None:
        args.eval_in_vocab = ['None']
    # Boolean conversions
    for argname in ['strict','negatives','features','encourage_rewrites',
                    'icl_in_vocab','eval_in_vocab']:
        setattr(args, argname, [None if a == 'None' else a == 'True' \
                                for a in getattr(args, argname)])
    if args.symbols is not None:
        if len(args.operators) != len(args.symbols):
            raise ValueError("Must use default symbols or provide one symbol "
                             "per selected operator!")
        for opname, symb in zip(args.operators, args.symbols):
            if symb == 'DEFAULT':
                continue
            op_use_signs[op_names.index(opname)] = symb
    # Make the substitutions list now that user-defined symbol remaps are available
    substitutions = [(op_signs[op_names.index(name)],
                      op_use_signs[op_names.index(name)])
                      for name in args.operators]
    used_op_calls = [op_calls[op_names.index(name)] for name in args.operators]

    # Create the options list
    for seed in LLM_SEEDS:
        """
            LLM options
            * top_p :       for nucleus sampling, discard the worst-5% of
                            tokens from consideration
            * temperature : for 'creativity', set to be somewhat creative but
                            mostly adhere to the prompt
            * num_predict : to limit excessive tangents when the model is
                            distracted, this is far more tokens than it needs
                            to solve these problems
            * seed :        to ensure consistency between repeated program
                            executions
        """
        OLLAMA_CONFIGS.append(ollama.Options(top_p=0.95,
                                             temperature=0.7,
                                             num_predict=None,
                                             seed=seed))
    # Number of LLM generations per call
    llm_generations = len(OLLAMA_CONFIGS) * args.n_evals
    print(args)
    print(OLLAMA_CONFIGS)
    # Precompute loop iteration counters so we can tqdm it
    n_to_compute =  llm_generations * len(args.operators) * len(args.n_digits) *\
                    len(args.negatives) * len(args.features) *\
                    len(args.encourage_rewrites) * len(args.icl_in_vocab) *\
                    len(args.eval_in_vocab) * len(args.prompts)
    overall_progress = tqdm.tqdm(total=n_to_compute)
    # Cumulative results of program execution
    results = dict()
    # Nest allllll the loops
    for (name, op, substitution) in zip(args.operators, used_op_calls, substitutions):
        # ... using itertools product
        for (N_DIGITS, NEGATIVE, STRICTNESS, FEATURE_EXPRESSION, REWRITE,
             ICL_IN_VOCAB, EVAL_IN_VOCAB, PROMPT_SETTING) in \
                itertools.product(args.n_digits, args.negatives, args.strict,
                                  args.features, args.encourage_rewrites,
                                  args.icl_in_vocab, args.eval_in_vocab,
                                  args.prompts):
            print(f"Trying LLM on operator '{name}' with {PROMPT_SETTING} "
                  f"prompt style for {'strictly' if STRICTNESS else 'up to'} "
                  f"{N_DIGITS} digit values that "
                  f"{'are not' if not NEGATIVE else 'may be'} "
                  f"negative ({'no' if ICL_IN_VOCAB is None else ICL_IN_VOCAB} "
                  "vocab preference for ICL; "
                  f"{'no' if EVAL_IN_VOCAB else EVAL_IN_VOCAB} vocab preference"
                  " for evaluations) using text expressions in "
                  f"{'mathematical' if not FEATURE_EXPRESSION else 'feature'} "
                  f"notation{' with rewriting' if REWRITE else ''}.")
            if PROMPT_SETTING == 'icl':
                bonus_kwargs = {}
            elif PROMPT_SETTING == 'system':
                if name == 'special_operation':
                    print("Skipping special_operation on system"
                          " message only -- nothing of value "
                          "to learn.")
                    overall_progress.update(n=1)
                    continue
                bonus_kwargs = {'with_context': False}
            elif PROMPT_SETTING == 'icl+system':
                bonus_kwargs = {'use_both': True}
            elif PROMPT_SETTING == 'none':
                bonus_kwargs = {'no_prompt': True}
            else:
                raise NotImplemented(PROMPT_SETTING)
            result_wrapper(results,
                            args.file,
                            llm_examples,
                            op,
                            tqdm_updater=overall_progress,
                            n_examples=args.n_examples,
                            token_limit=args.token_limit,
                            n_evals=args.n_evals,
                            max_digits=N_DIGITS,
                            strict_digits=STRICTNESS,
                            include_negative=NEGATIVE,
                            feature_expression=FEATURE_EXPRESSION,
                            substitution_operand=substitution,
                            encourage_rewrites=REWRITE,
                            icl_in_vocab=ICL_IN_VOCAB,
                            eval_in_vocab=EVAL_IN_VOCAB,
                            **bonus_kwargs)
    print(results)

