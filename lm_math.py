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
# Add any new operators/functionalities by following instructions in
# `op_extensions.py`
import op_extensions

# Name (which is also an argument), sign (used to represent the operation), and
# call to execute the operation
operator_names = ['add','mod','mul','pow','sub']
operator_signs = ['+','%','*','^','-']
operator_calls = [getattr(operator,op) for op in operator_names]
# Extend with local extensions
op_names = operator_names + op_extensions.names
op_signs = operator_signs + op_extensions.signs
op_calls = operator_calls + op_extensions.calls
# A runtime argument allows any operation's sign to be re-aliased; this list
# has the default values (the correct indiciated sign)
op_use_signs = [_ for _ in op_signs]

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
import lm_tokenizers

LLM_MODEL = 'llama3'
LLM_SEEDS = [1,2024,104987552,404,1337,987654321,777,13,4898,10648]
OLLAMA_CONFIGS = []
tokenizer = None
number_vocab_int = None

def tokenize(str_: Union[str,List[str],dict,List[dict]],
             count: bool = False) -> Union[List[str], List[List[str]], int]:
    """
        Wrapper to the tokenizer that gives access to the actual substrings
        that form individual tokens from an input string.

        `str_`: The string-like data (or nested structure with string-like data)
                to tokenize. Dictionaries are expected to be LLM-JSON responses
        `count`: Return the count of tokens composing the `str_` data rather
                 than the series of tokens
    """
    if isinstance(str_, str):
        # Not every tokenizer directly supports a count argument, so just count
        # the length of the returned list
        decoded = tokenizer.tokenize(str_)
        if count:
            return len(decoded)
        else:
            return decoded
    # Permit operating directly on a message dictionary
    if isinstance(str_, dict):
        return tokenize(str_['content'], count=count)
    # Permit recursion on lists to support bulk-tokenizing
    if isinstance(str_, list):
        all_strs = []
        for s_ in str_:
            all_strs.append(tokenize(s_, count=count))
        if count:
            return sum(all_strs)
        else:
            return all_strs
    raise NotImplemented

def natural_language_math(operands_list: List[object],
                          op: callable,
                          answer: object,
                          with_answer: bool = True,
                          feature_expression: bool = False,
                          feature_names: List[str] = None,
                          ) -> List[Dict[str,str]]:
    """
        Take a list of operands (every 2 operands form an expression) and make
        the natural language expression of each expression based on the
        provided callable.

        `operands_list`: List of inputs for expressions
        `with_answer`: Include the answer by calling the operation to compute
                       the actual value and display it after the equals sign.
        `feature_expression`: Convert from operand-operator-operand to a more
                              natural language format.
        `feature_names`: Replace 'operand_X' with a special name that may have
                         significance the LLM can meaningfully use to improve
                         its odds at generating valuable answers.
    """
    msg = ""
    if feature_expression:
        msg += f"operator is {op_use_signs[op_calls.index(op)]}, "
        msg += ", ".join([f"operand_{idx}" if name is None else name \
                          + f" is {val}"
                          for idx, (val, name) in \
                            enumerate(zip(operands_list, feature_names))])
        msg += " output is "
    else:
        msg += f"{' '.join([str(_) for _ in operands_list[:len(operands_list)//2]])}"
        msg += f" {op_use_signs[op_calls.index(op)]} "
        msg += f"{' '.join([str(_) for _ in operands_list[len(operands_list)//2:]])}"
        msg += " = "
    if with_answer:
        msg += f"## {answer} ##"
    return msg

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
        gte_min = np.where(number_vocab_int >= min_value)[0]
        lt_max = np.where(number_vocab_int < max_value)[0]
        tokens_in_range = number_vocab_int[np.intersect1d(gte_min,lt_max,
                                                      assume_unique=True)]
        if include_negative:
            # Generally I expect negative values to not be individual tokens,
            # but check anyways
            gte_min = np.where(number_vocab_int <= -1*min_value)[0]
            lt_max = np.where(number_vocab_int > -1*max_value)[0]
            negative_range = number_vocab_int[np.intersect1d(gte_min,lt_max,
                                                         assume_unique=True)]
            # If empty it will upcast to float, which messes up hstack's type
            # as well
            negative_range = negative_range.astype(int)
            tokens_in_range = np.hstack((negative_range,tokens_in_range))
        if in_vocab:
            operands = np.random.choice(tokens_in_range, size=n_to_create)
        else:
            # Determine power and representations for best sampling strategy
            tokened = len(tokens_in_range) / n_possible
            if tokened == 1.0:
                # It's not possible to sample out-of-vocabulary for this setup
                raise ValueError(f"All {max_digits}-digit integers are in-token "
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

def make_system_prompt(op: callable,
                       with_context: bool,
                       use_both: bool,
                       substitution_operand: Optional[Tuple[str,str]],
                       encourage_rewrites: bool,
                       token_expenditure: int,
                       count_tokens: bool) -> Tuple[Dict[str,str],str,int]:
    """
        Write the system prompt dictionary based on relevant settings.
        Return this dictionary, an operation explanation, and the updated token
        expenditure count.

        `op`: The callable function to utilize for these tests
        `with_context`: Generate ICL using natural_language_math, to be
                        presented to the LLM prior to each task as ground-truth
                        examples. This REPLACES the system prompt as
                        instructions, so ICL is all that the LLM will have for
                        context of its task to complete.
        `use_both`: Generate ICL using natural_language_math and ADD this as
                    additional instructions for the LLM to support proper task
                    fulfillment.
        `substitution_operand`: When provided, an exchanged pair of strings to
                                explain to the LLM that notation generated by
                                natural_language_math may use unfamiliar
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
        `token_expenditure`: Current token usage count
        `count_tokens`: Update the count of tokens based on content of the system messages
    """
    # Default system message (conditionally included in LLM prompt)
    system_messages = [{'role': 'system',
                        'content': 'You are a helpful AI assistant. You will '
                                   'be presented with a series of one or more '
                                   'expressions. Assume all given examples are '
                                   'correct, then use your knowledge to '
                                   'respond with the correct answer for the '
                                   'challenge expression that is left '
                                   'unanswered. Your answer for the challenge '
                                   'should be surrounded by "##" characters, '
                                   'ie: ## 1 ##.\n'}]
    # The special operation should be impossible for the LLM to observe in its
    # training data; ergo it is only fair to explain to the LLM what this
    # operation is generally anticipated to mean. However, we're using this as
    # a check against memorizing ICL data, so an exact representation is
    # unnecessary
    op_explanation = ""
    if op in op_extensions.calls:
        op_explanation = f'The \'{op_use_signs[op_calls.index(op)]}\' symbol '
        op_explanation += op_extensions.description
        if not op_explanation.endswith('\n'):
            op_explanation += "\n"
    elif substitution_operand is not None and \
         substitution_operand[0] != substitution_operand[1]:
        # For non-special operations, if there's an indication that a
        # nontraditional symbol/name for the operator will be used by
        # natural_language_math, we can explain to the LLM what substitution
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
    system_messages[0]['content'] += op_explanation
    # Only contributes to count if we utilize the system messages
    if count_tokens and use_both or not with_context:
        token_expenditure += tokenize(system_messages, count=True)
        print(f"Token count for system messages: {token_expenditure}")
    return system_messages, op_explanation, token_expenditure

def make_prompts(op: callable,
                 n_prompts: int,
                 with_answer: bool,
                 aggregate: bool,
                 max_digits: int,
                 strict_digits: bool,
                 include_negative: bool,
                 feature_expression: bool,
                 require_in_vocab: bool,
                 first_extension: Optional[str] = None,
                 each_extension: Optional[str] = None,
                 limit: Optional[int] = None) -> Tuple[List[Dict[str,str]], List[int], int, int]:
    """
        Create prompts for evaluations, return the prompt dict, the answers
        and the total token count as well as the longest prompt's token count

        `op`: The callable function to utilize for these tests
        `n_prompts`: The number of different expressions given to the LLM for
                     evaluating across different inputs. NOTE that the LLM will
                     use each configuration PER eval, so the total number of LLM
                     calls produced by this function are actually:
                        len(OLLAMA_CONFIGS) * n_prompts
                     Whether offline or online, plan your usage accordingly.
        `with_answer`: Determines if answer is included in natural language
                       prompt
        `aggregate`: Combine multiple prompts into a single content message
        `max_digits`: The maximum base-10 integers to generate in this
                      procedure are: |10 ^ (max_digits)|
        `strict_digits`: Guarantee maximum digits are used
        `include_negative`: Permits negative operand values
        `feature_expression`: Passed directly to uses of natural_language_math
        `require_in_vocab`: Prefer operands to be (in/out of) vocabulary single
                            tokens (unless None, wherein no preference)
        `first_extension`: Bonus text to prefix the very first prompt with
        `each_extension`: Bonus text to prefix every prompt with
        `limit`: Generate until a certain number of tokens would be exceeded,
                 regardless of n_prompts
    """
    prompt_queue = []
    prompt_answers = []
    total_prompt_len = 0
    longest_prompt_tokens = 0
    while limit is not None or (len(prompt_queue) < n_prompts):
        # Challenges for the LLM to solve are generated and converted into natural
        # language prompts based on desired settings
        prompt_operands = generate_operands(2,
                                            max_digits,
                                            strict_digits,
                                            include_negative,
                                            in_vocab=require_in_vocab)
        # Ground-truth answers to check LLM against
        answer = op(*prompt_operands)
        try:
            # Use .item() to convert numpy-dtypes to python, which are
            # JSON-serializable
            answer = answer.item()
        except AttributeError:
            pass
        if not with_answer:
            prompt_answers.append(answer)
        prompt = natural_language_math(prompt_operands,
                                       op,
                                       answer,
                                       with_answer=with_answer,
                                       feature_expression=feature_expression)
        if len(prompt_queue) == 0 and first_extension is not None:
            prompt = first_extension + prompt
        if each_extension is not None:
            prompt = each_extension + prompt
        prompt = {'role': 'user', 'content': prompt}
        new_prompt_len = tokenize(prompt, count=True)
        longest_prompt_tokens = max(new_prompt_len, longest_prompt_tokens)
        if limit is not None:
            if total_prompt_len + new_prompt_len >= limit:
                break
            total_prompt_len += new_prompt_len
            print(f"Prompt {len(prompt_queue)+1} adds {new_prompt_len} tokens, "
                  f"running total: {total_prompt_len}")
        else:
            total_prompt_len += new_prompt_len
        prompt_queue.append(prompt)
    if aggregate:
        # Flatten into a single message
        prompt_queue = [{'role': 'user', 'content': "\n".join([p['content'] \
                                                    for p in prompt_queue])}]

    return prompt_queue, prompt_answers, total_prompt_len, longest_prompt_tokens

def posteval_llm_trial(trial_history_ref: Dict,
                       known_values: Dict,
                       tokenized_trial: List[str],
                       answer: object,
                       response: Dict):
    """
        Attempt to automatically parse LLM response to determine certain
        attributes it may exhibit, such as following the intended format,
        producing a parseable answer, copying ICL data, or producing the right
        answer.

        `trial_history_ref`: Sub-dictionary for this series of trials to update
        `known_values`: Copycat dictionary to check if LLM may be copying ICL
        `tokenized_trial`: Representation of this trial in token-form
        `answer`: Actual answer the LLM should produce
        `reponse`: LLM's JSON-style response
    """
    # We check if the LLM memorized this answer
    llm_answer_tokenized = tuple(tokenize(response['message']))
    if llm_answer_tokenized in known_values:
        if tokenized_trial in known_values[llm_answer_tokenized]:
            print("This is a verbatim memorized answer")
            trial_history_ref['memorized_verbatim'][-1] = 1
        else:
            print("This is a previously-seen memorized answer (copycat), "
                  "but it is answering a different trial input")
            trial_history_ref['memorized_copycat'][-1] = 1
    # We check if the LLM followed our requested output format and
    # attempt to parse a comparable value to verify correctness

    # TODO: Permit these regexes to be defined as shallowregex (here) and
    # deepregex (second .findall, below) so operator extensions can match more
    # diverse types of output
    regex_portion = re.findall(r"## ?(-?\d+) ?##",
                               response['message']['content'])
    if regex_portion:
        trial_history_ref['follows_regex'][-1] = 1
        # No try/except here, following the regex SHOULD make this
        # parse, and if not I want the program to crash so I can see
        # the bug. We choose the last match in case the LLM repeated
        # some ICL examples.
        llm_answer = type(answer)(regex_portion[-1])
        print("LLM followed requested output format and answered:", llm_answer)
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
            llm_answer = type(answer)(final_match)
            print("LLM DID NOT follow requested output format, but we "
                  "recovered this value as its possible answer:", llm_answer)
        except:
            print("LLM DID NOT follow requested output format, and we were "
                  "UNABLE to recover any value as its possible answer")
            return trial_history_ref
    trial_history_ref['parseable'][-1] = 1
    print("LLM's answer is ", end='')
    if llm_answer == answer:
        print("correct")
        trial_history_ref['correct'][-1] = 1
    else:
        print("INCORRECT")
    return trial_history_ref

def llm_trial(mstate: List[Dict[str,str]],
              answer: object,
              trial_history_ref: Dict,
              known_values: Dict,
              tqdm_updater: Optional[tqdm.tqdm],
              save_file: str,
              ) -> Dict:
    """
        Repeat a trial across all OLLAMA_CONFIGS, logging results to disk.
        Return updated trial history

        `mstate`: Initial state for the trial (all prompt messages in history)
        `answer`: Correct answer the LLM should produce post-parsing
        `trial_history_ref`: Sub-dictionary for this series of trials to update
        `known_values`: Copycat dictionary to check if LLM may be copying ICL
        `tqdm_updater`: Progress bar handle to update with each generated response
        `save_file`: Path to save results to
    """
    trial = mstate[-1]
    tokenized_trial = None if len(known_values) == 0 else tuple(tokenize(trial))
    for (attempt,CONFIG) in enumerate(OLLAMA_CONFIGS):
        # The LLM produces its answer
        print(f"LLM Attempt {attempt+1}/{len(OLLAMA_CONFIGS)}")
        response = ollama.chat(model=LLM_MODEL,
                               messages=mstate,
                               options=CONFIG)
        trial_history_ref['time'].append(response['eval_duration'] / 1e9)
        trial_history_ref['response'].append(response['message']['content'])
        tlens = list(map(len, list(trial_history_ref.values())[1:]))
        max_tlen = max(tlens)
        for key, tlen in zip(list(trial_history_ref.keys())[1:], tlens):
            trial_history_ref[key].extend([0] * (max_tlen-tlen))
        print(f"TRUTH: {trial['content']}## {answer} ##")
        print(f"Response {attempt+1}/{len(OLLAMA_CONFIGS)}: "
              f"{trial['content']}{response['message']['content']}")
        trial_history_ref = posteval_llm_trial(trial_history_ref,
                                               known_values,
                                               tokenized_trial,
                                               answer,
                                               response)
        if tqdm_updater is not None:
            tqdm_updater.update(n=1)
        # Update results per LLM response, so we save everything that was
        # completely handled
        with open(save_file, 'w') as f:
            json.dump(trial_history_ref, f, indent=1)
    return trial_history_ref

def llm_examples(op: callable,
                 trial_history: Dict,
                 save_file: str,
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
                 substitution_operand: Optional[Tuple[str,str]] = None,
                 encourage_rewrites: bool = False,
                 icl_in_vocab: Optional[bool] = None,
                 eval_in_vocab: Optional[bool] = None,
                 ) -> Dict[str, Dict[str, int]]:
    """
        Main driver function of the test battery this script intends to provide.

        `op`: The callable function to utilize for these tests
        `trial_history`: History of prior evaluations to update with new ones
        `save_file`: Path to save updated trial_history to
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
        `with_context`: Generate ICL using natural_language_math, to be
                        presented to the LLM prior to each task as ground-truth
                        examples. This REPLACES the system prompt as
                        instructions, so ICL is all that the LLM will have for
                        context of its task to complete.
        `use_both`: Generate ICL using natural_language_math and ADD this as
                    additional instructions for the LLM to support proper task
                    fulfillment.
        `feature_expression`: Passed directly to uses of natural_language_math
        `substitution_operand`: When provided, an exchanged pair of strings to
                                explain to the LLM that notation generated by
                                natural_language_math may use unfamiliar
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
    #                    Create all prompt materials                          #
    ###########################################################################

    (system_messages,
     op_explanation,
     token_expenditure) = make_system_prompt(op,
                                             with_context,
                                             use_both,
                                             substitution_operand,
                                             encourage_rewrites,
                                             token_expenditure,
                                             count_tokens)

    (prompt_queue,
     prompt_answers,
     prompt_tokens,
     longest_prompt_tokens) = make_prompts(op,
                                           n_evals,
                                           False, # Evaluation prompts lack answer
                                           False, # Evaluation prompts are not aggregated
                                           max_digits,
                                           strict_digits,
                                           include_negative,
                                           feature_expression,
                                           eval_in_vocab,
                                           each_extension="CHALLENGE:\n")
    # Token expenditure is updated based on longest eval prompt
    token_expenditure += longest_prompt_tokens

    limit = None if token_limit is None else token_limit-token_expenditure
    (icl_queue,
     icl_answers_empty,
     prompt_tokens,
     longest_prompt_tokens) = make_prompts(op,
                                           n_examples,
                                           True, # ICL prompts have answer
                                           True, # ICL prompts are aggregated
                                           max_digits,
                                           strict_digits,
                                           include_negative,
                                           feature_expression,
                                           icl_in_vocab,
                                           first_extension="EXAMPLES:\n",
                                           limit=limit)
    # Token expenditure is updated based on collated prompts
    token_expenditure += prompt_tokens
    # This logic combination DOESN'T have the system message, so ensure that
    # the special instructions are passed along via the ICL mechanism
    if op_explanation != "" and with_context and not use_both:
        icl_queue = [{'role': 'system', 'content': op_explanation}] + icl_queue
        if count_tokens:
            op_explanation_tokens = tokenize(op_explanation, count=True)
            token_expenditure += op_explanation_tokens
            print(f"Operation requires explanation, adding {op_explanation_tokens} "
                  f"tokens, running total: {token_expenditure}")

    ###########################################################################
    #                  Template for evaluations and results                   #
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
    trial_history['prompt_template'] = "".join([_['content'] for _ in initial_mstate])
    print(trial_history['prompt_template'])
    if count_tokens:
        print(f"Counted tokens: {token_expenditure} / {token_limit}")

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
    #                          Perform evaluations                            #
    ###########################################################################

    for (trial, answer) in zip(prompt_queue, prompt_answers):
        trialkey = trial['content']
        trial_history[trialkey] = {'answer': answer,
                                   'time': list(),
                                   'response': list()}
        trial_history[trialkey].update({k: list() for k in result_metrics})
        # Update results (ALL trials, log that this one started so we know how
        # to merge any temporary results)
        with open(save_file, 'w') as f:
            json.dump(trial_history, f, indent=1)
        trial_history[trialkey] = llm_trial(initial_mstate + [trial],
                                            answer,
                                            trial_history[trialkey],
                                            known_values,
                                            tqdm_updater,
                                            save_file+'.tmp')
        # Update results (ALL trials)
        with open(save_file, 'w') as f:
            json.dump(trial_history, f, indent=1)
    return trial_history

if __name__ == '__main__':
    ###########################################################################
    #                       Command Line Interface                            #
    ###########################################################################

    prs = argparse.ArgumentParser()
    prs.add_argument('file',
                     help="Path to log results to (in JSON format)")
    prs.add_argument('--override', action='store_true',
                     help="Expect to override the file, preventing safety errors")
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
    llmconf.add_argument("--tokenizer", choices=lm_tokenizers.IMPLEMENTERS,
                     default=lm_tokenizers.IMPLEMENTERS[0],
                     help="Tokenizer used for parsing (default: %(default)s)")
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

    ###########################################################################
    #                       Command Line Parsing                              #
    ###########################################################################

    if pathlib.Path(args.file).exists() and not args.override:
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
        args.token_limit = None
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

    # Load the tokenizer and number vocabulary
    tokenizer = getattr(lm_tokenizers, args.tokenizer+'_tk')
    number_vocab_int = getattr(lm_tokenizers, args.tokenizer+'_number_vocab_int')
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
    results = {'configs': OLLAMA_CONFIGS}
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
                if name in op_extensions.names:
                    print("Skipping extended operator '{name}' on system "
                          "message only -- nothing of value to learn.")
                    overall_progress.update(n=1)
                    continue
                bonus_kwargs = {'with_context': False}
            elif PROMPT_SETTING == 'icl+system':
                bonus_kwargs = {'use_both': True}
            elif PROMPT_SETTING == 'none':
                bonus_kwargs = {'no_prompt': True}
            else:
                raise NotImplemented(PROMPT_SETTING)
            results = llm_examples(op,
                                   results,
                                   args.file,
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
    # Clean up the .tmp file on successful exit
    tmp_file= pathlib.Path(args.file+'.tmp')
    if tmp_file.exists():
        tmp_file.unlink()

