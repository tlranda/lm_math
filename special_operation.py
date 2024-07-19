import numpy as np

def bitshift_list(i_value, max_bits=None):
    if max_bits is None:
        max_bits = len(OLLAMA_CONFIGS)
    n_values = 0
    blist = []
    while n_values < max_bits:
        blist.append((i_value>>n_values) % 2)
        n_values += 1
    return blist

def special_operation(left,right):
    """
        A difficult function that an LLM will probably not have seen in its training data.
        The expectation is that the LLM will FAIL to accurately model the function, but how it fails may be interesting

        # BOTH OPERANDS
            The 'nbits' value represents the minimal number of bits used to express the integer's absolute value,
            with 1 bit explicitly allocated to represent the zero-value.
        # LEFT OPERAND
            10 + nbits + the sum of nonzero bits in the integer representation of the left value
        # RIGHT OPERAND
            The sum of cosines of powers of two raised to the bit position if the bit is nonzero, else cosine of 0 (1.0), with an additional
            cosine of the number of bits in the right value
        # COMBINATION
            Combination is the simple product of scalar values, cast to an integer
        Right operand is converted into the sum of cosines of the powers of two that form the number's binary representation

        Then the return is the product of the converted operands cast down to an integer value
    """
    nbits = np.ceil(np.log2(np.abs(left)) if left != 0 else 1).astype(int)
    lops = 10 + nbits + sum(bitshift_list(left, max_bits=nbits))
    nbits = np.ceil(np.log2(np.abs(right)) if right != 0 else 1).astype(int)
    rops = sum([np.cos(2**idx if x > 0 else 0) for idx,x in enumerate(bitshift_list(right, max_bits=nbits)+[nbits])])
    return int(lops*rops)

