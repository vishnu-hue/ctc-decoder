#!/usr/bin/env python3
# Perform beam-search decoding with word-level LM
# this is test with dumped acoustic model scores
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import os
import struct
import sys
import torch

import numpy as np
from ctcdecoder.decoderfree import (
    CriterionType,
    DecoderOptions,
    LexiconFreeDecoder
)


# def ptr_as_bytes(x):
#     return struct.pack("P", x)
#
#
# def get_numpy_ptr_as_bytes(arr):
#     if not arr.flags["C_CONTIGUOUS"]:
#         raise ValueError("numpy array is not contiguous")
#     return ptr_as_bytes(arr.ctypes.data)


def read_struct(file, fmt):
    return struct.unpack(fmt, file.read(struct.calcsize(fmt)))


def load_tn(path):
    """
    Load time size and number of tokens from the dump
    (defines the score to move from token_i to token_j)

    Returns:
    --------
    int, int
    """
    with open(path, "rb") as file:
        T = read_struct(file, "i")[0]
        N = read_struct(file, "i")[0]
        return T, N


def load_emissions(path):
    """
    Load precomputed transition matrix
    (defines the score to move from token_i to token_j)

    Returns:
    --------
    numpy.array of shape [Batch=1, Time, Ntokens]
    """
    with open(path, "rb") as file:
        return np.frombuffer(file.read(T * N * 4), dtype=np.float32)


def load_transitions(path):
    """
    Load precomputed transition matrix
    (defines the score to move from token_i to token_j)

    Returns:
    --------
    numpy.array of shape [Ntokens, Ntokens]
    """
    with open(path, "rb") as file:
        return np.frombuffer(file.read(N * N * 4), dtype=np.float32)


def assert_near(x, y, tol):
    assert abs(x - y) <= tol


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} decoder_test_data_path", file=sys.stderr)
        print("  (usually: <wav2letter_root>/src/decoder/test)", file=sys.stderr)
        sys.exit(1)

    data_path = sys.argv[1]
    T=6
    N=7
    emissions =[0.06390443, 0.21124858, 0.27323887, 0.06870235, 0.0361254,0.18184413, 0.16493624,0.03309247, 0.22866108, 0.24390638, 0.09699597, 0.31895462,0.0094893, 0.06890021,0.218104, 0.19992557, 0.18245131, 0.08503348, 0.14903535,0.08424043, 0.08120984,0.12094152, 0.19162472, 0.01473646, 0.28045061, 0.24246305,0.05206269, 0.09772094,0.1333387, 0.00550838, 0.00301669, 0.21745861, 0.20803985,0.41317442, 0.01946335,0.16468227, 0.1980699, 0.1906545, 0.18963251, 0.19860937,0.04377724, 0.01457421]
    emissions=np.array(emissions)
    emissions=np.log(emissions)
    transitions = load_transitions(os.path.join(data_path, "transition.bin"))
    token_dict = ['\'', ' ', 'a', 'b', 'c', 'd', '_']
    blank_idx=token_dict.index('_')
    sil_idx = 1
    # Define decoder options:
    # DecoderOptions (beam_size, token_beam_size, beam_threshold,
    #                 word_score, unk_score, sil_score,
    #                 eos_score, log_add, criterion_type (CTC))
    opts = DecoderOptions(
        20, 40, 100.0, 2.0,2.0, -math.inf, -1, 0, False, CriterionType.CTC
    )

    # define lexicon beam-search decoder with word-level lm
    # LexiconDecoder(decoder options, silence index,
    #                blank index (for CTC),
    #                transitiona matrix)
    decoder = LexiconFreeDecoder(opts,sil_idx, blank_idx)
    # run decoding
    # decoder.decode(emissions, Time, Ntokens)
    # result is a list of sorted hypothesis, 0-index is the best hypothesis
    # each hypothesis is a struct with "score" and "words" representation
    # in the hypothesis and the "tokens" representation
    results = decoder.decode(emissions, T, N)
    print("expected output:acdc")
    print(f"Decoding complete, obtained {len(results)} results")
    print("Showing top 5 results:")
    for i in range(min(5, len(results))):
        predictions = []
        prediction = str()
        result=results[i].tokens.split('-1')
        for idx in result:
            if idx!='':
                index=int(idx)
                prediction+=token_dict[index]
        print(f"score={results[i].score} prediction='{prediction}'")   
