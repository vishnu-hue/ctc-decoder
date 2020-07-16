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

def loadRNNOutput(fn):
	"load RNN output from csv file. Last entry in row terminated by semicolon."
	return np.genfromtxt(fn, delimiter=';')[:, : -1]

def softmax(mat):
	"calc softmax such that labels per time-step form probability distribution"
	maxT, _ = mat.shape # dim0=t, dim1=c
	res = np.zeros(mat.shape)
	for t in range(maxT):
		y = mat[t, :]
		e = np.exp(y)
		s = np.sum(e)
		res[t, :] = e/s
	return res

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} decoder_test_data_path", file=sys.stderr)
        print("  (usually: <wav2letter_root>/src/decoder/test)", file=sys.stderr)
        sys.exit(1)

    data_path = sys.argv[1]
    mat = softmax(loadRNNOutput('/home/local/ZOHOCORP/vishnu-pt3475/Downloads/CTCDecoder/data/line/rnnOutput.csv'))
    T,N=mat.shape
    mat=np.log(mat)
    mat.resize(T*N)
    transitions = load_transitions(os.path.join(data_path, "transition.bin"))
    token_dict =' !"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    blank_idx=len(token_dict)
    sil_idx = 0
    # Define decoder options:
    # DecoderOptions (beam_size, token_beam_size, beam_threshold,
    #                 word_score, unk_score, sil_score,
    #                 eos_score, log_add, criterion_type (CTC))
    opts = DecoderOptions(
        100, 40, 100.0, 2.0, 2.0, -math.inf, -1, 0, False, CriterionType.CTC
    )

    # define lexicon beam-search decoder with word-level lm
    # LexiconFreeDecoder(decoder options, silence index,
    #                blank index (for CTC),
    #                transitiona matrix)
    decoder = LexiconFreeDecoder(opts,sil_idx, blank_idx)
    # run decoding
    # decoder.decode(emissions, Time, Ntokens)
    # result is a list of sorted hypothesis, 0-index is the best hypothesis
    # each hypothesis is a struct with "score" and "words" representation
    # in the hypothesis and the "tokens" representation
    results = decoder.decode(mat, T, N)
    print("expected output:the fak friend of the fomcly hae tC")
    print(f"Decoding complete, obtained {len(results)} results")
    print("Showing top 5 results:")
    for i in range(min(5, len(results))):
        prediction = []
        for idx in results[i].tokens:
            if idx!=-1:
               prediction.append(token_dict[idx])
        prediction = " ".join(prediction)
        print(f"score={results[i].score} prediction='{prediction}'")

