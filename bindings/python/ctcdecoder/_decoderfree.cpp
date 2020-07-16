/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "/home/local/ZOHOCORP/vishnu-pt3475/decoder/src/decoder/LexiconFreeDecoder.h"

namespace py = pybind11;
using namespace w2l;
using namespace py::literals;

/**
 * Some hackery that lets pybind11 handle shared_ptr<void> (for old LMStatePtr).
 * See: https://github.com/pybind/pybind11/issues/820
 * PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>);
 * and inside PYBIND11_MODULE
 *   py::class_<std::shared_ptr<void>>(m, "encapsulated_data");
 */

namespace {

/**
 * A pybind11 "alias type" for abstract class LM, allowing one to subclass LM
 * with a custom LM defined purely in Python. For those who don't want to build
 * with KenLM, or have their own custom LM implementation.
 * See: https://pybind11.readthedocs.io/en/stable/advanced/classes.html
 *
 * TODO: ensure this works. Last time Jeff tried this there were slicing issues,
 * see https://github.com/pybind/pybind11/issues/1546 for workarounds.
 * This is low-pri since we assume most people can just build with KenLM.
 */

void LexiconFreeDecoder_decodeStep(
    LexiconFreeDecoder& decoder,
    const std::vector<float>& emissions,
    int T,
    int N) {
  decoder.decodeStep(emissions, T, N);
}

std::vector<DecodeResult> LexiconFreeDecoder_decode(
    LexiconFreeDecoder& decoder,
    const std::vector<float>& emissions,
    int T,
    int N) {
  return decoder.decode(emissions, T, N);
}

} // namespace

PYBIND11_MODULE(_decoderfree, m) {

  py::enum_<CriterionType>(m, "CriterionType")
      .value("CTC", CriterionType::CTC);

  py::class_<DecoderOptions>(m, "DecoderOptions")
      .def(
          py::init<
              const int,
              const int,
              const double,
              const double,
              const double,
              const double,
              const double,
              const double,
              const bool,
              const CriterionType>(),
          "beam_size"_a,
          "beam_size_token"_a,
          "beam_threshold"_a,
          "lm_weight"_a,
          "word_score"_a,
          "unk_score"_a,
          "sil_score"_a,
          "eos_score"_a,
          "log_add"_a,
          "criterion_type"_a)
      .def_readwrite("beam_size", &DecoderOptions::beamSize)
      .def_readwrite("beam_size_token", &DecoderOptions::beamSizeToken)
      .def_readwrite("beam_threshold", &DecoderOptions::beamThreshold)
      .def_readwrite("lm_weight", &DecoderOptions::lmWeight)
      .def_readwrite("word_score", &DecoderOptions::wordScore)
      .def_readwrite("unk_score", &DecoderOptions::unkScore)
      .def_readwrite("sil_score", &DecoderOptions::silScore)
      .def_readwrite("eos_score", &DecoderOptions::silScore)
      .def_readwrite("log_add", &DecoderOptions::logAdd)
      .def_readwrite("criterion_type", &DecoderOptions::criterionType);

  py::class_<DecodeResult>(m, "DecodeResult")
      .def(py::init<int>(), "length"_a)
      .def_readwrite("score", &DecodeResult::score)
      .def_readwrite("words", &DecodeResult::words)
      .def_readwrite("tokens", &DecodeResult::tokens);

  // NB: `decode` and `decodeStep` expect raw emissions pointers.
  py::class_<LexiconFreeDecoder>(m, "LexiconFreeDecoder")
      .def(py::init<
           const DecoderOptions&,
           const int,
           const int>())
      .def("decode_begin", &LexiconFreeDecoder::decodeBegin)
      .def(
          "decode_step",
          &LexiconFreeDecoder::decodeStep,
          "emissions"_a,
          "T"_a,
          "N"_a)
      .def("decode_end", &LexiconFreeDecoder::decodeEnd)
      .def("decode", &LexiconFreeDecoder_decode, "emissions"_a, "T"_a, "N"_a)
      .def("prune", &LexiconFreeDecoder::prune, "look_back"_a = 0)
      .def(
          "get_best_hypothesis",
          &LexiconFreeDecoder::getBestHypothesis,
          "look_back"_a = 0)
      .def("get_all_final_hypothesis", &LexiconFreeDecoder::getAllFinalHypothesis);
}
