/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "src/decoder/LexiconDecoder.h"

#ifdef W2L_LIBRARIES_USE_KENLM
#include "src/lm/KenLM.h"
#endif

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
class PyLM : public LM {
  using LM::LM;

  // needed for pybind11 or else it won't compile
  using LMOutput = std::pair<LMStatePtr, float>;

  LMStatePtr start(bool startWithNothing) override {
    PYBIND11_OVERLOAD_PURE(LMStatePtr, LM, start, startWithNothing);
  }

  LMOutput score(const LMStatePtr& state, const int usrTokenIdx) override {
    PYBIND11_OVERLOAD_PURE(LMOutput, LM, score, state, usrTokenIdx);
  }

  LMOutput finish(const LMStatePtr& state) override {
    PYBIND11_OVERLOAD_PURE(LMOutput, LM, finish, state);
  }
};

/**
 * Using custom python LMState derived from LMState is not working with
 * custom python LM (derived from PyLM) because we need to to custing of LMState
 * in score and finish functions to the derived class
 * (for example vie obj.__class__ = CustomPyLMSTate) which cause the error
 * "TypeError: __class__ assignment: 'CustomPyLMState' deallocator differs
 * from 'wav2letter._decoder.LMState'"
 * details see in https://github.com/pybind/pybind11/issues/1640
 * To define custom LM you can introduce map inside LM which maps LMstate to
 * additional state info (shared pointers pointing to the same underlying object
 * will have the same id in python in functions score and finish)
 */
void LexiconDecoder_decodeStep(
    LexiconDecoder& decoder,
    std::vector<float>& emissions,
    int T,
    int N) {
  decoder.decodeStep(emissions, T, N);
}

std::vector<DecodeResult> LexiconDecoder_decode(
    LexiconDecoder& decoder,
    std::vector<float>& emissions,
    int T,
    int N) {
  return decoder.decode(emissions, T, N);
}

} // namespace

PYBIND11_MODULE(_decoder, m) {
  py::enum_<SmearingMode>(m, "SmearingMode")
      .value("NONE", SmearingMode::NONE)
      .value("MAX", SmearingMode::MAX)
      .value("LOGADD", SmearingMode::LOGADD);

  py::class_<TrieNode, TrieNodePtr>(m, "TrieNode")
      .def(py::init<int>(), "idx"_a)
      .def_readwrite("children", &TrieNode::children)
      .def_readwrite("idx", &TrieNode::idx)
      .def_readwrite("labels", &TrieNode::labels)
      .def_readwrite("scores", &TrieNode::scores)
      .def_readwrite("max_score", &TrieNode::maxScore);

  py::class_<Trie, TriePtr>(m, "Trie")
      .def(py::init<int, int>(), "max_children"_a, "root_idx"_a)
      .def("get_root", &Trie::getRoot)
      .def("insert", &Trie::insert, "indices"_a, "label"_a, "score"_a)
      .def("search", &Trie::search, "indices"_a)
      .def("smear", &Trie::smear, "smear_mode"_a);

  py::class_<LM, LMPtr, PyLM>(m, "LM")
      .def(py::init<>())
      .def("start", &LM::start, "start_with_nothing"_a)
      .def("score", &LM::score, "state"_a, "usr_token_idx"_a)
      .def("finish", &LM::finish, "state"_a);

  py::class_<LMState, LMStatePtr>(m, "LMState")
      .def(py::init<>())
      .def_readwrite("children", &LMState::children)
      .def("compare", &LMState::compare, "state"_a)
      .def("child", &LMState::child<LMState>, "usr_index"_a);

#ifdef W2L_LIBRARIES_USE_KENLM
  py::class_<KenLM, KenLMPtr, LM>(m, "KenLM")
      .def(
          py::init<const std::string&, const Dictionary&>(),
          "path"_a,
          "usr_token_dict"_a);
  py::class_<create_word_file>(m,"create_word_file")
      .def(py::init<const std::string&, const std::string&>(),
        "path"_a,
        "letters"_a);
#endif

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
  py::class_<LexiconDecoder>(m, "LexiconDecoder")
      .def(py::init<
           const DecoderOptions&,
           const TriePtr,
           const LMPtr,
           const int,
           const int,
           const bool>())
      .def("decode_begin", &LexiconDecoder::decodeBegin)
      .def(
          "decode_step",
          &LexiconDecoder::decodeStep,
          "emissions"_a,
          "T"_a,
          "N"_a)
      .def("decode_end", &LexiconDecoder::decodeEnd)
      .def("decode", &LexiconDecoder_decode, "emissions"_a, "T"_a, "N"_a)
      .def("prune", &LexiconDecoder::prune, "look_back"_a = 0)
      .def(
          "get_best_hypothesis",
          &LexiconDecoder::getBestHypothesis,
          "look_back"_a = 0)
      .def("get_all_final_hypothesis", &LexiconDecoder::getAllFinalHypothesis);
}
