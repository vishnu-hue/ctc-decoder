/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <unordered_map>

#include "Decoder.h"
using namespace std;
namespace w2l {
/**
 * LexiconFreeDecoderState stores information for each hypothesis in the beam.
 */
struct LexiconFreeDecoderState {
  double score; // Accumulated total score so far
  double nb_score;//non blank score
  double b_score;//blank score
  const LexiconFreeDecoderState* parent; // Parent hypothesis
  int token; // Label of token
  //bool prevBlank; // If previous hypothesis is blank (for CTC only)
  std::string label;//total label

  LexiconFreeDecoderState(
      const double score,
      const double nb_score,
      const double b_score,
      string label,
      const LexiconFreeDecoderState* parent,
      const int token)
      : score(score),
        nb_score(nb_score),
        b_score(b_score),
        label(label),
        parent(parent),
        token(token){}

  LexiconFreeDecoderState()
      : score(0),
        nb_score(0),
        b_score(0),
        label(to_string(' ')),
        parent(nullptr),
        token(-1){}

  int compareNoScoreStates(const LexiconFreeDecoderState* node) const {
    if(label!=node->label)
    {
      return label>node->label? 1:-1;
    }
    return 0;
  }

  int getWord() const {
    return -1;
  }

  bool isComplete() const {
    return true;
  }
};

/**
 * Decoder implements a beam seach decoder that finds the word transcription
 * W maximizing:
 *
 * AM(W) + lmWeight_ * log(P_{lm}(W)) + silScore_ * |{i| pi_i = <sil>}|
 *
 * where P_{lm}(W) is the language model score, pi_i is the value for the i-th
 * frame in the path leading to W and AM(W) is the (unnormalized) acoustic model
 * score of the transcription W. We are allowed to generate words from all the
 * possible combination of tokens.
 */
class LexiconFreeDecoder : public Decoder {
 public:
  LexiconFreeDecoder(
      const DecoderOptions& opt,
      const int sil,
      const int blank)
      : Decoder(opt),
        sil_(sil),
        blank_(blank) {}

  void decodeBegin() override;

  void decodeStep(const std::vector<float>& emissions, int T, int N) override;

  void decodeEnd() override;

  int nHypothesis() const;

  void prune(int lookBack = 0) override;

  int nDecodedFramesInBuffer() const override;

  DecodeResult getBestHypothesis(int lookBack = 0) const override;

  std::vector<DecodeResult> getAllFinalHypothesis() const override;

 protected:
  // All the hypothesis new candidates (can be larger than beamsize) proposed
  // based on the ones from previous frame
  std::vector<LexiconFreeDecoderState> candidates_;

  // This vector is designed for efficient sorting and merging the candidates_,
  // so instead of moving around objects, we only need to sort pointers
  std::vector<LexiconFreeDecoderState*> candidatePtrs_;

  // Best candidate score of current frame
  double candidatesBestScore_;

  // Index of silence label
  int sil_;

  // Index of blank label (for CTC)
  int blank_;

  // Vector of hypothesis for all the frames so far
  std::unordered_map<int, std::vector<LexiconFreeDecoderState>> hyp_;

  // These 2 variables are used for online decoding, for hypothesis pruning
  int nDecodedFrames_; // Total number of decoded frames.
  int nPrunedFrames_; // Total number of pruned frames from hyp_.
};

} // namespace w2l
