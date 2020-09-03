/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <unordered_map>
#include "Trie.h"
#include "Decoder.h"
#include "src/lm/LM.h" 
using namespace std;
namespace w2l {
/**
 * LexiconFreeDecoderState stores information for each hypothesis in the beam.
 */
struct LexiconDecoderState {
  double score; // Accumulated total score so far
  double nb_score;
  double b_score;
  LMStatePtr lmState; // Language model state
  const TrieNode* lex;
  const LexiconDecoderState* parent; // Parent hypothesis
  int token; // Label of token
  int word;
  double lmScore; // Accumulated LM score so far
  //bool prevBlank; // If previous hypothesis is blank (for CTC only)
  std::string label;

  LexiconDecoderState(
      const double score,
      const double nb_score,
      const double b_score,
      string label,
      const LMStatePtr& lmState,
      const TrieNode* lex,
      const LexiconDecoderState* parent,
      const int token,
      int word,
      const double lmScore = 0)
      : score(score),
        nb_score(nb_score),
        b_score(b_score),
        label(label),
        lmState(lmState),
        lex(lex),
        parent(parent),
        token(token),
        word(word),
        lmScore(lmScore){}

  LexiconDecoderState()
      : score(0),
        nb_score(0),
        b_score(0),
        label(to_string(' ')),
        lmState(nullptr),
        lex(nullptr),
        parent(nullptr),
        token(-1),
        word(-1),
        lmScore(0.){}

  int compareNoScoreStates(const LexiconDecoderState* node) const {
    if(label!=node->label) 
    {
      return label>node->label? 1:-1;
    }
    //int lmCmp = lmState->compare(node->lmState);
    //if (lmCmp != 0) {
      //return lmCmp > 0 ? 1 : -1;
    //}
    else if(lex!=node->lex)
    {
      return lex>node->lex ? 1: -1;
    }
    return 0;
  }

  int getWord() const {
    return word;
  }

  bool isComplete() const {
    return !parent || parent->word >= 0;
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
class LexiconDecoder : public Decoder {
 public:
  LexiconDecoder(
      const DecoderOptions& opt,
      const TriePtr& lexicon,
      const LMPtr& lm,
      const int sil,
      const int blank,
      const bool isLmToken)
      : Decoder(opt),
        lexicon_(lexicon),
        lm_(lm),
        sil_(sil),
        blank_(blank),
        isLmToken_(isLmToken){}

  void decodeBegin() override;

  void decodeStep(const std::vector<float>& emissions, int T, int N) override;

  void decodeEnd() override;

  int nHypothesis() const;

  void prune(int lookBack = 0) override;

  int nDecodedFramesInBuffer() const override;

  DecodeResult getBestHypothesis(int lookBack = 0) const override;

  std::vector<DecodeResult> getAllFinalHypothesis() const override;

 protected:
  TriePtr lexicon_;
  LMPtr lm_;
  // Index of silence label
  int sil_;

  // Index of blank label (for CTC)
  int blank_;

  bool isLmToken_;

  // All the hypothesis new candidates (can be larger than beamsize) proposed
  // based on the ones from previous frame
  std::vector<LexiconDecoderState> candidates_;

  // This vector is designed for efficient sorting and merging the candidates_,
  // so instead of moving around objects, we only need to sort pointers
  std::vector<LexiconDecoderState*> candidatePtrs_;

  // Best candidate score of current frame
  double candidatesBestScore_;

  

  // Vector of hypothesis for all the frames so far
  std::unordered_map<int, std::vector<LexiconDecoderState>> hyp_;

  // These 2 variables are used for online decoding, for hypothesis pruning
  int nDecodedFrames_; // Total number of decoded frames.
  int nPrunedFrames_; // Total number of pruned frames from hyp_.
};

} // namespace w2l
