/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdlib.h>
#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric> 
#include<iostream>
#include<string>
#include "LexiconDecoder.h"
using namespace std;
namespace w2l {

void LexiconDecoder::decodeBegin() {
  hyp_.clear();
  hyp_.emplace(0, std::vector<LexiconDecoderState>());
  std::string letter=to_string(-1);
  /* note: the lm reset itself with :start() */
  hyp_[0].emplace_back(0.0,0.0,0.0,letter,lm_->start(1),lexicon_->getRoot(),nullptr, -1,-1);
  nDecodedFrames_ = 0;
  nPrunedFrames_ = 0;
}

void LexiconDecoder::decodeStep(const std::vector<float>& emissions, int T, int N) {
  int startFrame = nDecodedFrames_ - nPrunedFrames_;
  // Extend hyp_ buffer

  if (hyp_.size() < startFrame + T + 2) {
    for (int i = hyp_.size(); i < startFrame + T + 2; i++) {
      hyp_.emplace(i, std::vector<LexiconDecoderState>());
    }
  }

  std::vector<size_t> idx(N);
  int lenght;
  // Looping over all the frames
  for (int t = 0; t < T; t++) {
    std::iota(idx.begin(), idx.end(), 0);
    if (N > opt_.beamSizeToken) {
      lenght=opt_.beamSizeToken;
    }
    else
    {
      lenght=N;
    }
    
      std::nth_element(
          idx.begin(),
          idx.begin() + lenght,
          idx.end(),
          [&t, &N, &emissions](const size_t& l, const size_t& r) {
            return emissions[t * N + l] > emissions[t * N + r];
          });
    candidatesReset(candidatesBestScore_, candidates_, candidatePtrs_);
    for (const LexiconDecoderState& prevHyp : hyp_[startFrame + t]) {
      const int prevIdx = prevHyp.token;
      const TrieNode* prevLex = prevHyp.lex;
      const float lexMaxScore =
          prevLex == lexicon_->getRoot() ? 0 : prevLex->maxScore;
      for (int r = 0; r < std::min(opt_.beamSizeToken, N); ++r) {
        int n = idx[r];
        auto iter = prevLex->children.find(n);
        if (iter == prevLex->children.end()) {
          continue;
        }
        const TrieNodePtr& lex = iter->second;
        double amScore = emissions[t * N + n];
        LMStatePtr lmState;
        double lmScore = 0.;

        if (isLmToken_) {
          auto lmStateScorePair = lm_->score(prevHyp.lmState, n);
          lmState = lmStateScorePair.first;
          lmScore = lmStateScorePair.second;
        }
        if(n!=blank_ && n!=prevHyp.token)  
        {
          if(!lex->children.empty())
          {
            candidatesAdd(
              candidates_,
              candidatesBestScore_,
              opt_.beamThreshold,
              n,
              true,
              0,
              prevHyp.score+amScore,
              0,
              prevHyp.label,
              prevHyp.lmState,
              lex.get(),
              &prevHyp,
              n,
              -1,
              prevHyp.lmScore+lmScore);}
        }
        else if(n!=blank_ && n==prevHyp.token) 
        {
          
          if(prevHyp.b_score!=0)
          {candidatesAdd(
              candidates_,
              candidatesBestScore_,
              opt_.beamThreshold,
              n,
              true,
              0,
              prevHyp.b_score+amScore,
              0,
              prevHyp.label,
              prevHyp.lmState,
              lex.get(),
              &prevHyp,
              n,
              -1,
              prevHyp.lmScore+lmScore);
          }
        }
          for (auto label : lex->labels){
           if (!isLmToken_) {
            auto lmStateScorePair = lm_->score(prevHyp.lmState, label);
            lmState = lmStateScorePair.first;
            lmScore = lmStateScorePair.second;
          }
           candidatesAdd(
              candidates_,
              candidatesBestScore_,
              opt_.beamThreshold,
              n,
              true,
              0,
              prevHyp.score+amScore+opt_.wordScore+ opt_.lmWeight * lmScore,
              0,
              prevHyp.label,
              lmState,
              lexicon_->getRoot(),
              &prevHyp,
              n,
              label,
              prevHyp.lmScore+lmScore);
        }
      } 
      if(prevIdx!=-1) 
      {candidatesAdd(
              candidates_,
              candidatesBestScore_,
              opt_.beamThreshold,
              prevIdx,
              false,
              0,
              prevHyp.nb_score+emissions[t*N+prevIdx],
              prevHyp.score+emissions[t*N+blank_],
              prevHyp.label,
              prevHyp.lmState,
              prevLex,
              &prevHyp,
              prevIdx,
              -1,
              prevHyp.lmScore);}
    }
    candidatesStore(
        candidates_,
        candidatePtrs_,
        hyp_[startFrame + t + 1],
        opt_.beamSize,
        candidatesBestScore_ - opt_.beamThreshold,
        opt_.logAdd,
        true);
        updateLMCache(lm_, hyp_[startFrame + t + 1]);
  }
  nDecodedFrames_ += T;
}

void LexiconDecoder::decodeEnd() {

  
}

std::vector<DecodeResult> LexiconDecoder::getAllFinalHypothesis() const {
  int finalFrame = nDecodedFrames_ - nPrunedFrames_;
  return getAllHypothesis(hyp_.find(finalFrame)->second, finalFrame);
}

DecodeResult LexiconDecoder::getBestHypothesis(int lookBack) const {
  int finalFrame = nDecodedFrames_ - nPrunedFrames_;
  const LexiconDecoderState* bestNode =
      findBestAncestor(hyp_.find(finalFrame)->second, lookBack);

  return getHypothesis(bestNode, nDecodedFrames_ - nPrunedFrames_ - lookBack);
}

int LexiconDecoder::nHypothesis() const {
  int finalFrame = nDecodedFrames_ - nPrunedFrames_;
  return hyp_.find(finalFrame)->second.size();
}

int LexiconDecoder::nDecodedFramesInBuffer() const {
  return nDecodedFrames_ - nPrunedFrames_ + 1;
}

void LexiconDecoder::prune(int lookBack) {
  if (nDecodedFrames_ - nPrunedFrames_ - lookBack < 1) {
    return; // Not enough decoded frames to prune
  }

  /* (1) Find the last emitted word in the best path */
  int finalFrame = nDecodedFrames_ - nPrunedFrames_;
  const LexiconDecoderState* bestNode =
      findBestAncestor(hyp_.find(finalFrame)->second, lookBack);
  if (!bestNode) {
    return; // Not enough decoded frames to prune
  }

  int startFrame = nDecodedFrames_ - nPrunedFrames_ - lookBack;
  if (startFrame < 1) {
    return; // Not enough decoded frames to prune
  }

  /* (2) Move things from back of hyp_ to front and normalize scores */
  pruneAndNormalize(hyp_, startFrame, lookBack);

  nPrunedFrames_ = nDecodedFrames_ - lookBack;
}

} // namespace w2l
