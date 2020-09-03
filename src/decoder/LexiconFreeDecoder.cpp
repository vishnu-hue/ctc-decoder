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
#include "LexiconFreeDecoder.h"
using namespace std;
namespace w2l {

void LexiconFreeDecoder::decodeBegin() {
  hyp_.clear();
  hyp_.emplace(0, std::vector<LexiconFreeDecoderState>());
  std::string letter=to_string(-1);
  /* note: the lm reset itself with :start() */
  hyp_[0].emplace_back(0.0,0.0,0.0,letter,nullptr, -1);
  nDecodedFrames_ = 0;
  nPrunedFrames_ = 0;
}

void LexiconFreeDecoder::decodeStep(const std::vector<float>& emissions, int T, int N) {
  int startFrame = nDecodedFrames_ - nPrunedFrames_;
  // Extend hyp_ buffer
  if (hyp_.size() < startFrame + T + 2) {
    for (int i = hyp_.size(); i < startFrame + T + 2; i++) {
      hyp_.emplace(i, std::vector<LexiconFreeDecoderState>());
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
    for (const LexiconFreeDecoderState& prevHyp : hyp_[startFrame + t]) {
      const int prevIdx = prevHyp.token;
      for (int r = 0; r < std::min(opt_.beamSizeToken, N); ++r) {
        int n = idx[r];
        double amScore = emissions[t * N + n];
        
        if(n!=blank_ && n!=prevHyp.token)  
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
              &prevHyp,
              n);
        }else if(n!=blank_ && n==prevHyp.token) 
        {
          candidatesAdd(
              candidates_,
              candidatesBestScore_,
              opt_.beamThreshold,
              n,
              false,
              0,
              prevHyp.nb_score+amScore,
              prevHyp.score+emissions[t*N+blank_],
              prevHyp.label,
              &prevHyp,
              n);
          if(prevHyp.b_score!=0 && n==prevHyp.token)
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
              &prevHyp,
              n);
          }
        }   
      }
      
    }
    candidatesStore(
        candidates_,
        candidatePtrs_,
        hyp_[startFrame + t + 1],
        opt_.beamSize,
        candidatesBestScore_ - opt_.beamThreshold,
        opt_.logAdd,
        true);
  }
  nDecodedFrames_ += T;
}

void LexiconFreeDecoder::decodeEnd() {
  
}

std::vector<DecodeResult> LexiconFreeDecoder::getAllFinalHypothesis() const {
  int finalFrame = nDecodedFrames_ - nPrunedFrames_;
  return getAllHypothesis(hyp_.find(finalFrame)->second, finalFrame);
}

DecodeResult LexiconFreeDecoder::getBestHypothesis(int lookBack) const {
  int finalFrame = nDecodedFrames_ - nPrunedFrames_;
  const LexiconFreeDecoderState* bestNode =
      findBestAncestor(hyp_.find(finalFrame)->second, lookBack);

  return getHypothesis(bestNode, nDecodedFrames_ - nPrunedFrames_ - lookBack);
}

int LexiconFreeDecoder::nHypothesis() const {
  int finalFrame = nDecodedFrames_ - nPrunedFrames_;
  return hyp_.find(finalFrame)->second.size();
}

int LexiconFreeDecoder::nDecodedFramesInBuffer() const {
  return nDecodedFrames_ - nPrunedFrames_ + 1;
}

void LexiconFreeDecoder::prune(int lookBack) {
  if (nDecodedFrames_ - nPrunedFrames_ - lookBack < 1) {
    return; // Not enough decoded frames to prune
  }

  /* (1) Find the last emitted word in the best path */
  int finalFrame = nDecodedFrames_ - nPrunedFrames_;
  const LexiconFreeDecoderState* bestNode =
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
