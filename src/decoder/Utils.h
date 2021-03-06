/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include<limits>
#include <vector>
#include <string>
#include <iostream>
#include "src/lm/LM.h"
using namespace std; 

namespace w2l {

/* ===================== Definitions ===================== */

const double kNegativeInfinity = -std::numeric_limits<double>::infinity();
const int kLookBackLimit = 100;

enum class CriterionType {CTC = 1};

struct DecoderOptions {
  int beamSize; // Maximum number of hypothesis we hold after each step
  int beamSizeToken; // Maximum number of tokens we consider at each step
  double beamThreshold; // Threshold to prune hypothesis
  double lmWeight; // Weight of lm
  double wordScore; // Word insertion score
  double unkScore; // Unknown word insertion score
  double silScore; // Silence insertion score
  double eosScore; // Score for inserting an EOS
  bool logAdd; // If or not use logadd when merging hypothesis
  CriterionType criterionType; // CTC or ASG

  DecoderOptions(
      const int beamSize,
      const int beamSizeToken,
      const double beamThreshold,
      const double lmWeight,
      const double wordScore,
      const double unkScore,
      const double silScore,
      const double eosScore,
      const bool logAdd,
      const CriterionType criterionType)
      : beamSize(beamSize),
        beamSizeToken(beamSizeToken),
        beamThreshold(beamThreshold),
        lmWeight(lmWeight),
        wordScore(wordScore),
        unkScore(unkScore),
        silScore(silScore),
        eosScore(eosScore),
        logAdd(logAdd),
        criterionType(criterionType) {}

  DecoderOptions() {}
};

struct DecodeResult {
  double score;
  std::vector<int> words;
  string tokens;

  explicit DecodeResult(int length = 0)
      : score(0), words(length, -1), tokens(to_string(' ')) {}
};

/* ===================== Candidate-related operations ===================== */
template <typename T>
T log_sum_exp(const T &x, const T &y) {
  static T num_min = -std::numeric_limits<T>::max();
  if (x ==0.0) return y;
  if (y ==0.0) return x;
  T xmax = std::max(x, y);
  return std::log(std::exp(x - xmax) + std::exp(y - xmax)) + xmax;
}

template <class DecoderState>
void candidatesReset(
    double& candidatesBestScore,
    std::vector<DecoderState>& candidates,
    std::vector<DecoderState*>& candidatePtrs) {
  candidatesBestScore = kNegativeInfinity;
  candidates.clear();
  candidatePtrs.clear();
}

template <class DecoderState, class... Args>
void candidatesAdd(
    std::vector<DecoderState>& candidates,
    double& candidatesBestScore,
    const double beamThreshold,
    int n,
    bool insert,
    const double score,
    const double nb_score,
    const double b_score,
    string label,
    const Args&... args) {
  if (score >= candidatesBestScore) {
    candidatesBestScore = score;
  }
    if(insert)
    {
      label.append(to_string(-1));
      label.append(to_string(n));
    }
    candidates.emplace_back(score ,nb_score,b_score,label ,args...);
}

template <class DecoderState>
void candidatesStore(
    std::vector<DecoderState>& candidates,
    std::vector<DecoderState*>& candidatePtrs,
    std::vector<DecoderState>& outputs,
    const int beamSize,
    const double threshold,
    const bool logAdd,
    const bool returnSorted) {
  outputs.clear();
  if (candidates.empty()) {
    return;
  }

  /* 1. Select valid candidates */
  for (auto& candidate : candidates) {
      candidatePtrs.emplace_back(&candidate);
  }
  /* 2. Merge candidates */
  std::sort(
      candidatePtrs.begin(),
      candidatePtrs.end(),
      [](const DecoderState* node1, const DecoderState* node2) {
        int cmp = node1->compareNoScoreStates(node2);
        return cmp == 0 ? node1->score > node2->score : cmp > 0;
      });
  int nHypAfterMerging = 1;
  for (int i = 1; i < candidatePtrs.size(); i++) {
    if (candidatePtrs[i]->compareNoScoreStates(
            candidatePtrs[nHypAfterMerging - 1]) != 0) {
      // Distinct candidate
      candidatePtrs[nHypAfterMerging] = candidatePtrs[i];
      nHypAfterMerging++;
    } else {
      // Same candidate
      if (candidatePtrs[nHypAfterMerging-1]->nb_score==0)
      {
        candidatePtrs[nHypAfterMerging - 1]->nb_score=candidatePtrs[i]->nb_score;
      }
      else if(candidatePtrs[nHypAfterMerging-1]->nb_score!=0 && candidatePtrs[i]->nb_score!=0){
          candidatePtrs[nHypAfterMerging - 1]->nb_score = std::log(std::exp(candidatePtrs[nHypAfterMerging - 1]->nb_score)+std::exp(candidatePtrs[i]->nb_score));
      }
      if(candidatePtrs[nHypAfterMerging-1]->b_score==0)
      {
        candidatePtrs[nHypAfterMerging - 1]->b_score=candidatePtrs[i]->b_score;
      }
      else if(candidatePtrs[nHypAfterMerging-1]->b_score!=0 && candidatePtrs[i]->b_score!=0){
          candidatePtrs[nHypAfterMerging - 1]->b_score =  std::log(std::exp(candidatePtrs[nHypAfterMerging - 1]->b_score)+std::exp(candidatePtrs[i]->b_score));
      }
    }
  }
  candidatePtrs.resize(nHypAfterMerging);
  for(int i=0;i<candidatePtrs.size();i++){
        if(candidatePtrs[i]->nb_score==0 && candidatePtrs[i]->b_score==0)
        {
          
          candidatePtrs[i]->score=0;
          
        }
        else if(candidatePtrs[i]->b_score==0)
        {
          candidatePtrs[i]->score=candidatePtrs[i]->nb_score;
        }
        else if(candidatePtrs[i]->nb_score==0)
        {
          candidatePtrs[i]->score=candidatePtrs[i]->b_score;
        }
        
        else
        {
          candidatePtrs[i]->score=std::log(std::exp(candidatePtrs[i]->nb_score)+std::exp(candidatePtrs[i]->b_score));
        }
  }
  /* 3. Sort and prune */
  auto compareNodeScore = [](const DecoderState* x,
                             const DecoderState* y) {
    if (x->score == y->score) {
    if (x->token == y->token) {
      return false;
    } else {
      return (x->token < y->token);
    }
  } else {
    return x->score > y->score;
  }
  };

  int nValidHyp = candidatePtrs.size();
  int finalSize = std::min(nValidHyp, beamSize);
  if (!returnSorted && nValidHyp > beamSize) {
    std::nth_element(
        candidatePtrs.begin(),
        candidatePtrs.begin() + finalSize,
        candidatePtrs.begin() + nValidHyp,
        compareNodeScore);
  } else if (returnSorted) {
    std::partial_sort(
        candidatePtrs.begin(),
        candidatePtrs.begin() + finalSize,
        candidatePtrs.begin() + nValidHyp,
        compareNodeScore);
  }

  for (int i = 0; i < finalSize; i++) {
    outputs.emplace_back(std::move(*candidatePtrs[i]));
  }
}

/* ===================== Result-related operations ===================== */

template <class DecoderState>
DecodeResult getHypothesis(const DecoderState* node, const int finalFrame) {
  const DecoderState* node_ = node;
  if (!node_) {
    return DecodeResult();
  }

  DecodeResult res(finalFrame + 1);
  res.score = node_->score;

  int i = 0;
  while (node_) {
    res.words[finalFrame - i] = node_->getWord();
    node_ = node_->parent;
    i++;
  }
  res.tokens=node->label;
  return res;
}

template <class DecoderState>
std::vector<DecodeResult> getAllHypothesis(
    const std::vector<DecoderState>& finalHyps,
    const int finalFrame) {
  int nHyp = finalHyps.size();

  std::vector<DecodeResult> res(nHyp);

  for (int r = 0; r < nHyp; r++) {
    const DecoderState* node = &finalHyps[r];
    res[r] = getHypothesis(node, finalFrame);
  }

  return res;
}

template <class DecoderState>
const DecoderState* findBestAncestor(
    const std::vector<DecoderState>& finalHyps,
    int& lookBack) {
  int nHyp = finalHyps.size();
  if (nHyp == 0) {
    return nullptr;
  }

  double bestScore = finalHyps.front().score;
  const DecoderState* bestNode = finalHyps.data();
  for (int r = 1; r < nHyp; r++) {
    const DecoderState* node = &finalHyps[r];
    if (node->score > bestScore) {
      bestScore = node->score;
      bestNode = node;
    }
  }

  int n = 0;
  while (bestNode && n < lookBack) {
    n++;
    bestNode = bestNode->parent;
  }

  const int maxLookBack = lookBack + kLookBackLimit;
  while (bestNode) {
    // Check for first emitted word.
    if (bestNode->isComplete()) {
      break;
    }

    n++;
    bestNode = bestNode->parent;

    if (n == maxLookBack) {
      break;
    }
  }

  lookBack = n;
  return bestNode;
}

template <class DecoderState>
void pruneAndNormalize(
    std::unordered_map<int, std::vector<DecoderState>>& hypothesis,
    const int startFrame,
    const int lookBack) {
  /* 1. Move things from back of hypothesis to front. */
  for (int i = 0; i < hypothesis.size(); i++) {
    if (i <= lookBack) {
      hypothesis[i].swap(hypothesis[i + startFrame]);
    } else {
      hypothesis[i].clear();
    }
  }

  /* 2. Avoid further back-tracking */
  for (DecoderState& hyp : hypothesis[0]) {
    hyp.parent = nullptr;
  }

  /* 3. Avoid score underflow/overflow. */
  double largestScore = hypothesis[lookBack].front().score;
  for (int i = 1; i < hypothesis[lookBack].size(); i++) {
    if (largestScore < hypothesis[lookBack][i].score) {
      largestScore = hypothesis[lookBack][i].score;
    }
  }

  for (int i = 0; i < hypothesis[lookBack].size(); i++) {
    hypothesis[lookBack][i].score -= largestScore;
  }
}
/* ===================== LM-related operations ===================== */

template <class DecoderState>
void updateLMCache(const LMPtr& lm, std::vector<DecoderState>& hypothesis) {
  // For ConvLM update cache
  std::vector<LMStatePtr> states;
  for (const auto& hyp : hypothesis) {
    states.emplace_back(hyp.lmState);
  }
  lm->updateCache(states);
}

} // namespace w2l
