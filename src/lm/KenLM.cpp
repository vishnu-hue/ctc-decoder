/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "KenLM.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <valarray>
#include <lm/model.hh>
#include "lm/config.hh"
#include "lm/model.hh"
#include "lm/state.hh"

#include "util/string_piece.hh"
#include "util/tokenize_piece.hh"

using namespace std;
namespace w2l {

KenLM::KenLM(const std::string& path, const Dictionary& usrTknDict, const bool create) {
  // Load LM
  size_t max_order_=0;
  std::vector<std::string> vocabulary_;
  RetriveStrEnumerateVocab enumerate;
  lm::ngram::Config config;
  config.enumerate_vocab = &enumerate;
  model_.reset(lm::ngram::LoadVirtual(path.c_str(), config));
  if (!model_) {
    throw std::runtime_error("[KenLM] LM loading failed.");
  }
  max_order_ = static_cast<std::shared_ptr<lm::base::Model>>(model_)->Order();
  vocabulary_ = enumerate.vocabulary;
  /*
  if create is set a new words.lst file will be created which can be used to create a word dict.
  this can be used when a proper words.lst file is not available for the lm model
  but the lm should be loaded again with the new word dict created 
  */
  if(create){
  ofstream file;
  file.open("words.lst");
  for (size_t i = 0; i < vocabulary_.size(); ++i) {
    file<<vocabulary_[i]<<" ";
    for(size_t j=0; j<vocabulary_[i].size();j++){
      if (vocabulary_[i][j] == '<'){
        break;
      }
      file<<vocabulary_[i][j]<<" ";
    }
    file<<"|\n";
  }
  file.close();
  }
  vocab_ = &model_->BaseVocabulary();
  if (!vocab_) {
    throw std::runtime_error("[KenLM] LM vocabulary loading failed.");
  }
  
  // Create index map
  usrToLmIdxMap_.resize(usrTknDict.indexSize());
  for (int i = 0; i < usrTknDict.indexSize(); i++) {
    auto token = usrTknDict.getEntry(i);
    int lmIdx = vocab_->Index(token.c_str());
    usrToLmIdxMap_[i] = lmIdx;
  }
}

LMStatePtr KenLM::start(bool startWithNothing) {
  auto outState = std::make_shared<KenLMState>();
  if (startWithNothing) {
    model_->NullContextWrite(outState->ken());
  } else {
    model_->BeginSentenceWrite(outState->ken());
  }

  return outState;
}

std::pair<LMStatePtr, float> KenLM::score(
    const LMStatePtr& state,
    const int usrTokenIdx) {
  if (usrTokenIdx < 0 || usrTokenIdx >= usrToLmIdxMap_.size()) {
    throw std::runtime_error(
        "[KenLM] Invalid user token index: " + std::to_string(usrTokenIdx));
  }
  auto inState = std::static_pointer_cast<KenLMState>(state);
  auto outState = inState->child<KenLMState>(usrTokenIdx);
  float score = model_->BaseScore(
      inState->ken(), usrToLmIdxMap_[usrTokenIdx], outState->ken());
  return std::make_pair(std::move(outState), score);
}

std::pair<LMStatePtr, float> KenLM::finish(const LMStatePtr& state) {
  auto inState = std::static_pointer_cast<KenLMState>(state);
  auto outState = inState->child<KenLMState>(-1);
  float score =
      model_->BaseScore(inState->ken(), vocab_->EndSentence(), outState->ken());
  return std::make_pair(std::move(outState), score);
}

} // namespace w2l
