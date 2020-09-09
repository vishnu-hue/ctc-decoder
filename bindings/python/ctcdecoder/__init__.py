#!/usr/bin/env python3

__all__ = ["common", "decoder","decoderfree"]
def merge_ctc_outputs(results,sil_idx,blank_idx,token_dict,no_predictions)->list:
    merged_results = []
    positions = []
    for i in range(min(no_predictions, len(results))):
                    prediction = str()
                    position = []
                    prev_token = -1
                    silence = False
                    pos = 0
                    for idx in results[i].tokens:
                        if idx != -1 and idx != prev_token and (idx != sil_idx or silence == False):
                            if idx != blank_idx and idx != sil_idx:
                                if type(token_dict) != list:
                                    prediction=prediction+token_dict.get_entry(idx)
                                else:
                                    prediction=prediction+token_dict[idx]
                                position.append(pos)
                                prev_token = idx
                                silence = False
                            elif idx == sil_idx:
                                silence = True
                            elif idx == blank_idx:
                                prev_token = idx
                                silence = False
                        if idx != -1 and idx != prev_token and silence and idx == sil_idx:
                            prediction=prediction+" "
                            prev_token = idx
                            position.append(pos)
                            silence = False
                        pos += 1
                    scores = results[i].score
                    merged_result = (scores,prediction)
                    merged_results.append(merged_result)
                    positions.append(position)
    return merged_results, positions
