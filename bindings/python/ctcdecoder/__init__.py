#!/usr/bin/env python3

__all__ = ["common", "decoder","decoderfree"]
def merge_ctc_output(results,sil_idx,blank_idx,token_dict,no_predictions)->list:
    merged_results = []
    for i in range(min(no_predictions, len(results))):
                    prediction = str()
                    positions = []
                    prev_token = -1
                    silence = False
                    pos = 0
                    for idx in results[i].tokens:
                        if idx != -1 and idx != prev_token and (idx != sil_idx or silence == False):
                            if idx != blank_idx and idx != sil_idx:
                                prediction=prediction+token_dict.get_entry(idx)
                                positions.append(pos)
                                prev_token = idx
                                silence = False
                            elif idx == sil_idx:
                                #prediction=prediction+" "
                                silence = True
                                #positions.append(pos)
                            elif idx == blank_idx:
                                prev_token = idx
                                silence = False
                        if idx != -1 and idx != prev_token and silence and idx ==sil_idx:
                            prediction=prediction+" "
                            prev_token = idx
                            positions.append(pos)
                            silence = False
                        pos += 1
                    scores = results[i].score
                    merged_result = (scores,prediction)
                    merged_results.append(merged_result)
    return merged_results
