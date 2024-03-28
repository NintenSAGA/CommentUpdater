import pathlib

import jsonlines
import nltk.translate
from nltk import edit_distance, word_tokenize


def evaluate_each(candidates, origin, reference, params=None):
    cand_tuples = []
    for hyp in candidates:
        data = {
            'content': hyp,
        }
        if reference is not None:
            tokenized_ref = word_tokenize(reference)
            tokenized_hyp = word_tokenize(hyp)
            tokenized_origin = word_tokenize(origin)

            # Edit distance
            data['ed'] = edit_distance(tokenized_ref, tokenized_hyp)
            # Relative edit distance
            data['red'] = data['ed'] / edit_distance(tokenized_ref, tokenized_origin)
            # GLEU metric
            data['gleu'] = 100 * nltk.translate.gleu_score.sentence_gleu([tokenized_ref], tokenized_hyp)
            # METEOR metric
            data['meteor'] = 100 * nltk.translate.meteor_score.meteor_score([tokenized_ref], tokenized_hyp)

        cand_tuples.append(data)

    return cand_tuples


def evaluate(jsonl_path: pathlib.Path | str, result_field: str = None):
    result_array = []
    with jsonlines.open(jsonl_path, mode='r') as reader:
        for obj in reader:
            if obj is None:
                continue
            origin = obj['Origin']
            reference = obj['Reference']
            if result_field is None:
                candidates = obj['LLMCandidates']
            else:
                candidates = [obj[result_field]]
            cand_tuples = evaluate_each(candidates, origin, reference)
            result_array.append(cand_tuples if result_field is None else cand_tuples[0])
    return result_array