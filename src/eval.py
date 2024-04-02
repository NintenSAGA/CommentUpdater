import pathlib
import string

import jsonlines
import nltk.translate
import rouge
from nltk import edit_distance, word_tokenize


def tokenize(s: str):
    s = s.strip(string.punctuation)
    return s.split()


def evaluate_each(candidates, origin, reference, params=None):
    cand_tuples = []
    for hyp in candidates:
        data = {
            'origin': origin,
            'content': hyp,
        }
        if reference is not None:
            tokenized_ref = tokenize(reference)
            tokenized_hyp = tokenize(hyp)
            tokenized_origin = tokenize(origin)

            # Edit distance
            data['ed'] = edit_distance(tokenized_ref, tokenized_hyp)
            # Relative edit distance
            ed1 = edit_distance(tokenized_ref, tokenized_origin)
            if ed1 == 0:
                data['red'] = 0
            else:
                data['red'] = (data['ed'] / ed1)
            # GLEU metric
            data['gleu'] = 100 * nltk.translate.gleu_score.sentence_gleu([tokenized_ref], tokenized_hyp)
            # METEOR metric
            data['meteor'] = 100 * nltk.translate.meteor_score.meteor_score([tokenized_ref], tokenized_hyp)
            # ROUGE
            r = rouge.Rouge()
            try:
                r_score = r.get_scores(hyps=string.whitespace.join(tokenized_hyp),
                                       refs=string.whitespace.join(tokenized_ref))
                rl_score = r_score[0]['rouge-l']
                data['rouge-recall'] = 100 * rl_score['r']
                data['rouge-f1'] = 100 * rl_score['f']
            except ValueError:
                data['rouge-recall'] = 100
                data['rouge-f1'] = 100 if len(tokenized_hyp) == 0 else 0

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
