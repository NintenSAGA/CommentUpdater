import nltk.translate.bleu_score
import rouge
from nltk import edit_distance

from eval import tokenize


def sort_by_evaluation_metric(candidates: list, metric_name: str, reverse: bool) -> list:
    return sorted(candidates, key=lambda cand: cand[metric_name], reverse=reverse)


def sort_by_rouge(candidates: list, rouge_metric_name: str) -> list:
    n = len(candidates)
    origin = candidates[0]['origin']
    rr = rouge.Rouge()
    r = rr.get_scores(hyps=list(map(lambda x: x['content'], candidates)), refs=[origin] * n)
    for i in range(n):
        candidates[i]['rouge'] = r[i]['rouge-l'][rouge_metric_name]
    return sorted(candidates, key=lambda cand: cand['rouge'], reverse=True)


def sort_by_levenshtein_distance(candidates: list) -> list:
    return sorted(candidates, key=lambda cand: edit_distance(tokenize(cand['origin']), tokenize(cand['content'])))


def sort_by_gleu(candidates: list) -> list:
    sf = nltk.translate.bleu_score.SmoothingFunction()
    return sorted(candidates,
                  key=lambda cand: nltk.translate.bleu_score
                  .sentence_bleu([tokenize(cand['origin'])], tokenize(cand['content']),
                                 smoothing_function=sf.method5)
                  , reverse=True)
