import nltk.translate.bleu_score
import rouge
from nltk import edit_distance

from eval import tokenize


def sort_by_evaluation_metric(candidates: list, metric_name: str, reverse: bool) -> list:
    return sorted(candidates, key=lambda cand: cand[metric_name], reverse=reverse)


def sort_by_rouge(candidates: list, rouge_metric_name: str) -> list:
    rr = rouge.Rouge()
    return sorted(candidates,
                  key=lambda cand:
                  rr.get_scores(cand['content'], cand['origin'])[0]['rouge-l'][rouge_metric_name]
                  , reverse=True)


def sort_by_levenshtein_distance(candidates: list) -> list:
    return sorted(candidates, key=lambda cand: edit_distance(tokenize(cand['origin']), tokenize(cand['content'])))


def sort_by_gleu(candidates: list) -> list:
    sf = nltk.translate.bleu_score.SmoothingFunction()
    return sorted(candidates,
                  key=lambda cand: nltk.translate.gleu_score
                  .sentence_gleu([tokenize(cand['origin'])], tokenize(cand['content']))
                  , reverse=True)
