import difflib

import rouge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def calculate_diff(original: str, updated: str):
    matcher = difflib.SequenceMatcher(lambda x: x == " ", original, updated)
    print(matcher.get_opcodes())
    modifications = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != 'equal':
            # Add the modification pair (old, new) to the list
            modifications.append((original[i1:i2], updated[j1:j2]))

    return modifications


def cal_cosine_similarity(text1, text2):
    # Create a TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()

    # Transform texts to TF-IDF vectors
    tfidf_matrix = vectorizer.fit_transform([text1, text2])

    # Calculate Cosine Similarity
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    return similarity


def calc_and_filter(candidates, dst_method, src_javadoc, m, p):
    _rouge = rouge.Rouge()
    scores = _rouge.get_scores(hyps=candidates,
                               refs=[src_javadoc] * len(candidates))
    cand_tuples = []
    for _i, score in enumerate(scores):
        hyp = candidates[_i]
        recall = score['rouge-l']['r']
        overall = score['rouge-l']['f']
        cs = cal_cosine_similarity(dst_method, hyp)
        cand_tuples.append((hyp, recall, cs, overall))

    cand_tuples = sorted(cand_tuples, key=lambda x: len(x[0]))
    cand_tuples = sorted(cand_tuples, key=lambda x: x[2], reverse=True)
    cand_tuples = list(filter(lambda x: x[3] < 0.99, cand_tuples))
    # cand_tuples = cand_tuples[:min(m, len(cand_tuples))]
    cand_tuples = sorted(cand_tuples, key=lambda x: x[1], reverse=True)
    cand_tuples = cand_tuples[:min(len(cand_tuples), p)]

    for t in cand_tuples:
        print(f'recall: {t[1]:.2f} cs: {t[2]:.2f} f1: {t[3]: .2f} \n\t- {t[0]}')

    if len(cand_tuples) == 0:
        print('No candidates found')
        return candidates[:min(len(candidates), p)]

    return list(map(lambda x: x[0], cand_tuples))
