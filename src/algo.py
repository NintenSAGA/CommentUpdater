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


def calc_rouge_l(hyp_list: list[str], ref: str):
    r = rouge.Rouge()
    scores = r.get_scores(hyps=hyp_list, refs=[ref] * len(hyp_list))
    return list(
        map(lambda score: {
            'recall': score['rouge-l']['r'],
            'precision': score['rouge-l']['p'],
            'f1': score['rouge-l']['f'],
        },
            scores)
    )


def calc_and_filter(candidates, dst_method, src_javadoc, params: dict, exp_javadoc=None, silent=False):
    rouge_result = calc_rouge_l(candidates, src_javadoc)
    cand_tuples = []

    for i, rouge_element in enumerate(rouge_result):
        hyp = candidates[i]
        # 计算 Method Body 和 Comment 的 Cosine Similarity
        cs = cal_cosine_similarity(dst_method, hyp)
        # 如果有预期结果，计算结果准确度
        accuracy = -1
        if exp_javadoc is not None:
            r = calc_rouge_l([hyp], exp_javadoc)[0]
            accuracy = r['recall']

        cand_tuples.append({
            'content': hyp,
            'recall': rouge_element['recall'],
            'precision': rouge_element['precision'],
            'overall': rouge_element['f1'],
            'cosine': cs,
            'accuracy': accuracy
        })

    # 筛除跟原注释完全一样的结果
    cand_tuples = list(filter(lambda x: x['overall'] < 0.99, cand_tuples))
    # 根据 Cosine Similarity 降序排列，越靠前的与方法本身越相关
    cand_tuples = sorted(cand_tuples, key=lambda x: x['cosine'], reverse=True)
    # 保留前 nr_cand1 位，其余淘汰
    nr_cand1 = params['nr_cand1']
    cand_tuples = cand_tuples if len(cand_tuples) <= nr_cand1 else cand_tuples[:nr_cand1]
    # 根据 Rouge Metric 降序排列，越靠前的与原注释越相似
    # cand_tuples = sorted(cand_tuples, key=lambda x: x['recall'], reverse=True)
    cand_tuples = sorted(cand_tuples, key=lambda x: x['overall'], reverse=True)
    # 保留前 nr_cand 位，其余淘汰
    nr_cand = params['nr_cand']
    cand_tuples = cand_tuples if len(cand_tuples) <= nr_cand else cand_tuples[:nr_cand]

    if not silent:
        for t in cand_tuples:
            print(f'''❇️ recall: {t["recall"]:.2f} cs: {t["cosine"]:.2f} f1: {t["overall"]: .2f}
❇️ accuracy: {t["accuracy"]:.2f}
🛑 - {t["content"]}''')

    if len(cand_tuples) == 0:
        if not silent:
            print('No candidates found')
        return candidates[:min(len(candidates), nr_cand)]

    return list(map(lambda x: x['content'], cand_tuples))
