import difflib
import re

import javalang
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
    try:
        # Create a TF-IDF Vectorizer
        vectorizer = TfidfVectorizer()

        # Transform texts to TF-IDF vectors
        tfidf_matrix = vectorizer.fit_transform([text1, text2])

        # Calculate Cosine Similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

        return similarity
    except ValueError as e:
        return 0


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


def split_pascal_case(input_string):
    # Insert a space before each uppercase letter that is
    # preceded by a lowercase letter or another uppercase letter in a sequence (to handle acronyms)
    formatted_string = re.sub('((?<=[a-z])[A-Z]|(?<=[A-Z])[A-Z](?=[a-z]))', r' \1', input_string)
    # Split the string based on spaces added
    return formatted_string.split()


def get_identifier_set(method_body):
    try:
        tokens = list(javalang.tokenizer.tokenize(method_body))
        identifier_set = set()
        for token in tokens:
            if type(token) is javalang.tokenizer.Identifier:
                identifier_set.update(split_pascal_case(token.value))
        return identifier_set
    except Exception as e:
        return set(method_body)


def get_token_str(old_method, new_method):
    s1 = get_identifier_set(new_method)
    s2 = get_identifier_set(old_method)
    s3 = s1 - s2
    return ' '.join(s3)


def calculate_coverage_ratio(word_set, sentence):
    # 将句子分割成单词，这里使用空格分割，并转换为小写以匹配单词集合
    # 可根据需要调整为更复杂的分词方法
    words_in_sentence = sentence.lower().split()

    # 使用集合操作计算句子中单词与集合中单词的交集
    intersection = set(words_in_sentence) & set(word_set)

    # 计算交集中单词数量占单词集合总数的比例
    if len(word_set) == 0:
        return float(0)  # 避免除以零的错误
    coverage_ratio = float(len(intersection)) / float(len(word_set))

    return coverage_ratio


